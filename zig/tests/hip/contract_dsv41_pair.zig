//! Host contract check for the dsv41 decode linear: gate 4's stress sums, bound and margins on host buffers.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const kit = @import("contract_kit.zig");
const dsv41_k = @import("contract_dsv41_exl3.zig");
const dsv41 = hip.launches.dsv41;

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

test "HIP pair contract: dsv41 exl3 stress sums within their bound in every order" {
    const a = std.testing.allocator;
    const shape = c.Shape{ .rows = 1, .n = 5120, .k = 4096 }; // gate 4's shape: the largest served k
    var e: dsv41_k.Exl3 = .{
        .rig = undefined,
        .l = undefined,
        .k = .bf16,
        .arena = std.heap.ArenaAllocator.init(a),
    };
    defer e.arena.deinit();
    const al = e.arena.allocator();

    // prepare's stress planting, host-only: the same bytes the gate plants on the device
    e.n = shape.n;
    e.kk = shape.k;
    const half_bits: u32 = 8;
    const tile_k = 16;
    const tw: usize = 4 * half_bits;
    const stride_k: usize = 8 * tw;
    e.stride_nb = @divExact(shape.k, tile_k) * stride_k;
    const nb = shape.n / 128;
    const xn = shape.rows * shape.k;
    const xs = try al.alloc(u8, xn * 2);
    for (0..xn) |i| {
        const v: f64 = @floatFromInt(1 + (i % 7));
        kit.put(.f16, xs, i, kit.round(.f16, v));
    }
    for (0..xn) |i| {
        if (i % 29 == 0) kit.put(.f16, xs, i, 0); // a lane's zero must enter as +0, never skipped
    }
    const words = try al.alloc(u32, nb * e.stride_nb + stride_k);
    for (words, 0..) |*w, i| w.* = 0x9e37_79b9 *% (@as(u32, @intCast(i)) +% 0x85eb_ca6b);
    const svh = try al.alloc(u8, shape.n * 2);
    for (0..shape.n) |i| kit.put(.f16, svh, i, kit.round(.f16, 0.02 + @as(f64, @floatFromInt(i % 97)) / 4096));
    const bias = try al.alloc(u8, shape.n * 2);
    for (0..shape.n) |i| kit.put(.f16, bias, i, kit.round(.f16, @as(f64, @floatFromInt(i % 131)) / 8192 - 0.008));
    const p = dsv41.plan(shape.k, shape.n);
    e.sk = @intCast(p.sk);
    e.xs = xs;
    e.words = words;
    e.svh = svh;
    e.bias = bias;
    e.out_len = shape.rows * shape.n;

    const n = e.count();
    const want = try al.alloc(f64, n);
    const mag = try al.alloc(f64, n);
    try e.reference(want, mag);
    const spec = dsv41_k.spec(.bf16);
    const allow = try al.alloc(f64, n);
    for (allow, want, mag) |*l, w, m| l.* = c.bound(spec, shape, w, m);

    const got = try al.alloc(f64, n);
    var margin = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |o| {
        try e.simulate(.f32, o, got);
        var worst: f64 = 0;
        var at: usize = 0;
        for (got, want, allow, 0..) |gv, wv, lv, i| {
            const miss = if (gv == wv) 0 else @abs(gv - wv) / lv;
            if (miss > worst) {
                worst = miss;
                at = i;
            }
        }
        std.debug.print("dsv41 f32 {s}: worst miss {d:.4}x at {d}: want {e}, got {e}, mag {e}, allow {e}, depth {d}\n", .{
            @tagName(o), worst, at, want[at], got[at], mag[at], allow[at], spec.depth(shape),
        });
        try std.testing.expect(worst <= 1);
        for ([_]c.Accum{ .f16, .bf16 }, &margin) |acc, *m| {
            try e.simulate(acc, o, got);
            var most: f64 = 0;
            for (got, want, allow) |gv, wv, lv| {
                const miss = if (gv == wv) 0 else if (std.math.isFinite(gv) or !std.math.isFinite(wv)) @abs(gv - wv) / lv else std.math.inf(f64);
                most = @max(most, miss);
            }
            m.* = @min(m.*, most);
        }
    }
    // the spec's caps are 0: the fwht finish couples every accum kind to one amplified bound, margins stay near 1

    // only the fp32 chain's own bound is asserted above
    std.debug.print("dsv41 margins: fp16 {d:.1}x, bf16 {d:.1}x\n", .{ margin[0], margin[1] });
}
