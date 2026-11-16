//! The dsv41 decode EXL3 linear: glinear's split-K tiles and rot_many's rotation, in the kernels' fixed order.

const std = @import("std");
const abi = @import("../abi.zig");
const runtime = @import("../runtime.zig");
const hl = @import("../launch.zig");
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");

const Error = runtime.Error;

/// The layers one launch takes (abi.h's TF_DS_EXL3_GMAX).
pub const gmax = 8;

/// abi.h's tf_ds_exl3_glayer, field for field.
pub const GLayer = extern struct {
    xh: u64 = 0,
    T: u64 = 0,
    svh: u64 = 0,
    bias: u64 = 0,
    y: u64 = 0,
    Z: u64 = 0,
    counters: u64 = 0,
    ldx: i64 = 0,
    ldy: i64 = 0,
    stride_k: i64 = 0,
    stride_nb: i64 = 0,
    y_dtype: c_int = 0,
    K: c_int = 0,
    N: c_int = 0,
    SK: c_int = 0,
    first: c_int = 0,
    pad0: c_int = 0,
    rsuh: u64 = 0,
    rxh: u64 = 0,
    ldr: i64 = 0,
    roff: c_int = 0,
    pad1: c_int = 0,
    rcos: u64 = 0,
    rsin: u64 = 0,
    rpos: u64 = 0,
    rhd: c_int = 0,
    rrd: c_int = 0,
};

/// abi.h's tf_ds_exl3_gargs: the group's layers, their count, the row window and the L2 hint.
pub const Arg = extern struct {
    l: [gmax]GLayer = @splat(.{}),
    n: c_int = 0,
    M: c_int = 0,
    discard: c_int = 0,
    pad0: c_int = 0,
};

/// abi.h's tf_ds_exl3_rlayer.
pub const RLayer = extern struct {
    x: u64 = 0,
    suh: u64 = 0,
    xh: u64 = 0,
    ldx: i64 = 0,
    x_dtype: c_int = 0,
    K: c_int = 0,
    first: c_int = 0,
    pad0: c_int = 0,
};

/// abi.h's tf_ds_exl3_rargs.
pub const RotArg = extern struct {
    l: [gmax]RLayer = @splat(.{}),
    n: c_int = 0,
    M: c_int = 0,
};

comptime {
    // the sizes the kernels were compiled with (abi.h asserts the same)
    std.debug.assert(@sizeOf(GLayer) == 176 and @sizeOf(Arg) == 1424);
    std.debug.assert(@sizeOf(RLayer) == 48 and @sizeOf(RotArg) == 392);
}

/// The half-bit widths an EXL3 tile instantiates with (dsv41/exl3.hip's TF_DS_EXL3_GLINEAR list).
pub const k2s = [_]c_int{ 4, 6, 8, 10, 12, 16 };
const wks = [_]c_int{ 4, 8 };

fn k2Index(k2: c_int) ?usize {
    for (k2s, 0..) |w, i| if (w == k2) return i;
    return null;
}

/// SK splits and WK waves: narrow splits K to fill about 192 blocks; wide keeps one split, eight waves, tiles bound.
pub fn plan(kk: usize, n: usize) struct { sk: c_int, wk: c_int } {
    const blocks = 192;
    const min_tiles = 8;
    const kt = kk / 16;
    const nb = n / 128;
    var sk: usize = 1;
    var wk: usize = if (nb >= 64) 8 else 4;
    if (nb < 64) {
        while (nb * sk < blocks and sk < 64) sk *= 2;
    }
    while ((wk > 4 or sk > 1) and (kt % (sk * wk) != 0 or kt / (sk * wk) < min_tiles)) {
        if (wk > 4) {
            wk = 4;
        } else if (sk > 1) {
            sk /= 2;
        } else break;
    }
    return .{ .sk = @intCast(sk), .wk = @intCast(wk) };
}

/// The words between k tiles and between 128-column blocks of a layer's strips-layout weights.
pub fn strides(kk: usize, k2: c_int) [2]i64 {
    const tw: i64 = 4 * @as(i64, @intCast(k2));
    return .{ 8 * tw, @divExact(@as(i64, @intCast(kk)), 16) * 8 * tw };
}

/// The glinear symbols in the k2s x wks order of the tables above.
const glinear_names = [_][wks.len][:0]const u8{
    .{ "tf_ds_exl3_glinear_k4_w4", "tf_ds_exl3_glinear_k4_w8" },
    .{ "tf_ds_exl3_glinear_k6_w4", "tf_ds_exl3_glinear_k6_w8" },
    .{ "tf_ds_exl3_glinear_k8_w4", "tf_ds_exl3_glinear_k8_w8" },
    .{ "tf_ds_exl3_glinear_k10_w4", "tf_ds_exl3_glinear_k10_w8" },
    .{ "tf_ds_exl3_glinear_k12_w4", "tf_ds_exl3_glinear_k12_w8" },
    .{ "tf_ds_exl3_glinear_k16_w4", "tf_ds_exl3_glinear_k16_w8" },
};

pub const Fns = struct {
    /// glinear by half-bit width and waves a program, and rot_many.
    glinear: [k2s.len][wks.len]Function,
    rot_many: Function,

    pub fn load(m: Module) Error!Fns {
        var g: [k2s.len][wks.len]Function = undefined;
        for (glinear_names, 0..) |pair, i| for (pair, 0..) |name, j| {
            g[i][j] = try m.function(name);
        };
        return .{ .glinear = g, .rot_many = try m.function("tf_ds_exl3_rot_many") };
    }
};

fn cdiv(n: anytype, by: anytype) u32 {
    return @intCast(@divTrunc(@as(i64, @intCast(n)) + by - 1, by));
}

fn refuse(what: []const u8) Error {
    std.log.err("dsv41 {s}: shape", .{what});
    return error.Invalid;
}

/// Blocks first_i .. are column blocks times splits; items repeats the grid; shared is WK * min(M, 8) * 128 floats.
pub fn glinearGo(k: *const Fns, r: *const runtime.Runtime, a: Arg, k2: c_int, items: c_int, s: abi.Stream) Error!void {
    if (a.n < 1 or a.M < 1) return refuse("glinear");
    const b = k2Index(k2) orelse return refuse("glinear k2");
    var blocks: c_int = 0;
    const wk = plan(@intCast(a.l[0].K), @intCast(a.l[0].N)).wk;
    for (0..@intCast(a.n)) |i| {
        const l = a.l[i];
        if (@rem(l.N, 128) != 0 or @rem(l.K, 16) != 0) return refuse("glinear tile");
        // same-plan layers share a launch (the fork's protocol); a mixed-plan group is invalid
        if (plan(@intCast(l.K), @intCast(l.N)).wk != wk) return refuse("glinear plan");
        blocks += @divTrunc(l.N, 128) * l.SK;
    }
    const wi: usize = if (wk == 8) 1 else 0;
    const f = k.glinear[b][wi];
    var args: hl.Args = .{};
    try args.add(a);
    const smem: u32 = @intCast(wk * @min(@as(usize, @intCast(a.M)), 8) * 128 * 4);
    if (s == util.counting) return;
    try hl.launch(f, .{
        .grid = .{ .x = @intCast(blocks * items), .y = 1, .z = 1 },
        .block = .{ .x = @intCast(wk * 32), .y = 1, .z = 1 },
        .shared = smem,
    }, .{ .r = r, .handle = s }, &args);
}

/// rot_many over the RotArg's layers: a wave a (layer, row, 128-column block), four waves a program of 128 threads.
pub fn rotManyGo(k: *const Fns, r: *const runtime.Runtime, a: RotArg, items: c_int, s: abi.Stream) Error!void {
    if (a.n < 1 or a.M < 1) return refuse("rot_many");
    var waves: c_int = 0;
    for (0..@intCast(a.n)) |i| {
        const l = a.l[i];
        if (@rem(l.K, 128) != 0) return refuse("rot_many tile");
        waves += a.M * @divTrunc(l.K, 128);
    }
    var args: hl.Args = .{};
    try args.add(a);
    if (s == util.counting) return;
    try hl.launch(k.rot_many, .{
        .grid = .{ .x = cdiv(waves * items, 4), .y = 1, .z = 1 },
        .block = .{ .x = 128, .y = 1, .z = 1 },
    }, .{ .r = r, .handle = s }, &args);
}

test "the decode linear plans and strides as the kernels expect" {
    // a wide layer keeps one split and eight waves; a narrow one splits K to about 192 blocks, then tiles trim it
    try std.testing.expectEqual(@as(c_int, 8), plan(1280, 32768).wk);
    try std.testing.expectEqual(@as(c_int, 1), plan(1280, 32768).sk);
    const q = plan(5120, 1280);
    try std.testing.expectEqual(@as(c_int, 4), q.wk);
    try std.testing.expect(q.sk > 1 and (5120 / 16) % (@as(usize, @intCast(q.sk)) * @as(usize, @intCast(q.wk))) == 0);
    const s = strides(1280, 8);
    try std.testing.expectEqual(@as(i64, 256), s[0]);
    try std.testing.expectEqual(@as(i64, 20480), s[1]);
}
