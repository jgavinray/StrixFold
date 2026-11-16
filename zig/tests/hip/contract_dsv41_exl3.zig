//! The dsv41 decode EXL3 linear under the HIP contract: fp16 rows over MUL1-decoded tiles, fixed-order dot2 sums.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const dsv41 = hip.launches.dsv41;

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Kernels = hip.launches.dsv41.Fns;
/// launches/dsv41.zig's Arg, as glinearGo takes it.
pub const Arg = @typeInfo(@TypeOf(hip.launches.dsv41.glinearGo)).@"fn".param_types[2].?;
/// launches/dsv41.zig's RotArg, as rotManyGo takes it.
pub const RotArg = @typeInfo(@TypeOf(hip.launches.dsv41.rotManyGo)).@"fn".param_types[2].?;

const half_bits: u32 = 8; // the EXL3 tile's half-bit width the served layers carry
const tile_k = 16; // the k rows of one 16 x 16 tile
/// The longest split-K a served decode layer takes, by dsv41.plan at the recorded shapes (8-way on 4096 x 5120).
const max_splits = 8;

/// E(p): one past the last stream bit of value p's window (exl3.hip stream_end<K2>).
fn streamEnd(k2: u32, p: u32) u32 {
    return if (k2 & 1 != 0) ((p + 1) * k2 - ((p + 1) & 1)) / 2 else (p + 1) * (k2 / 2);
}

/// Tile value p as LaneWin reads it: MSB-first bits, the last 16 ending at E(p), wrapped mod the tile's bits.
fn tileBits(words: []const u32, stride_nb: usize, nb: usize, kt: usize, t: usize, p: u32) u16 {
    const tw: u32 = 4 * half_bits;
    const base: usize = nb * stride_nb + kt * (8 * tw) + t * tw;
    const bits: u32 = 128 * half_bits;
    const e = streamEnd(half_bits, p) % bits;
    var v: u16 = 0;
    for (0..16) |i| {
        const bit = (bits + e - 16 + @as(u32, @intCast(i))) % bits;
        const word = words[base + bit / 32];
        const on = (word >> @as(u5, @intCast(31 - bit % 32))) & 1;
        v = (v << 1) | @as(u16, @intCast(on));
    }
    return v;
}

/// MUL1's exact decode: fp16(0x6400 + byte sum of w * 0x83DCD12D mod 2^32), then one fp16 fma by 0x1EEE plus 0xC931.
fn mul1Value(w: u16) f64 {
    const h: u32 = @truncate(@as(u32, w) *% 0x83DC_D12D);
    const bytesum = (h & 0xff) + ((h >> 8) & 0xff) + ((h >> 16) & 0xff) + (h >> 24);
    const bits: u16 = @truncate(0x6400 + bytesum);
    return kit.round(.f16, kit.get(.f16, std.mem.asBytes(&bits), 0) *
        kit.get(.f16, std.mem.asBytes(&@as(u16, 0x1EEE)), 0) +
        kit.get(.f16, std.mem.asBytes(&@as(u16, 0xC931)), 0));
}

/// Longest fp32 chain, conservatively: four dot2 roundings plus one half-wave add per tile, SK splits, fwht levels.
fn depth(s: c.Shape) usize {
    return s.k / tile_k * 4 + s.k / tile_k + max_splits - 1 + 8 + 3;
}

/// `v` rounded to the accumulator kind, as the simulated sums round their terms.
fn roundAcc(acc: c.Accum, v: f64) f64 {
    return switch (acc) {
        .f32 => @as(f32, @floatCast(v)),
        .f16 => @as(f16, @floatCast(v)),
        .bf16 => c.fromBf16(c.toBf16(@floatCast(v))),
        .none => v,
    };
}

const names = [2][]const u8{
    "hip dsv41 exl3_glinear (fp16)",
    "hip dsv41 exl3_glinear (bf16)",
};
const sums_why = "the fp32 dot2 chain is not exact (fp16-decoded weights), and the fwht finish mixes all 128 columns' sums into every output, so the bound carries the block's amplified magnitude and the fp16/bf16 margins against it stay near 1";

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        // the HAD_SCALE multiply, svh's product and the bias' fma after the sums
        .ops = 3,
        .out = kit.elem(k),
        .nan = .{ .stated = "every float64 output class" },
        // the recorded decode launches: the wide head-like layer at SK 1, the narrow one split 8 ways, then two rows
        .served = &.{
            .{ .rows = 1, .n = 16384, .k = 1280 },
            .{ .rows = 1, .n = 5120, .k = 4096 },
            .{ .rows = 2, .n = 16384, .k = 1280 },
        },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = sums_why, .exact = false },
    };
}

/// x [rows][k] fp16 rows over the strips tiles [n][k]: y [rows][n] of the kind from the dot2 sums and the finish.
pub const Exl3 = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    xs: []const u8 = &.{},
    words: []const u32 = &.{},
    svh: []const u8 = &.{},
    bias: []const u8 = &.{},
    arg: Arg = undefined,
    sum: u64 = 0,
    out_len: usize = 0,
    n: usize = 0,
    kk: usize = 0,
    sk: usize = 1,
    stride_nb: usize = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Exl3 {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Exl3) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Exl3) g.Kernel {
        return .{ .spec = spec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Exl3);

    pub fn prepare(self: *Exl3, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k % tile_k != 0 or shape.n % 128 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.n = shape.n;
        self.kk = shape.k;
        const tw: usize = 4 * half_bits;
        const stride_k: usize = 8 * tw;
        self.stride_nb = @divExact(shape.k, tile_k) * stride_k;
        const nb = shape.n / 128;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the rotated fp16 rows glinear reads; stress keeps them small integers so every fp32 sum stays exact
        const xn = shape.rows * shape.k;
        const xs = try a.alloc(u8, xn * 2);
        for (0..xn) |i| {
            const v: f64 = if (fill == .stress) @floatFromInt(1 + (i % 7)) else rng.float(f64) * 4 - 2;
            kit.put(.f16, xs, i, kit.round(.f16, v));
        }
        if (fill == .stress) for (0..xn) |i| {
            if (i % 29 == 0) kit.put(.f16, xs, i, 0); // a lane's zero must enter as +0, never skipped
        };
        if (fill == .nonfinite) for ([_]usize{ shape.k / 3, shape.k / 2, shape.k - 1 }) |at| {
            if (at < shape.rows * shape.k) kit.put(.f16, xs, at, std.math.nan(f64));
        };
        const words = try a.alloc(u32, nb * self.stride_nb + stride_k);
        for (words, 0..) |*w, i| w.* = 0x9e37_79b9 *% (@as(u32, @intCast(i)) +% 0x85eb_ca6b);
        const svh = try a.alloc(u8, shape.n * 2);
        for (0..shape.n) |i| kit.put(.f16, svh, i, kit.round(.f16, 0.02 + @as(f64, @floatFromInt(i % 97)) / 4096));
        const bias = try a.alloc(u8, shape.n * 2);
        for (0..shape.n) |i| kit.put(.f16, bias, i, kit.round(.f16, @as(f64, @floatFromInt(i % 131)) / 8192 - 0.008));
        const p = dsv41.plan(shape.k, shape.n);
        self.sk = @intCast(p.sk);
        const x = try self.rig.input(xs);
        const tiles = try self.rig.input(std.mem.sliceAsBytes(words));
        const scale = try self.rig.input(svh);
        const b = try self.rig.input(bias);
        const out = try self.rig.output(shape.rows * shape.n * 2);
        const z = try self.rig.scratch(@as(usize, @intCast(p.sk)) * shape.rows * shape.n * 4);
        const counters = try self.rig.scratch(8 * nb * 4);
        self.xs = xs;
        self.words = words;
        self.svh = svh;
        self.bias = bias;
        self.out_len = shape.rows * shape.n;
        var layers: [dsv41.gmax]dsv41.GLayer = @splat(.{});
        layers[0] = .{
            .xh = x,
            .T = tiles,
            .svh = scale,
            .bias = b,
            .y = out,
            .Z = if (p.sk > 1) z else 0,
            .counters = counters,
            .ldx = @intCast(shape.k),
            .ldy = @intCast(shape.n),
            .stride_k = @intCast(stride_k),
            .stride_nb = @intCast(self.stride_nb),
            .y_dtype = @intFromBool(self.k == .bf16),
            .K = @intCast(shape.k),
            .N = @intCast(shape.n),
            .SK = p.sk,
        };
        self.arg = .{ .l = layers, .n = 1, .M = @intCast(shape.rows) };
        return true;
    }

    pub fn run(self: *Exl3) !void {
        try hip.launches.dsv41.glinearGo(&self.l.dsv41, self.l.r, self.arg, half_bits, 1, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The non-finite plants must match every float64 output class.
    pub fn check(self: *Exl3, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(kit.elem(self.k), out, i)) != c.classOf(w))
                return "non-finite output class differs from float64";
        }
        return null;
    }

    pub fn count(self: *Exl3) usize {
        return self.out_len;
    }

    /// Column sums per k step, fwht128, svh, bias; mag is the block's term magnitude, fwht mixes every column in.
    pub fn reference(self: *Exl3, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        const terms = try a.alloc(f64, self.kk);
        defer a.free(terms);
        for (0..self.out_len / self.n) |r| for (0..self.n / 128) |nb| {
            var cols: [128]f64 = undefined;
            var block: f64 = 0;
            for (0..128) |cc| {
                self.fillTerms(terms, r, nb, cc);
                cols[cc] = rows.sumIn(.none, .sequential, terms);
                for (terms) |x| block += @abs(x);
            }
            fwht128(&cols);
            for (0..128) |cc| {
                const j = nb * 128 + cc;
                const t = cols[cc] * had_scale;
                const sv = kit.get(.f16, self.svh, j);
                const bs = kit.get(.f16, self.bias, j);
                want[r * self.n + j] = kit.round(self.k, t * sv + bs);
                mag[r * self.n + j] = block * had_scale * @abs(sv) + @abs(bs);
            }
        };
    }

    /// Rounded sums in order; SK > 1 splits become fp32 partials summed in split order by the completing block.
    pub fn simulate(self: *Exl3, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.arena.allocator();
        const terms = try a.alloc(f64, self.kk);
        defer a.free(terms);
        const sk: usize = @intCast(self.sk);
        const chunk = @divExact(self.kk, tile_k * sk) * tile_k;
        var at: usize = 0;
        for (0..self.out_len / self.n) |r| for (0..self.n / 128) |nb| {
            var cols: [128]f64 = undefined;
            for (0..128) |cc| {
                self.fillTerms(terms, r, nb, cc);
                var v = rows.sumIn(acc, order, terms[0..chunk]);
                for (1..sk) |q| v = roundAcc(acc, v + rows.sumIn(acc, order, terms[q * chunk .. (q + 1) * chunk]));
                cols[cc] = v;
            }
            fwhtRound(acc, &cols);
            for (0..128) |cc| {
                const j = nb * 128 + cc;
                const t = roundAcc(acc, cols[cc] * 0.08838834764831845);
                const sv = roundAcc(acc, kit.get(.f16, self.svh, j));
                const bs = roundAcc(acc, kit.get(.f16, self.bias, j));
                out[at] = kit.round(self.k, roundAcc(acc, t * sv) + bs);
                at += 1;
            }
        };
    }

    /// Column cc of block nb over row r: each k step's MUL1 weight and x's fp16 row at tile_k * kt + row.
    fn fillTerms(self: *Exl3, terms: []f64, r: usize, nb: usize, cc: usize) void {
        const t: usize = cc / 16;
        const c0: u32 = @intCast(cc % 16);
        const per = self.kk / tile_k;
        for (0..per) |kt| for (0..tile_k) |row| {
            const h: u32 = @intCast(row / 8);
            const q: u32 = @intCast((row % 8) / 2);
            const half: u32 = @intCast(row % 2);
            const p = 32 * (c0 & 7) + 8 * q + 4 * (c0 >> 3) + 2 * h + half;
            const bits = tileBits(self.words, self.stride_nb, nb, kt, t, p);
            const wv = mul1Value(bits);
            const xv = kit.get(.f16, self.xs, r * self.kk + kt * tile_k + row);
            terms[kt * tile_k + row] = xv * wv;
        };
    }
};

const rot_names = [2][]const u8{
    "hip dsv41 exl3_rot_many (fp16)",
    "hip dsv41 exl3_rot_many (bf16)",
};

/// rot_many's own spec: fp32 butterfly sums into fp16 (`store4(..., F16, ...)` writes fp16 whatever the cache kind is).
pub fn rotSpec(k: kit.Kind) c.Spec {
    return .{
        .name = rot_names[@intFromBool(k == .bf16)],
        .accum = .f32,
        // the first butterfly's three levels plus the xor stages and scale: nine roundings at most per element
        .depth = rotDepth,
        .ops = 1,
        .out = .f16,
        .nan = .{ .stated = "every float64 output class" },
        // the recorded narrow decode layer's rotation: k = 5120, one row
        .served = &.{.{ .rows = 1, .n = 5120, .k = 5120 }},
    };
}

fn rotDepth(_: c.Shape) usize {
    return 9;
}

/// The Hadamard finish's scale (exl3.hip HAD_SCALE = 1 / sqrt(128)).
const had_scale = 0.08838834764831845;

/// x's fp16 rows rotated by suh per 128-column block: y [rows][k] fp16.
pub const Rot = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    xs: []const u8 = &.{},
    suh: []const u8 = &.{},
    arg: RotArg = undefined,
    out_len: usize = 0,
    kk: usize = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Rot {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Rot) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Rot) g.Kernel {
        return .{ .spec = rotSpec(self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Rot);

    pub fn prepare(self: *Rot, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.kk = shape.k;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // the unrotated fp16 rows and the fp16 scales rot128_in reads; stress keeps them small integers
        const xn = shape.rows * shape.k;
        const xs = try a.alloc(u8, xn * 2);
        for (0..xn) |i| {
            const v: f64 = if (fill == .stress) @floatFromInt(1 + (i % 7)) else rng.float(f64) * 4 - 2;
            kit.put(.f16, xs, i, kit.round(.f16, v));
        }
        if (fill == .stress) for (0..xn) |i| {
            if (i % 29 == 0) kit.put(.f16, xs, i, 0); // a lane's zero must enter as +0, never skipped
        };
        if (fill == .nonfinite) for ([_]usize{ shape.k / 3, shape.k / 2, shape.k - 1 }) |at| {
            if (at < shape.rows * shape.k) kit.put(.f16, xs, at, std.math.nan(f64));
        };
        const suh = try a.alloc(u8, shape.k * 2);
        for (0..shape.k) |i| kit.put(.f16, suh, i, kit.round(.f16, 0.5 + @as(f64, @floatFromInt(i % 97)) / 4096));
        const x = try self.rig.input(xs);
        const scales = try self.rig.input(suh);
        const out = try self.rig.output(shape.rows * shape.k * 2);
        self.xs = xs;
        self.suh = suh;
        self.out_len = shape.rows * shape.k;
        var layers: [dsv41.gmax]dsv41.RLayer = @splat(.{});
        layers[0] = .{
            .x = x,
            .suh = scales,
            .xh = out,
            .ldx = @intCast(shape.k),
            .x_dtype = 0, // F16: the engine's fp16 rows
            .K = @intCast(shape.k),
            .first = 0,
        };
        self.arg = .{ .l = layers, .n = 1, .M = @intCast(shape.rows) };
        return true;
    }

    pub fn run(self: *Rot) !void {
        try hip.launches.dsv41.rotManyGo(&self.l.dsv41, self.l.r, self.arg, 1, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The non-finite plants must match every float64 output class.
    pub fn check(self: *Rot, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const want = try a.alloc(f64, self.count());
        const mag = try a.alloc(f64, self.count());
        try self.reference(want, mag);
        for (want, 0..) |w, i| {
            if (c.classOf(c.read(.f16, out, i)) != c.classOf(w))
                return "non-finite output class differs from float64";
        }
        return null;
    }

    pub fn count(self: *Rot) usize {
        return self.out_len;
    }

    /// One row of a block as rot128_in computes it: lane l owns elements 4 l + j; the xor stages then the scale follow.
    pub fn reference(self: *Rot, want: []f64, mag: []f64) !void {
        const rows_n = self.out_len / self.kk;
        const nblk = self.kk / 128;
        for (0..rows_n) |r| for (0..nblk) |nb| {
            var v: [32][4]f64 = undefined;
            var m: f64 = 0;
            for (0..32) |lane| for (0..4) |j| {
                const e = nb * 128 + 4 * lane + j;
                const xv = kit.get(.f16, self.xs, r * self.kk + e);
                const sv = kit.get(.f16, self.suh, e);
                v[lane][j] = xv * sv;
                m += @abs(xv * sv);
            };
            rot128(&v);
            for (0..32) |lane| for (0..4) |j| {
                const e = nb * 128 + 4 * lane + j;
                want[r * self.kk + e] = kit.round(.f16, v[lane][j] * had_scale);
                mag[r * self.kk + e] = m * had_scale;
            };
        };
    }

    /// The same rounded to `acc` at every step; rot128_in's order is fixed, so `order` cannot move it.
    pub fn simulate(self: *Rot, acc: c.Accum, _: g.Order, out: []f64) !void {
        const rows_n = self.out_len / self.kk;
        const nblk = self.kk / 128;
        var at: usize = 0;
        for (0..rows_n) |r| for (0..nblk) |nb| {
            var v: [32][4]f64 = undefined;
            for (0..32) |lane| for (0..4) |j| {
                const e = nb * 128 + 4 * lane + j;
                const xv = kit.get(.f16, self.xs, r * self.kk + e);
                const sv = roundAcc(acc, kit.get(.f16, self.suh, e));
                v[lane][j] = roundAcc(acc, xv * sv);
            };
            rot128Round(acc, &v);
            for (0..32) |lane| for (0..4) |j| {
                out[at] = kit.round(.f16, roundAcc(acc, v[lane][j] * had_scale));
                at += 1;
            };
        };
    }
};

/// rot128_in's butterfly on 32 lanes x 4 values: fma'd first stage, then xor stages m = 1, 2, 4, 8, 16.
fn rot128(v: *[32][4]f64) void {
    var w: [32][4]f64 = undefined;
    for (0..32) |lane| {
        const p1 = v[lane][1];
        const p3 = v[lane][3];
        const a = v[lane][0] + p1;
        const b = v[lane][0] - p1;
        const cc = v[lane][2] + p3;
        const d = v[lane][2] - p3;
        w[lane][0] = a + cc;
        w[lane][1] = b + d;
        w[lane][2] = a - cc;
        w[lane][3] = b - d;
    }
    var m: u32 = 1;
    while (m < 32) : (m <<= 1) {
        var nxt: [32][4]f64 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const o = w[lane ^ m][j];
            nxt[lane][j] = if ((lane & m) != 0) o - w[lane][j] else w[lane][j] + o;
        };
        w = nxt;
    }
    v.* = w;
}

/// The same butterflies with every add rounded to `acc` (gate 4's simulated sums).
fn rot128Round(acc: c.Accum, v: *[32][4]f64) void {
    var w: [32][4]f64 = undefined;
    for (0..32) |lane| {
        const p1 = v[lane][1];
        const p3 = v[lane][3];
        const a = roundAcc(acc, v[lane][0] + p1);
        const b = roundAcc(acc, v[lane][0] - p1);
        const cc = roundAcc(acc, v[lane][2] + p3);
        const d = roundAcc(acc, v[lane][2] - p3);
        w[lane][0] = roundAcc(acc, a + cc);
        w[lane][1] = roundAcc(acc, b + d);
        w[lane][2] = roundAcc(acc, a - cc);
        w[lane][3] = roundAcc(acc, b - d);
    }
    var m: u32 = 1;
    while (m < 32) : (m <<= 1) {
        var nxt: [32][4]f64 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const o = w[lane ^ m][j];
            nxt[lane][j] = if ((lane & m) != 0) roundAcc(acc, o - w[lane][j]) else roundAcc(acc, w[lane][j] + o);
        };
        w = nxt;
    }
    v.* = w;
}

/// fwht128 in the kernel's butterfly order: elements 4 lane + j, the first stage's four sums, then five xor steps.
fn fwht128(v: *[128]f64) void {
    var w: [32][4]f64 = undefined;
    for (0..32) |lane| {
        const a = v[4 * lane] + v[4 * lane + 1];
        const b = v[4 * lane] - v[4 * lane + 1];
        const cc = v[4 * lane + 2] + v[4 * lane + 3];
        const d = v[4 * lane + 2] - v[4 * lane + 3];
        w[lane][0] = a + cc;
        w[lane][1] = b + d;
        w[lane][2] = a - cc;
        w[lane][3] = b - d;
    }
    var m: u32 = 1;
    while (m < 32) : (m <<= 1) {
        var nxt: [32][4]f64 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const o = w[lane ^ m][j];
            nxt[lane][j] = if ((lane & m) != 0) o - w[lane][j] else w[lane][j] + o;
        };
        w = nxt;
    }
    for (0..32) |lane| {
        for (0..4) |j| v[4 * lane + j] = w[lane][j];
    }
}

/// The same butterflies with every add rounded to `acc` (gate 4's simulated sums).
fn fwhtRound(acc: c.Accum, v: *[128]f64) void {
    var w: [32][4]f64 = undefined;
    for (0..32) |lane| {
        const a = roundAcc(acc, v[4 * lane] + v[4 * lane + 1]);
        const b = roundAcc(acc, v[4 * lane] - v[4 * lane + 1]);
        const cc = roundAcc(acc, v[4 * lane + 2] + v[4 * lane + 3]);
        const d = roundAcc(acc, v[4 * lane + 2] - v[4 * lane + 3]);
        w[lane][0] = roundAcc(acc, a + cc);
        w[lane][1] = roundAcc(acc, b + d);
        w[lane][2] = roundAcc(acc, a - cc);
        w[lane][3] = roundAcc(acc, b - d);
    }
    var m: u32 = 1;
    while (m < 32) : (m <<= 1) {
        var nxt: [32][4]f64 = undefined;
        for (0..32) |lane| for (0..4) |j| {
            const o = w[lane ^ m][j];
            nxt[lane][j] = if ((lane & m) != 0) roundAcc(acc, o - w[lane][j]) else roundAcc(acc, w[lane][j] + o);
        };
        w = nxt;
    }
    for (0..32) |lane| {
        for (0..4) |j| v[4 * lane + j] = w[lane][j];
    }
}

pub fn run(a: std.mem.Allocator, _: std.Io, _: @import("contract_entry.zig").Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(a);
    var one = Exl3.init(a, &r, s, &l);
    defer one.deinit();
    try kernels.append(a, one.kernel());
    var two = Rot.init(a, &r, s, &l);
    defer two.deinit();
    try kernels.append(a, two.kernel());
    var t: g.Tally = .{};
    try g.runAll(a, kernels.items, 0x6b63, &t);
    return t;
}
