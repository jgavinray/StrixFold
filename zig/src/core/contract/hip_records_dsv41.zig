//! hipcc launch usage for the dsv41 decode linear on gfx1030/1100/1151/1201; the objects audit rechecks it.
const Record = @import("hip_records.zig").Record;

pub const records = [2]Record{
    // the wide decode layer at SK 1: eight waves a block, dynamic staging over the static stage
    .{
        .launch = .{ .name = "dsv41 exl3_glinear k8 w8", .symbol = "tf_ds_exl3_glinear_k8_w8", .image = "dsv41", .threads = 256, .dynamic = 4096 },
        .usage = .{
            .{ .vgprs = 227, .sgprs = 63, .lds = 8208, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 76, .lds = 8208, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 76, .lds = 8208, .private = 0, .bound = 256, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 64, .lds = 8208, .private = 0, .bound = 256, .wave = 32 },
        },
    },
    // the narrow split-K layer: four waves a block, the splits summed by the completing block
    .{
        .launch = .{ .name = "dsv41 exl3_glinear k8 w4", .symbol = "tf_ds_exl3_glinear_k8_w4", .image = "dsv41", .threads = 128, .dynamic = 2048 },
        .usage = .{
            .{ .vgprs = 228, .sgprs = 69, .lds = 4112, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 82, .lds = 4112, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 232, .sgprs = 82, .lds = 4112, .private = 0, .bound = 128, .wave = 32 },
            .{ .vgprs = 231, .sgprs = 70, .lds = 4112, .private = 0, .bound = 128, .wave = 32 },
        },
    },
};
