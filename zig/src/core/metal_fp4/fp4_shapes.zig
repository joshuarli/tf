//! Inkling 2 Mini's routed-expert geometry, as the checkpoint's safetensors header stores it:
//! NVFP4 and MXFP8 stacks of 128 experts, a gate/up pair 2c and 2c+1 of 1152 intermediate columns,
//! a down matrix of 2304 rows over those 1152.
const std = @import("std");

pub const topk: usize = 8;
pub const hidden: usize = 2304;
pub const intermediate: usize = 1152;
pub const expert_count: usize = 128;
pub const decode_rows = [_]usize{ 1, 4, 16 };

pub const Format = enum(u1) {
    nvfp4,
    mxfp8,

    pub fn name(f: Format) []const u8 {
        return if (f == .nvfp4) "nvfp4" else "mxfp8";
    }

    pub fn groupBytes(f: Format) usize {
        return if (f == .nvfp4) 2 else 1;
    }

    pub fn scaleGroup(f: Format) usize {
        return if (f == .nvfp4) 16 else 32;
    }
};

pub const Projection = enum(u1) {
    gu,
    down,

    pub fn name(p: Projection) []const u8 {
        return if (p == .gu) "gu" else "down";
    }
};

pub const Geometry = struct {
    k: usize,
    n: usize,
    rows: usize,
    code_bytes: usize,
    scale_bytes: usize,
    group: usize,
};

pub fn geometry(f: Format, p: Projection) Geometry {
    const width: usize = if (p == .gu) hidden else intermediate;
    const columns: usize = if (p == .gu) intermediate else hidden;
    const stored: usize = if (p == .gu) 2 * intermediate else hidden;
    return .{
        .k = width,
        .n = columns,
        .rows = stored,
        .code_bytes = stored * width / f.groupBytes(),
        .scale_bytes = stored * width / f.scaleGroup(),
        .group = f.scaleGroup(),
    };
}

test "the checkpoint's expert geometry" {
    const gu = geometry(.nvfp4, .gu);
    try std.testing.expectEqual(@as(usize, 2304), gu.k);
    try std.testing.expectEqual(@as(usize, 1152), gu.n);
    try std.testing.expectEqual(@as(usize, 2304), gu.rows);
    try std.testing.expectEqual(@as(usize, 128 * 2304 * 1152), gu.code_bytes * 128);
    try std.testing.expectEqual(@as(usize, 128 * 2304 * 144), gu.scale_bytes * 128);
    const mx = geometry(.mxfp8, .down);
    try std.testing.expectEqual(@as(usize, 1152), mx.k);
    try std.testing.expectEqual(@as(usize, 2304), mx.n);
    try std.testing.expectEqual(@as(usize, 128 * 2304 * 1152), mx.code_bytes * 128);
    try std.testing.expectEqual(@as(usize, 128 * 2304 * 36), mx.scale_bytes * 128);
}
