const std = @import("std");
const mtl = @import("metal");
const objc = mtl.objc;

const Id = objc.Id;

pub const Error = error{ ArchiveCreate, ArchiveOpen, ArchiveAdd, ArchiveSerialize, PipelineLoad, NoFunction };

pub fn deviceArch(device: mtl.Device, buf: []u8) []const u8 {
    if (!device.responds("architecture")) return "unknown";
    const arch = objc.msg(?Id, device.id, "architecture", .{}) orelse return "unknown";
    const name = std.mem.span(objc.utf8(objc.msg(Id, arch, "name", .{})));
    const n = @min(name.len, buf.len);
    @memcpy(buf[0..n], name[0..n]);
    return buf[0..n];
}

pub const BinaryArchive = struct {
    id: Id,

    pub fn create(device: mtl.Device) Error!BinaryArchive {
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLBinaryArchiveDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newBinaryArchiveWithDescriptor:error:", .{ desc, &err });
        return .{ .id = id orelse {
            std.log.err("binary archive create: {s}", .{objc.errorText(err)});
            return error.ArchiveCreate;
        } };
    }

    pub fn open(device: mtl.Device, path: []const u8) Error!BinaryArchive {
        const str = objc.string(path);
        defer objc.release(str);
        const url = objc.msg(Id, objc.class("NSURL"), "fileURLWithPath:", .{str});
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLBinaryArchiveDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setUrl:", .{url});
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newBinaryArchiveWithDescriptor:error:", .{ desc, &err });
        return .{ .id = id orelse {
            std.log.err("binary archive open {s}: {s}", .{ path, objc.errorText(err) });
            return error.ArchiveOpen;
        } };
    }

    pub fn addCompute(self: BinaryArchive, lib: mtl.Library, name: []const u8, indirect: bool) Error!void {
        const func = lib.function(name) catch return error.NoFunction;
        defer objc.release(func);
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLComputePipelineDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setComputeFunction:", .{func});
        objc.msg(void, desc, "setSupportIndirectCommandBuffers:", .{indirect});
        var err: ?Id = null;
        const ok = objc.msg(bool, self.id, "addComputePipelineFunctionsWithDescriptor:error:", .{ desc, &err });
        if (!ok) {
            std.log.err("binary archive add {s}: {s}", .{ name, objc.errorText(err) });
            return error.ArchiveAdd;
        }
    }

    pub fn serialize(self: BinaryArchive, path: []const u8) Error!void {
        const str = objc.string(path);
        defer objc.release(str);
        const url = objc.msg(Id, objc.class("NSURL"), "fileURLWithPath:", .{str});
        var err: ?Id = null;
        const ok = objc.msg(bool, self.id, "serializeToURL:error:", .{ url, &err });
        if (!ok) {
            std.log.err("binary archive serialize {s}: {s}", .{ path, objc.errorText(err) });
            return error.ArchiveSerialize;
        }
    }

    pub fn loadPipeline(self: BinaryArchive, device: mtl.Device, lib: mtl.Library, name: []const u8, indirect: bool) Error!mtl.Pipeline {
        const func = lib.function(name) catch return error.NoFunction;
        defer objc.release(func);
        const desc = objc.msg(Id, objc.msg(Id, objc.class("MTLComputePipelineDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(desc);
        objc.msg(void, desc, "setComputeFunction:", .{func});
        objc.msg(void, desc, "setSupportIndirectCommandBuffers:", .{indirect});
        const arr = objc.msg(Id, objc.class("NSArray"), "arrayWithObject:", .{self.id});
        objc.msg(void, desc, "setBinaryArchives:", .{arr});
        var err: ?Id = null;
        const id = objc.msg(?Id, device.id, "newComputePipelineStateWithDescriptor:options:reflection:error:", .{ desc, @as(usize, 0), @as(?*anyopaque, null), &err });
        return .{ .id = id orelse {
            std.log.err("binary archive pipeline {s}: {s}", .{ name, objc.errorText(err) });
            return error.PipelineLoad;
        } };
    }

    pub fn deinit(self: BinaryArchive) void {
        objc.release(self.id);
    }
};
