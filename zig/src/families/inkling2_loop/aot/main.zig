const std = @import("std");
const rt_mod = @import("runtime.zig");
const fmt = @import("format.zig");

const Prompt = struct { name: []const u8, ids: []const u32 };

fn digestBytes(bytes: []const u8) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(bytes);
    return fmt.hexEncode(h.finalResult());
}

fn median(vals: []f64) f64 {
    std.mem.sort(f64, vals, {}, std.sort.asc(f64));
    return if (vals.len % 2 == 1) vals[vals.len / 2] else (vals[vals.len / 2 - 1] + vals[vals.len / 2]) / 2.0;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3) {
        std.debug.print(
            \\usage:
            \\  tf-inkling2-mini-aot <bundle-dir> verify [--device-arch-override ARCH]
            \\  tf-inkling2-mini-aot <bundle-dir> logits <tokens.json> <out.bin> [--mode=direct|icb] [--mmap]
            \\  tf-inkling2-mini-aot <bundle-dir> bench <prompts.json> <reps> <out.jsonl> [--mode=direct|icb] [--mmap] [--gpu-argmax]
            \\  tf-inkling2-mini-aot <bundle-dir> generate <tokens.json> <steps> <out.json> [--mode=direct|icb] [--mmap] [--gpu-argmax]
            \\
        , .{});
        return error.ExpectedArgs;
    }

    const bundle_dir = args[1];
    const mode = args[2];
    var exec_icb = false;
    var use_mmap = false;
    var gpu_argmax = false;
    var arch_override: ?[]const u8 = null;

    var pos_args: std.ArrayList([]const u8) = .empty;
    defer pos_args.deinit(gpa);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--mode=icb")) {
            exec_icb = true;
        } else if (std.mem.eql(u8, args[i], "--mode=direct")) {
            exec_icb = false;
        } else if (std.mem.eql(u8, args[i], "--mmap")) {
            use_mmap = true;
        } else if (std.mem.eql(u8, args[i], "--gpu-argmax")) {
            gpu_argmax = true;
        } else if (std.mem.eql(u8, args[i], "--device-arch-override") and i + 1 < args.len) {
            i += 1;
            arch_override = args[i];
        } else {
            try pos_args.append(gpa, args[i]);
        }
    }

    if (std.mem.eql(u8, mode, "verify")) {
        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .verify_only = true,
            .device_arch_override = arch_override,
        });
        defer rt.deinit();
        std.debug.print("aot verify: OK arch={s} gen={d} weights={d} verify_ms={d:.1} pipelines_ms={d:.1}\n", .{
            rt.sched.archName(), rt.sched.min_generation, rt.sched.weights_size, rt.stats.verify_ms, rt.stats.pipelines_ms,
        });
        return;
    }

    if (std.mem.eql(u8, mode, "logits")) {
        if (pos_args.items.len != 2) return error.ExpectedArgs;
        const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(tok_bytes);
        const ids = try std.json.parseFromSlice([]const u32, gpa, tok_bytes, .{});
        defer ids.deinit();

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(ids.value.len + 16, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1}, mmap={})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms, rt.stats.use_mmap,
        });

        const vocab: usize = rt.sched.geom.unpadded_vocab;
        const out_logits = try gpa.alloc(f32, ids.value.len * vocab);
        defer gpa.free(out_logits);
        const t0 = rt_mod.nowNs();
        for (ids.value, 0..) |tok, k| {
            _ = if (exec_icb) try rt.stepIcb(tok, false) else try rt.stepDirect(tok, true, false);
            @memcpy(out_logits[k * vocab ..][0..vocab], rt.logitsSlice());
        }
        const elapsed_ms = @as(f64, @floatFromInt(rt_mod.nowNs() - t0)) / 1e6;
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[1], .data = std.mem.sliceAsBytes(out_logits) });
        std.debug.print("aot logits: {d} rows in {d:.1} ms ({d:.2} ms/row)\n", .{ ids.value.len, elapsed_ms, elapsed_ms / @as(f64, @floatFromInt(ids.value.len)) });
        return;
    }

    if (std.mem.eql(u8, mode, "generate")) {
        if (pos_args.items.len != 3) return error.ExpectedArgs;
        const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(tok_bytes);
        const ids = try std.json.parseFromSlice([]const u32, gpa, tok_bytes, .{});
        defer ids.deinit();
        const steps = try std.fmt.parseInt(usize, pos_args.items[1], 10);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(ids.value.len + steps + 16, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1}, mmap={})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms, rt.stats.use_mmap,
        });

        var tok: u32 = 0;
        for (ids.value, 0..) |id, k| {
            const last = k + 1 == ids.value.len;
            const st_ = try rt.stepDirect(id, last, gpu_argmax);
            if (last) tok = st_.token;
        }

        const out_ids = try gpa.alloc(u32, steps);
        defer gpa.free(out_ids);
        const walls = try gpa.alloc(f64, steps);
        defer gpa.free(walls);
        const gpus = try gpa.alloc(f64, steps);
        defer gpa.free(gpus);

        out_ids[0] = tok;
        var sum_wall: f64 = 0;
        var sum_gpu: f64 = 0;
        for (1..steps) |s| {
            const st_ = if (exec_icb) try rt.stepIcb(tok, gpu_argmax) else try rt.stepDirect(tok, true, gpu_argmax);
            tok = st_.token;
            out_ids[s] = tok;
            walls[s - 1] = st_.wall_ms;
            gpus[s - 1] = st_.gpu_ms;
            sum_wall += st_.wall_ms;
            sum_gpu += st_.gpu_ms;
        }
        const n_dec = steps - 1;
        const med_wall = median(walls[0..n_dec]);
        const med_gpu = median(gpus[0..n_dec]);
        const sha = digestBytes(std.mem.sliceAsBytes(out_ids));
        std.debug.print("aot generate ({s}, gpu_argmax={}): prompt={d} steps={d} wall_p50={d:.3} ms gpu_p50={d:.3} ms mean_tok_s={d:.2} sha={s}\n", .{
            if (exec_icb) "icb" else "direct", gpu_argmax, ids.value.len, steps, med_wall, med_gpu, @as(f64, @floatFromInt(n_dec)) * 1000.0 / sum_wall, sha[0..16],
        });
        var w: std.Io.Writer.Allocating = .init(gpa);
        defer w.deinit();
        try std.json.Stringify.value(out_ids, .{}, &w.writer);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[2], .data = w.written() });
        return;
    }

    if (std.mem.eql(u8, mode, "bench")) {
        if (pos_args.items.len != 3) return error.ExpectedArgs;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(pbytes);
        const prompts = try std.json.parseFromSlice([]const Prompt, gpa, pbytes, .{});
        defer prompts.deinit();
        const reps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        var max_len: usize = 1024;
        for (prompts.value) |pr| max_len = @max(max_len, pr.ids.len + reps + 8);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = max_len,
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1}, mmap={})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms, rt.stats.use_mmap,
        });

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        const walls = try gpa.alloc(f64, reps);
        defer gpa.free(walls);
        const gpus = try gpa.alloc(f64, reps);
        defer gpa.free(gpus);

        for (prompts.value) |pr| {
            rt.reset();
            var tok: u32 = 0;
            for (pr.ids, 0..) |id, k| {
                const last = k + 1 == pr.ids.len;
                const st_ = try rt.stepDirect(id, last, gpu_argmax);
                if (last) tok = st_.token;
            }
            const first_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
            var last_stat: rt_mod.StepStats = .{};
            var step_ids = try gpa.alloc(u32, reps);
            defer gpa.free(step_ids);
            for (0..reps) |r| {
                last_stat = if (exec_icb) try rt.stepIcb(tok, gpu_argmax) else try rt.stepDirect(tok, true, gpu_argmax);
                walls[r] = last_stat.wall_ms;
                gpus[r] = last_stat.gpu_ms;
                tok = last_stat.token;
                step_ids[r] = tok;
            }
            const w50 = median(walls);
            const g50 = median(gpus);
            const tsha = digestBytes(std.mem.sliceAsBytes(step_ids));
            std.debug.print("aot bench {s}: ctx={d} mode={s} wall_p50={d:.3} ms gpu_p50={d:.3} ms tok/s={d:.2} disp={d} cb={d} first_sha={s} tok_sha={s}\n", .{
                pr.name, pr.ids.len, if (exec_icb) "icb" else "direct", w50, g50, 1000.0 / w50, last_stat.dispatches, last_stat.command_buffers, first_sha[0..12], tsha[0..12],
            });
            try out.print(gpa, "{{\"prompt\":\"{s}\",\"context\":{d},\"mode\":\"{s}\",\"gpu_argmax\":{},\"reps\":{d},\"wall_p50_ms\":{d:.4},\"gpu_p50_ms\":{d:.4},\"dispatches\":{d},\"cbs\":{d},\"first_logit_sha256\":\"{s}\",\"tokens_sha256\":\"{s}\"}}\n", .{
                pr.name, pr.ids.len, if (exec_icb) "icb" else "direct", gpu_argmax, reps, w50, g50, last_stat.dispatches, last_stat.command_buffers, first_sha, tsha,
            });
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[2], .data = out.items });
        return;
    }

    return error.UnknownMode;
}
