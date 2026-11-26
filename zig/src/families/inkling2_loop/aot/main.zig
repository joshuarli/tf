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
    std.debug.assert(vals.len > 0);
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
    var burst_size: usize = 0;
    var opt_mask: u32 = 7;
    var chunked_prefill = false;
    var half_inputs = true;
    var capture_taps = false;
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
        } else if (std.mem.eql(u8, args[i], "--prefill=chunked")) {
            chunked_prefill = true;
        } else if (std.mem.eql(u8, args[i], "--prefill=step")) {
            chunked_prefill = false;
        } else if (std.mem.eql(u8, args[i], "--half=false")) {
            half_inputs = false;
        } else if (std.mem.eql(u8, args[i], "--half=true")) {
            half_inputs = true;
        } else if (std.mem.eql(u8, args[i], "--taps")) {
            capture_taps = true;
        } else if (std.mem.startsWith(u8, args[i], "--burst=")) {
            burst_size = try std.fmt.parseInt(usize, args[i]["--burst=".len..], 10);
        } else if (std.mem.startsWith(u8, args[i], "--opt=")) {
            const val = args[i]["--opt=".len..];
            if (std.mem.eql(u8, val, "all")) {
                opt_mask = 0x1f;
            } else if (std.mem.eql(u8, val, "none")) {
                opt_mask = 0;
            } else {
                opt_mask = try std.fmt.parseInt(u32, val, 0);
            }
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
        if (ids.value.len == 0) return error.EmptyInput;

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(ids.value.len + 16, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
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
        if (ids.value.len == 0) return error.EmptyInput;
        const steps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        if (steps < 2) return error.InvalidSteps;

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(ids.value.len + steps + 16, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
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
        if (prompts.value.len == 0) return error.EmptyInput;
        for (prompts.value) |pr| if (pr.ids.len == 0) return error.EmptyInput;
        const reps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        if (reps == 0) return error.InvalidReps;
        var max_len: usize = 1024;
        for (prompts.value) |pr| max_len = @max(max_len, pr.ids.len + reps + 8);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = max_len,
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
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
            if (chunked_prefill) {
                const ps = try rt.prefillPrompt(pr.ids, true, half_inputs, capture_taps);
                tok = ps.last_token;
                if ((opt_mask & 256) != 0) try rt.convertKvToPerm();
            } else {
                for (pr.ids, 0..) |id, k| {
                    const last = k + 1 == pr.ids.len;
                    const st_ = try rt.stepDirect(id, last, gpu_argmax);
                    if (last) tok = st_.token;
                }
            }
            const first_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
            var last_stat: rt_mod.StepStats = .{};
            var step_ids = try gpa.alloc(u32, reps);
            defer gpa.free(step_ids);
            var n_syncs: usize = reps;
            if (burst_size > 0) {
                var produced: usize = 0;
                var n_bursts: usize = 0;
                var disp: u32 = 0;
                while (produced < reps) {
                    const take = @min(burst_size, reps - produced);
                    const bs = try rt.generateBurst(tok, take, 0xffffffff, step_ids[produced .. produced + take]);
                    const per_w = bs.wall_ms / @as(f64, @floatFromInt(bs.produced));
                    const per_g = bs.gpu_ms / @as(f64, @floatFromInt(bs.produced));
                    for (0..bs.produced) |bi| {
                        walls[produced + bi] = per_w;
                        gpus[produced + bi] = per_g;
                    }
                    disp = bs.dispatches;
                    tok = step_ids[produced + bs.produced - 1];
                    produced += bs.produced;
                    n_bursts += 1;
                }
                n_syncs = n_bursts;
                last_stat = .{ .dispatches = disp, .command_buffers = 1, .syncs = @intCast(n_bursts) };
            } else {
                for (0..reps) |r| {
                    last_stat = if (exec_icb) try rt.stepIcb(tok, gpu_argmax) else try rt.stepDirect(tok, true, gpu_argmax);
                    walls[r] = last_stat.wall_ms;
                    gpus[r] = last_stat.gpu_ms;
                    tok = last_stat.token;
                    step_ids[r] = tok;
                }
            }
            const w50 = median(walls);
            const g50 = median(gpus);
            const tsha = digestBytes(std.mem.sliceAsBytes(step_ids));
            std.debug.print("aot bench {s}: ctx={d} mode={s} burst={d} syncs={d} wall_p50={d:.3} ms gpu_p50={d:.3} ms tok/s={d:.2} disp={d} first_sha={s} tok_sha={s}\n", .{
                pr.name, pr.ids.len, if (exec_icb) "icb" else "direct", burst_size, n_syncs, w50, g50, 1000.0 / w50, last_stat.dispatches, first_sha[0..12], tsha[0..12],
            });
            try out.print(gpa, "{{\"prompt\":\"{s}\",\"context\":{d},\"mode\":\"{s}\",\"burst\":{d},\"syncs\":{d},\"gpu_argmax\":{},\"reps\":{d},\"wall_p50_ms\":{d:.4},\"gpu_p50_ms\":{d:.4},\"dispatches\":{d},\"first_logit_sha256\":\"{s}\",\"tokens_sha256\":\"{s}\"}}\n", .{
                pr.name, pr.ids.len, if (exec_icb) "icb" else "direct", burst_size, n_syncs, gpu_argmax, reps, w50, g50, last_stat.dispatches, first_sha, tsha,
            });
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[2], .data = out.items });
        return;
    }

    if (std.mem.eql(u8, mode, "ablate")) {
        if (pos_args.items.len < 2) return error.ExpectedArgs;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(pbytes);
        const prompts = try std.json.parseFromSlice([]const Prompt, gpa, pbytes, .{});
        defer prompts.deinit();
        if (prompts.value.len == 0) return error.EmptyInput;
        for (prompts.value) |pr| if (pr.ids.len == 0) return error.EmptyInput;
        const reps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        if (reps == 0) return error.InvalidReps;
        const pr = prompts.value[0];

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(pr.ids.len + reps + 32, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = 0,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms,
        });

        var base_tok: u32 = 0;
        for (pr.ids, 0..) |id, k| {
            const last = k + 1 == pr.ids.len;
            const st_ = try rt.stepDirect(id, last, false);
            if (last) base_tok = st_.token;
        }
        const saved = try rt.snapshotState();
        defer saved.deinit();

        const masks = [_]struct { mask: u32, name: []const u8 }{
            .{ .mask = 1, .name = "D1_comb_sig" },
            .{ .mask = 2, .name = "D2_nv_tg" },
            .{ .mask = 4, .name = "D3_ctl_u6" },
            .{ .mask = 7, .name = "D1+D2+D3" },
        };
        const base_g = try gpa.alloc(f64, reps);
        defer gpa.free(base_g);
        const cand_g = try gpa.alloc(f64, reps);
        defer gpa.free(cand_g);
        const base_w = try gpa.alloc(f64, reps);
        defer gpa.free(base_w);
        const cand_w = try gpa.alloc(f64, reps);
        defer gpa.free(cand_w);

        rt.opt_mask = 0;
        {
            var wt = base_tok;
            for (0..8) |_| wt = (try rt.stepDirect(wt, true, gpu_argmax)).token;
        }

        for (masks) |m| {
            var base_sha: [64]u8 = undefined;
            var cand_sha: [64]u8 = undefined;
            var base_disp: u32 = 0;
            var cand_disp: u32 = 0;
            const chunk = @max(reps / 4, 1);
            var bi: usize = 0;
            var ci: usize = 0;
            for (0..4) |round| {
                const base_first = (round % 2) == 0;
                for (0..2) |half| {
                    const is_base = (half == 0) == base_first;
                    const use_mask = if (is_base) 0 else m.mask;
                    rt.restoreState(saved);
                    rt.opt_mask = use_mask;
                    var tok = base_tok;
                    for (0..chunk) |k| {
                        const st_ = try rt.stepDirect(tok, true, gpu_argmax);
                        if (round == 0 and k == 0) {
                            if (is_base) base_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice())) else cand_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
                        }
                        if (is_base) {
                            if (bi < reps) {
                                base_g[bi] = st_.gpu_ms;
                                base_w[bi] = st_.wall_ms;
                                bi += 1;
                            }
                            base_disp = st_.dispatches;
                        } else {
                            if (ci < reps) {
                                cand_g[ci] = st_.gpu_ms;
                                cand_w[ci] = st_.wall_ms;
                                ci += 1;
                            }
                            cand_disp = st_.dispatches;
                        }
                        tok = st_.token;
                    }
                }
            }
            const bg50 = median(base_g[0..bi]);
            const cg50 = median(cand_g[0..ci]);
            const bw50 = median(base_w[0..bi]);
            const cw50 = median(cand_w[0..ci]);
            const exact = std.mem.eql(u8, &base_sha, &cand_sha);
            std.debug.print("paired {s:<15} disp={d}->{d} base_gpu={d:.3} cand_gpu={d:.3} d_gpu={d:.3} ms (wall {d:.3}->{d:.3}) exact={}\n", .{
                m.name, base_disp, cand_disp, bg50, cg50, cg50 - bg50, bw50, cw50, exact,
            });
        }
        return;
    }

    if (std.mem.eql(u8, mode, "profile")) {
        if (pos_args.items.len < 2) return error.ExpectedArgs;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(pbytes);
        const prompts = try std.json.parseFromSlice([]const Prompt, gpa, pbytes, .{});
        defer prompts.deinit();
        if (prompts.value.len == 0) return error.EmptyInput;
        for (prompts.value) |pr| if (pr.ids.len == 0) return error.EmptyInput;
        const reps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        if (reps == 0) return error.InvalidReps;
        const pr = prompts.value[0];
        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(pr.ids.len + 32, 1024),
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
        });
        defer rt.deinit();
        for (pr.ids, 0..) |id, k| {
            _ = try rt.stepDirect(id, k + 1 == pr.ids.len, false);
        }
        try rt.profileClasses(reps);
        return;
    }

    if (std.mem.eql(u8, mode, "histories")) {
        if (pos_args.items.len != 2) return error.ExpectedArgs;
        const History = struct { name: []const u8, token_ids: []const u32 };
        const hbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(hbytes);
        const histories = try std.json.parseFromSlice([]const History, gpa, hbytes, .{ .ignore_unknown_fields = true });
        defer histories.deinit();
        if (histories.value.len == 0) return error.EmptyInput;
        for (histories.value) |h| if (h.token_ids.len == 0) return error.EmptyInput;
        const out_dir = pos_args.items[1];
        try std.Io.Dir.cwd().createDirPath(io, out_dir);
        var max_len: usize = 1024;
        for (histories.value) |h| max_len = @max(max_len, h.token_ids.len + 16);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = max_len,
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms,
        });

        const vocab: usize = rt.sched.geom.unpadded_vocab;
        var total_rows: usize = 0;
        const t_all = rt_mod.nowNs();
        for (histories.value) |h| {
            rt.reset();
            const out_logits = try gpa.alloc(f32, h.token_ids.len * vocab);
            defer gpa.free(out_logits);
            const t0 = rt_mod.nowNs();
            for (h.token_ids, 0..) |tok, k| {
                _ = try rt.stepDirect(tok, true, false);
                @memcpy(out_logits[k * vocab ..][0..vocab], rt.logitsSlice());
            }
            const out_path = try std.fs.path.join(arena, &.{ out_dir, try std.fmt.allocPrint(arena, "{s}.bin", .{h.name}) });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = std.mem.sliceAsBytes(out_logits) });
            const ms = @as(f64, @floatFromInt(rt_mod.nowNs() - t0)) / 1e6;
            total_rows += h.token_ids.len;
            std.debug.print("aot history {s}: {d} rows in {d:.1} ms ({d:.2} ms/row)\n", .{ h.name, h.token_ids.len, ms, ms / @as(f64, @floatFromInt(h.token_ids.len)) });
        }
        const total_s = @as(f64, @floatFromInt(rt_mod.nowNs() - t_all)) / 1e9;
        std.debug.print("aot histories: {d} files, {d} positions in {d:.2} s\n", .{ histories.value.len, total_rows, total_s });
        return;
    }

    if (std.mem.eql(u8, mode, "prefill-bench")) {
        if (pos_args.items.len != 2) return error.ExpectedArgs;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(pbytes);
        const prompts = try std.json.parseFromSlice([]const Prompt, gpa, pbytes, .{});
        defer prompts.deinit();
        if (prompts.value.len == 0) return error.EmptyInput;
        for (prompts.value) |pr| if (pr.ids.len == 0) return error.EmptyInput;
        var max_len: usize = 1024;
        for (prompts.value) |pr| max_len = @max(max_len, pr.ids.len + 16);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = max_len,
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms,
        });

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        for (prompts.value, 0..) |pr, pi| {
            rt.reset();
            const ps = try rt.prefillPrompt(pr.ids, true, half_inputs, capture_taps);
            const tok_s = @as(f64, @floatFromInt(pr.ids.len)) * 1000.0 / ps.wall_ms;
            const gpu_tok_s = @as(f64, @floatFromInt(pr.ids.len)) * 1000.0 / ps.gpu_ms;
            const sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
            std.debug.print("aot prefill[{d}] {s}: n={d} half={} taps={} wall={d:.1} ms ({d:.1} tok/s) gpu={d:.1} ms ({d:.1} tok/s) tok={d} sha={s}\n", .{
                pi, pr.name, pr.ids.len, half_inputs, capture_taps, ps.wall_ms, tok_s, ps.gpu_ms, gpu_tok_s, ps.last_token, sha[0..12],
            });
            try out.print(gpa, "{{\"idx\":{d},\"prompt\":\"{s}\",\"tokens\":{d},\"half\":{},\"taps\":{},\"wall_ms\":{d:.3},\"gpu_ms\":{d:.3},\"wall_tok_s\":{d:.2},\"gpu_tok_s\":{d:.2},\"last_token\":{d},\"last_logit_sha256\":\"{s}\"}}\n", .{
                pi, pr.name, pr.ids.len, half_inputs, capture_taps, ps.wall_ms, ps.gpu_ms, tok_s, gpu_tok_s, ps.last_token, sha,
            });
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[1], .data = out.items });
        return;
    }

    if (std.mem.eql(u8, mode, "verify-bench")) {
        if (pos_args.items.len != 3) return error.ExpectedArgs;
        const pbytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(pbytes);
        const prompts = try std.json.parseFromSlice([]const Prompt, gpa, pbytes, .{});
        defer prompts.deinit();
        if (prompts.value.len == 0) return error.EmptyInput;
        for (prompts.value) |pr| if (pr.ids.len == 0) return error.EmptyInput;
        const reps = try std.fmt.parseInt(usize, pos_args.items[1], 10);
        if (reps == 0) return error.InvalidReps;
        var max_len: usize = 1024;
        for (prompts.value) |pr| max_len = @max(max_len, pr.ids.len + 64);

        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = max_len,
            .use_mmap = use_mmap,
            .device_arch_override = arch_override,
            .opt_mask = opt_mask,
        });
        defer rt.deinit();
        std.debug.print("aot load: total={d:.1} ms (verify={d:.1}, weights={d:.1}, pipelines={d:.1}, residency={d:.1})\n", .{
            rt.stats.total_ms, rt.stats.verify_ms, rt.stats.weights_ms, rt.stats.pipelines_ms, rt.stats.residency_ms,
        });

        const vocab: usize = rt.sched.geom.unpadded_vocab;
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        const widths = [_]usize{ 1, 2, 4, 8, 12, 16 };
        const walls = try gpa.alloc(f64, reps);
        defer gpa.free(walls);
        const gpus = try gpa.alloc(f64, reps);
        defer gpa.free(gpus);

        for (prompts.value, 0..) |pr, pi| {
            rt.reset();
            const ps = try rt.prefillPrompt(pr.ids, true, half_inputs, false);
            const anchor = ps.last_token;
            const saved = try rt.snapshotState();
            defer saved.deinit();

            var chain_ids: [16]u32 = undefined;
            chain_ids[0] = anchor;
            for (1..16) |k| chain_ids[k] = @intCast((anchor + k * 97) % vocab);

            if (pi == 0) {
                var ref_chain: [16][64]u8 = undefined;
                for (0..16) |k| {
                    _ = try rt.stepDirect(chain_ids[k], true, false);
                    ref_chain[k] = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
                }
                rt.restoreState(saved);
                _ = try rt.verifyTree(chain_ids[0..8], null, false, capture_taps);
                const vs = &(rt.verify_scratch.?);
                const vlogs = vs.logits.slice(f32, 16 * vocab);
                var chain8_ok = true;
                for (0..8) |k| {
                    const got = digestBytes(std.mem.sliceAsBytes(vlogs[k * vocab ..][0..vocab]));
                    if (!std.mem.eql(u8, &got, &ref_chain[k])) chain8_ok = false;
                }
                _ = try rt.verifyTree(chain_ids[0..16], null, false, capture_taps);
                var chain16_ok = true;
                for (0..16) |k| {
                    const got = digestBytes(std.mem.sliceAsBytes(vlogs[k * vocab ..][0..vocab]));
                    if (!std.mem.eql(u8, &got, &ref_chain[k])) chain16_ok = false;
                }

                const tree_parents = [_]u8{ 0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10 };
                _ = try rt.verifyTree(chain_ids[0..16], &tree_parents, false, capture_taps);
                var tree_shas: [16][64]u8 = undefined;
                for (0..16) |k| tree_shas[k] = digestBytes(std.mem.sliceAsBytes(vlogs[k * vocab ..][0..vocab]));
                var tree16_ok = true;
                const check_leaves = [_][5]usize{ .{ 0, 1, 3, 7, 12 }, .{ 0, 1, 3, 8, 13 }, .{ 0, 1, 4, 9, 14 }, .{ 0, 2, 5, 10, 15 } };
                for (check_leaves) |branch| {
                    rt.restoreState(saved);
                    for (branch) |node| {
                        _ = try rt.stepDirect(chain_ids[node], true, false);
                        const ref_s = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
                        if (!std.mem.eql(u8, &ref_s, &tree_shas[node])) tree16_ok = false;
                    }
                }
                rt.restoreState(saved);

                var accept_ids: [8]u32 = undefined;
                accept_ids[0] = anchor;
                {
                    var t = anchor;
                    for (1..8) |k| {
                        t = (try rt.stepDirect(t, true, false)).token;
                        accept_ids[k] = t;
                    }
                }
                accept_ids[4] ^= 1;
                rt.restoreState(saved);
                const cv = try rt.verifyTree(accept_ids[0..8], null, true, false);
                const commit_ok = cv.accepted_last == 3 and cv.kept_tokens == 4 and rt.len == saved.len + 4;
                const after_commit = try rt.stepDirect(accept_ids[4], true, false);
                const after_commit_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
                rt.restoreState(saved);
                for (0..4) |k| _ = try rt.stepDirect(accept_ids[k], false, false);
                const ref_after = try rt.stepDirect(accept_ids[4], true, false);
                const ref_after_sha = digestBytes(std.mem.sliceAsBytes(rt.logitsSlice()));
                const commit_state_ok = commit_ok and after_commit.token == ref_after.token and std.mem.eql(u8, &after_commit_sha, &ref_after_sha);
                std.debug.print("aot verify parity: chain8={} chain16={} tree16={} commit_rollback_exact={} (kept={d})\n", .{
                    chain8_ok, chain16_ok, tree16_ok, commit_state_ok, cv.kept_tokens,
                });
                if (!chain8_ok or !chain16_ok or !tree16_ok or !commit_state_ok) return error.VerifyParityFailed;
                rt.restoreState(saved);
            }

            for (widths) |w| {
                for (0..reps) |r| {
                    const st_ = try rt.verifyTree(chain_ids[0..w], null, false, capture_taps);
                    walls[r] = st_.wall_ms;
                    gpus[r] = st_.gpu_ms;
                }
                const w50 = median(walls);
                const g50 = median(gpus);
                std.debug.print("aot verify {s}: ctx={d} width={d} taps={} wall_p50={d:.3} ms gpu_p50={d:.3} ms\n", .{
                    pr.name, pr.ids.len, w, capture_taps, w50, g50,
                });
                try out.print(gpa, "{{\"prompt\":\"{s}\",\"context\":{d},\"width\":{d},\"taps\":{},\"reps\":{d},\"wall_p50_ms\":{d:.4},\"gpu_p50_ms\":{d:.4}}}\n", .{
                    pr.name, pr.ids.len, w, capture_taps, reps, w50, g50,
                });
            }
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = pos_args.items[2], .data = out.items });
        return;
    }

    if (std.mem.eql(u8, mode, "cancel-test")) {
        if (pos_args.items.len < 1) return error.ExpectedArgs;
        const tok_bytes = try std.Io.Dir.cwd().readFileAlloc(io, pos_args.items[0], gpa, .limited(64 << 20));
        defer gpa.free(tok_bytes);
        const ids = try std.json.parseFromSlice([]const u32, gpa, tok_bytes, .{});
        defer ids.deinit();
        if (ids.value.len == 0) return error.EmptyInput;
        const rt = try rt_mod.Runtime.open(gpa, io, bundle_dir, .{
            .capacity = @max(ids.value.len + 64, 1024),
            .opt_mask = opt_mask,
        });
        defer rt.deinit();
        var tok: u32 = 0;
        for (ids.value, 0..) |id, k| {
            const st_ = try rt.stepDirect(id, k + 1 == ids.value.len, true);
            if (k + 1 == ids.value.len) tok = st_.token;
        }
        const saved = try rt.snapshotState();
        defer saved.deinit();

        var ref_toks: [16]u32 = undefined;
        {
            var t = tok;
            for (0..16) |k| {
                t = (try rt.stepDirect(t, true, false)).token;
                ref_toks[k] = t;
            }
        }
        rt.restoreState(saved);
        var burst_toks: [16]u32 = undefined;
        const b1 = try rt.generateBurst(tok, 8, 0xffffffff, burst_toks[0..8]);
        const b2 = try rt.generateBurst(burst_toks[7], 8, 0xffffffff, burst_toks[8..16]);
        const parity_ok = std.mem.eql(u32, &ref_toks, &burst_toks);

        rt.restoreState(saved);
        const eos_stop = ref_toks[3];
        const expected_eos_n = (std.mem.indexOfScalar(u32, &ref_toks, eos_stop) orelse 3) + 1;
        var eos_toks: [8]u32 = undefined;
        const be = try rt.generateBurst(tok, 8, eos_stop, &eos_toks);
        const eos_ok = be.stopped and !be.cancelled and be.produced == expected_eos_n and std.mem.eql(u32, ref_toks[0..expected_eos_n], eos_toks[0..expected_eos_n]);

        rt.restoreState(saved);
        const CancelTask = struct {
            rt: *rt_mod.Runtime,
            fn run(task: *@This()) void {
                var ts = std.c.timespec{ .sec = 0, .nsec = 30 * std.time.ns_per_ms };
                _ = std.c.nanosleep(&ts, null);
                task.rt.requestCancel();
            }
        };
        var ctask = CancelTask{ .rt = rt };
        const th = try std.Thread.spawn(.{}, CancelTask.run, .{&ctask});
        defer th.join();
        var can_toks: [16]u32 = undefined;
        const bc = try rt.generateBurst(tok, 16, 0xffffffff, &can_toks);
        const cancel_ok = bc.cancelled and !bc.stopped and bc.produced < 16 and std.mem.eql(u32, ref_toks[0..bc.produced], can_toks[0..bc.produced]);

        std.debug.print("aot phase-e: burst16_parity={} b1={d:.2}ms/tok b2={d:.2}ms/tok eos_stop_ok={} (n={d}) host_cancel_ok={} (n={d}/16 in {d:.1}ms)\n", .{
            parity_ok, b1.wall_ms / 8.0, b2.wall_ms / 8.0, eos_ok, be.produced, cancel_ok, bc.produced, bc.wall_ms,
        });
        if (!parity_ok or !eos_ok or !cancel_ok) return error.PhaseECheckFailed;
        return;
    }

    return error.UnknownMode;
}
