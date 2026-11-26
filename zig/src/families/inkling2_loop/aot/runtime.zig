const std = @import("std");
const mtl = @import("metal");
const fmt = @import("format.zig");
const ba = @import("binary_archive.zig");
pub const prefill_mod = @import("prefill.zig");

const F_NOCACHE: c_int = 48;
const mix_states: u32 = 5;
const conv_ring: u32 = 4;
const attn_chunk: u32 = 64;
const attn_splits: u32 = 16;

const EmbedA = extern struct { width: u32, rows: u32, eps: f32 };
const PairA = extern struct { count: u32 };
const PassCopyA = extern struct { width: u32, stride: u32, start: u32, count: u32 };
const MixArgs = extern struct { width: u32, depth: u32, stride: u32, k: u32, norm: u32, scale: f32, eps: f32 };
const ControlsArgs = extern struct { h: u32, rank: u32, rows: u32, xs: [3]u32, xb: [3]u32, msg0: u32 = 0, cols: u32 = 1 };
const Rms3A = extern struct { width: u32, xs: [3]u32, xb: [3]u32, eps: f32 };
const RmsBA = extern struct { width: u32, rows: u32, xs: u32, xb: u32, ys: u32, yb: u32, wb: u32, eps: f32 };
const RmsPairA = extern struct { width: u32, xs0: u32, xb0: u32, xs1: u32, xb1: u32, eps: f32 };
const AffineProjA = extern struct { k: u32, full: u32, m: [8]u32 };
const QkvArgs = extern struct { kv: u32, heads: u32, dim: u32, rel: u32, extent: u32, layout: u32, dyn: u32, tau: u32, norm: f32, q_threshold: f32, rel_threshold: f32, eps: f32, floor: f32, alpha: f32 };
const BatchQkvRow = extern struct { k_ring: u64, v_ring: u64, keys: u64, values: u64, pos: u32, pad: u32 = 0 };
const DecodeParts = extern struct { heads: u32, kv: u32, dim: u32, rel: u32, extent: u32, window: u32, pos: u32, tau: u32, splits: u32, chunk: u32, scale: f32, floor: f32, alpha: f32 };
const BatchRow = extern struct { keys: u64, values: u64, args: DecodeParts };
const AffineDenseA = extern struct { n: u32, k: u32, rows: u32, xs: u32, xb: u32, bits: u32, cap: f32 };
const DecodeConvA = extern struct { channels: u32, heads: u32, ring: u32, pos: u32, layout: u32, norm: f32, residual: u32, topk: u32 = 0 };
const QkvPermA = extern struct { kv: u32, heads: u32, dim: u32, rel: u32, extent: u32, layout: u32, dyn: u32, tau: u32, kv_stride_rows: u32, kv_row: u32, norm: f32, q_threshold: f32, rel_threshold: f32, eps: f32, floor: f32, alpha: f32 };
const SplitPermRow = extern struct { keys: u64, values: u64, args: DecodeParts, kv_stride_rows: u32, pad: u32 = 0 };
const ConvRow = extern struct { ring: u64, pos: u32, pad: u32 = 0 };
const ScatterA = extern struct { width: u32, stride: u32, slot: u32 };
const AffineBatchCtlA = extern struct { k: u32, m0: u32, m1: u32, h: u32, cxs: u32, cxb: u32, mxs: u32, mxb: u32, ys: u32, yb: u32 };
const PickA = extern struct { experts: u32, topk: u32, rows: u32, route_scale: f32 };
const MoeBA = extern struct { k: u32, n: u32, slots: u32, topk: u32, gate_cap: f32, up_cap: f32 };
const ArgmaxA = extern struct { vocab: u32 };
const Argmax2A = extern struct { vocab: u32, step_idx: u32, stop_token: u32, write_loop: u32 };
pub const LoopCtrl = extern struct { cancelled: u32 = 0, stopped: u32 = 0, produced: u32 = 0, pad: u32 = 0 };

pub const Pipes = struct {
    embed: mtl.Pipeline,
    batch_proj: mtl.Pipeline,
    dense: mtl.Pipeline,
    float_mv: mtl.Pipeline,
    batch_float_ctl: mtl.Pipeline,
    aff_up: mtl.Pipeline,
    aff_down: mtl.Pipeline,
    qkv: mtl.Pipeline,
    split: mtl.Pipeline,
    combine: mtl.Pipeline,
    ctl_norm: mtl.Pipeline,
    ctl_down: mtl.Pipeline,
    ctl_apply: mtl.Pipeline,
    argmax: mtl.Pipeline,
    copy: mtl.Pipeline,
    sig_mul: mtl.Pipeline,
    pass_copy: mtl.Pipeline,
    rms: mtl.Pipeline,
    rms3: mtl.Pipeline,
    rms_pair: mtl.Pipeline,
    batch_conv: mtl.Pipeline,
    scatter: mtl.Pipeline,
    routepick: mtl.Pipeline,
    gu_nv: mtl.Pipeline,
    batch_mix: mtl.Pipeline,
    batch_mix2: mtl.Pipeline,
    down1_nv: mtl.Pipeline,
    aot_embed_sb: ?mtl.Pipeline = null,
    aot_dense8_sb: ?mtl.Pipeline = null,
    aot_batch_proj8_sb: ?mtl.Pipeline = null,
    aot_expert_up4_sb: ?mtl.Pipeline = null,
    aot_expert_down4_sb: ?mtl.Pipeline = null,
    aot_combine_sig_mul: ?mtl.Pipeline = null,
    aot_gu_nv: ?mtl.Pipeline = null,
    aot_down1_nv: ?mtl.Pipeline = null,
    aot_mix2_inv: ?mtl.Pipeline = null,
    aot_mix_inv: ?mtl.Pipeline = null,
    aot_ctl_down3: ?mtl.Pipeline = null,
    aot_ctl_down1: ?mtl.Pipeline = null,
    aot_ctl_apply3_gf2: ?mtl.Pipeline = null,
    aot_ctl_apply1_gf2: ?mtl.Pipeline = null,
    aot_gu_nv_packed: ?mtl.Pipeline = null,
    aot_argmax_p1: ?mtl.Pipeline = null,
    aot_argmax_p2: ?mtl.Pipeline = null,
    aot_qkv_perm: ?mtl.Pipeline = null,
    aot_split_perm: ?mtl.Pipeline = null,

    pub fn load(device: mtl.Device, lib: mtl.Library, archive: ba.BinaryArchive, layout_flags: u32) !Pipes {
        const ind = (layout_flags & fmt.layout_non_indirect) == 0;
        const has_fused = (layout_flags & fmt.layout_fused_kernels) != 0;
        var res = Pipes{
            .embed = try archive.loadPipeline(device, lib, "i2a_embed", ind),
            .batch_proj = try archive.loadPipeline(device, lib, "i2a_batch_projections", ind),
            .dense = try archive.loadPipeline(device, lib, "i2a_dense", ind),
            .float_mv = try archive.loadPipeline(device, lib, "i2a_float", ind),
            .batch_float_ctl = try archive.loadPipeline(device, lib, "i2a_batch_float_control", ind),
            .aff_up = try archive.loadPipeline(device, lib, "i2a_expert_up", ind),
            .aff_down = try archive.loadPipeline(device, lib, "i2a_expert_down", ind),
            .qkv = try archive.loadPipeline(device, lib, "i2_batch_qkv", ind),
            .split = try archive.loadPipeline(device, lib, "i2_batch_split", ind),
            .combine = try archive.loadPipeline(device, lib, "i2_batch_combine", ind),
            .ctl_norm = try archive.loadPipeline(device, lib, "i2_controls_norm", ind),
            .ctl_down = try archive.loadPipeline(device, lib, "i2_controls_down", ind),
            .ctl_apply = try archive.loadPipeline(device, lib, "i2_controls_apply", ind),
            .argmax = try archive.loadPipeline(device, lib, "i2v_argmax", ind),
            .copy = try archive.loadPipeline(device, lib, "i2_copy", ind),
            .sig_mul = try archive.loadPipeline(device, lib, "i2_sig_mul", ind),
            .pass_copy = try archive.loadPipeline(device, lib, "i2p_pass_copy", ind),
            .rms = try archive.loadPipeline(device, lib, "i2p_rms", ind),
            .rms3 = try archive.loadPipeline(device, lib, "i2p_rms3", ind),
            .rms_pair = try archive.loadPipeline(device, lib, "i2p_rms_pair", ind),
            .batch_conv = try archive.loadPipeline(device, lib, "i2p_batch_conv", ind),
            .scatter = try archive.loadPipeline(device, lib, "i2p_scatter", ind),
            .routepick = try archive.loadPipeline(device, lib, "i2p_routepick", ind),
            .gu_nv = try archive.loadPipeline(device, lib, "i2p_gu", ind),
            .batch_mix = try archive.loadPipeline(device, lib, "i2pr_batch_mix", ind),
            .batch_mix2 = try archive.loadPipeline(device, lib, "i2pr_batch_mix2", ind),
            .down1_nv = try archive.loadPipeline(device, lib, "i2e_down1", ind),
        };
        if (has_fused) {
            res.aot_embed_sb = try archive.loadPipeline(device, lib, "i2_aot_embed_sb", ind);
            res.aot_dense8_sb = try archive.loadPipeline(device, lib, "i2_aot_dense8_sb", ind);
            res.aot_batch_proj8_sb = try archive.loadPipeline(device, lib, "i2_aot_batch_proj8_sb", ind);
            res.aot_expert_up4_sb = try archive.loadPipeline(device, lib, "i2_aot_expert_up4_sb", ind);
            res.aot_expert_down4_sb = try archive.loadPipeline(device, lib, "i2_aot_expert_down4_sb", ind);
            res.aot_combine_sig_mul = try archive.loadPipeline(device, lib, "i2_aot_combine_sig_mul", ind);
            res.aot_gu_nv = try archive.loadPipeline(device, lib, "i2_aot_gu_nv", ind);
            res.aot_down1_nv = try archive.loadPipeline(device, lib, "i2_aot_down1_nv", ind);
            res.aot_mix2_inv = try archive.loadPipeline(device, lib, "i2_aot_mix2_inv", ind);
            res.aot_mix_inv = try archive.loadPipeline(device, lib, "i2_aot_mix_inv", ind);
            res.aot_ctl_down3 = try archive.loadPipeline(device, lib, "i2_aot_ctl_down3", ind);
            res.aot_ctl_down1 = try archive.loadPipeline(device, lib, "i2_aot_ctl_down1", ind);
            res.aot_ctl_apply3_gf2 = try archive.loadPipeline(device, lib, "i2_aot_ctl_apply3_gf2", ind);
            res.aot_ctl_apply1_gf2 = try archive.loadPipeline(device, lib, "i2_aot_ctl_apply1_gf2", ind);
            res.aot_gu_nv_packed = try archive.loadPipeline(device, lib, "i2_aot_gu_nv_packed", ind);
            res.aot_argmax_p1 = try archive.loadPipeline(device, lib, "i2_aot_argmax_p1", ind);
            res.aot_argmax_p2 = try archive.loadPipeline(device, lib, "i2_aot_argmax_p2", ind);
            res.aot_qkv_perm = try archive.loadPipeline(device, lib, "i2_aot_qkv_perm", ind);
            res.aot_split_perm = try archive.loadPipeline(device, lib, "i2_aot_split_perm", ind);
        }
        return res;
    }

    pub fn deinit(p: *Pipes) void {
        p.embed.deinit();
        p.batch_proj.deinit();
        p.dense.deinit();
        p.float_mv.deinit();
        p.batch_float_ctl.deinit();
        p.aff_up.deinit();
        p.aff_down.deinit();
        p.qkv.deinit();
        p.split.deinit();
        p.combine.deinit();
        p.ctl_norm.deinit();
        p.ctl_down.deinit();
        p.ctl_apply.deinit();
        p.argmax.deinit();
        p.copy.deinit();
        p.sig_mul.deinit();
        p.pass_copy.deinit();
        p.rms.deinit();
        p.rms3.deinit();
        p.rms_pair.deinit();
        p.batch_conv.deinit();
        p.scatter.deinit();
        p.routepick.deinit();
        p.gu_nv.deinit();
        p.batch_mix.deinit();
        p.batch_mix2.deinit();
        p.down1_nv.deinit();
        if (p.aot_embed_sb) |x| x.deinit();
        if (p.aot_dense8_sb) |x| x.deinit();
        if (p.aot_batch_proj8_sb) |x| x.deinit();
        if (p.aot_expert_up4_sb) |x| x.deinit();
        if (p.aot_expert_down4_sb) |x| x.deinit();
        if (p.aot_combine_sig_mul) |x| x.deinit();
        if (p.aot_gu_nv) |x| x.deinit();
        if (p.aot_down1_nv) |x| x.deinit();
        if (p.aot_mix2_inv) |x| x.deinit();
        if (p.aot_mix_inv) |x| x.deinit();
        if (p.aot_ctl_down3) |x| x.deinit();
        if (p.aot_ctl_down1) |x| x.deinit();
        if (p.aot_ctl_apply3_gf2) |x| x.deinit();
        if (p.aot_ctl_apply1_gf2) |x| x.deinit();
        if (p.aot_gu_nv_packed) |x| x.deinit();
        if (p.aot_argmax_p1) |x| x.deinit();
        if (p.aot_argmax_p2) |x| x.deinit();
        if (p.aot_qkv_perm) |x| x.deinit();
        if (p.aot_split_perm) |x| x.deinit();
    }
};

pub const LoadStats = struct {
    verify_ms: f64 = 0,
    weights_ms: f64 = 0,
    pipelines_ms: f64 = 0,
    residency_ms: f64 = 0,
    total_ms: f64 = 0,
    weights_bytes: u64 = 0,
    use_mmap: bool = false,
};

pub const StepStats = struct {
    wall_ms: f64 = 0,
    gpu_ms: f64 = 0,
    dispatches: u32 = 0,
    command_buffers: u32 = 0,
    syncs: u32 = 0,
    token: u32 = 0,
};

fn attnPartition(n: usize) struct { splits: u32, chunk: u32 } {
    const splits: u32 = @intCast(@max(1, @min(attn_splits, (n + attn_chunk - 1) / attn_chunk)));
    return .{ .splits = splits, .chunk = @intCast((n + splits - 1) / splits) };
}

fn sha256File(io: std.Io, path: []const u8) !struct { size: u64, digest: [32]u8 } {
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.MissingFile;
    defer file.close(io);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 20]u8 = undefined;
    var total: u64 = 0;
    while (true) {
        const n = file.readStreaming(io, &.{&buf}) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        if (n == 0) break;
        hasher.update(buf[0..n]);
        total += n;
    }
    return .{ .size = total, .digest = hasher.finalResult() };
}

const max_weight_chunks = fmt.max_weight_chunks;

const WeightChunk = struct {
    buf: mtl.Buffer,
    start: u64,
    end: u64,
};

fn loadWeightsPreadChunks(device: mtl.Device, path: [:0]const u8, sched: *const fmt.Schedule, chunks: *[max_weight_chunks]WeightChunk) !usize {
    const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.MissingFile;
    defer _ = std.c.close(fd);
    _ = std.c.fcntl(fd, F_NOCACHE, @as(c_int, 1));
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or @as(u64, @intCast(end)) != sched.weights_size) return error.SizeMismatch;

    const plan = try fmt.planWeightChunks(sched);
    var loaded: usize = 0;
    errdefer for (0..loaded) |i| chunks[i].buf.deinit();

    if (sched.chunk_count != 0 and sched.chunk_count != plan.count) return error.CorruptArtifact;
    if (sched.chunk_count == plan.count and plan.count > 0) {
        for (0..plan.count) |ci| {
            const cs = sched.chunk_starts[ci];
            const ce = sched.chunk_ends[ci];
            if (cs != plan.starts[ci] or ce != plan.ends[ci]) return error.CorruptArtifact;
            const clen: usize = @intCast(ce - cs);
            const buf = try device.buffer(clen, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            chunks[ci] = .{ .buf = buf, .start = cs, .end = ce };
            loaded += 1;
        }
        const ChunkRead = struct {
            fd: std.c.fd_t,
            dst: []u8,
            start: u64,
            expected: [32]u8,
            err: ?anyerror = null,
            fn run(task: *@This()) void {
                var hasher = std.crypto.hash.sha2.Sha256.init(.{});
                var off: usize = 0;
                while (off < task.dst.len) {
                    const want = @min(task.dst.len - off, 16 << 20);
                    const got = std.c.pread(task.fd, task.dst.ptr + off, want, @intCast(task.start + off));
                    if (got <= 0) {
                        task.err = error.CorruptArtifact;
                        return;
                    }
                    const n: usize = @intCast(got);
                    hasher.update(task.dst[off .. off + n]);
                    off += n;
                }
                const actual = hasher.finalResult();
                if (!std.mem.eql(u8, &actual, &task.expected)) task.err = error.DigestMismatch;
            }
        };
        var tasks: [max_weight_chunks]ChunkRead = undefined;
        var threads: [max_weight_chunks]std.Thread = undefined;
        var spawned: usize = 0;
        errdefer for (0..spawned) |ti| threads[ti].join();
        for (0..plan.count) |ci| {
            const clen: usize = @intCast(chunks[ci].end - chunks[ci].start);
            tasks[ci] = .{
                .fd = fd,
                .dst = chunks[ci].buf.contents()[0..clen],
                .start = chunks[ci].start,
                .expected = sched.chunk_sha256[ci],
            };
            threads[ci] = try std.Thread.spawn(.{}, ChunkRead.run, .{&tasks[ci]});
            spawned += 1;
        }
        for (0..spawned) |ci| threads[ci].join();
        spawned = 0;
        for (0..plan.count) |ci| if (tasks[ci].err) |e| return e;
        return loaded;
    }

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (0..plan.count) |ci| {
        const cs = plan.starts[ci];
        const ce = plan.ends[ci];
        const clen: usize = @intCast(ce - cs);
        const buf = try device.buffer(clen, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buf.deinit();
        const dst = buf.contents()[0..clen];
        var off: usize = 0;
        while (off < clen) {
            const want = @min(clen - off, 16 << 20);
            const got = std.c.pread(fd, dst.ptr + off, want, @intCast(cs + off));
            if (got <= 0) return error.CorruptArtifact;
            const n: usize = @intCast(got);
            hasher.update(dst[off .. off + n]);
            off += n;
        }
        chunks[ci] = .{ .buf = buf, .start = cs, .end = ce };
        loaded += 1;
    }
    const actual = hasher.finalResult();
    if (!std.mem.eql(u8, &actual, &sched.weights_sha256)) return error.DigestMismatch;
    return loaded;
}

fn loadWeightsMmapChunks(device: mtl.Device, map: mtl.MappedFile, sched: *const fmt.Schedule, chunks: *[max_weight_chunks]WeightChunk) !usize {
    const page: u64 = std.heap.pageSize();
    const plan = try fmt.planWeightChunks(sched);
    if (sched.chunk_count != 0 and sched.chunk_count != plan.count) return error.CorruptArtifact;
    if (sched.chunk_count == plan.count and plan.count > 0) {
        const ChunkVerify = struct {
            bytes: []const u8,
            expected: [32]u8,
            ok: bool = false,
            fn run(task: *@This()) void {
                var hasher = std.crypto.hash.sha2.Sha256.init(.{});
                hasher.update(task.bytes);
                const actual = hasher.finalResult();
                task.ok = std.mem.eql(u8, &actual, &task.expected);
            }
        };
        var tasks: [max_weight_chunks]ChunkVerify = undefined;
        var threads: [max_weight_chunks]std.Thread = undefined;
        var spawned: usize = 0;
        errdefer for (0..spawned) |ti| threads[ti].join();
        for (0..plan.count) |ci| {
            const cs = sched.chunk_starts[ci];
            const ce = sched.chunk_ends[ci];
            if (cs != plan.starts[ci] or ce != plan.ends[ci]) return error.CorruptArtifact;
            tasks[ci] = .{
                .bytes = map.bytes[@intCast(cs)..@intCast(ce)],
                .expected = sched.chunk_sha256[ci],
            };
            threads[ci] = try std.Thread.spawn(.{}, ChunkVerify.run, .{&tasks[ci]});
            spawned += 1;
        }
        for (0..spawned) |ci| threads[ci].join();
        spawned = 0;
        for (0..plan.count) |ci| if (!tasks[ci].ok) return error.DigestMismatch;
    } else {
        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        hasher.update(map.bytes[0..@intCast(sched.weights_size)]);
        const actual = hasher.finalResult();
        if (!std.mem.eql(u8, &actual, &sched.weights_sha256)) return error.DigestMismatch;
    }
    var loaded: usize = 0;
    errdefer for (0..loaded) |i| chunks[i].buf.deinit();

    for (0..plan.count) |ci| {
        const base = plan.starts[ci] & ~(page - 1);
        const end_aligned = @min(@as(u64, map.bytes.len), (plan.ends[ci] + page - 1) & ~(page - 1));
        const len: usize = @intCast(end_aligned - base);
        const buf = try device.bufferNoCopy(map.bytes.ptr + @as(usize, @intCast(base)), len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        chunks[ci] = .{ .buf = buf, .start = base, .end = plan.ends[ci] };
        loaded += 1;
    }
    return loaded;
}

pub const Runtime = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    sched: fmt.Schedule,
    weight_chunks: [max_weight_chunks]WeightChunk = undefined,
    weight_chunk_count: usize = 0,
    mapped_weights: ?mtl.MappedFile = null,
    pipes: Pipes,
    extra_pipes: ?prefill_mod.ExtraPipes = null,
    prefill_scratch: ?prefill_mod.PrefillScratch = null,
    verify_scratch: ?prefill_mod.VerifyScratch = null,
    residency: ?mtl.ResidencySet = null,
    stats: LoadStats = .{},

    capacity: usize = 0,
    len: usize = 0,
    keys: mtl.Buffer,
    values: mtl.Buffer,
    k_conv: mtl.Buffer,
    v_conv: mtl.Buffer,
    attn_conv: mtl.Buffer,
    mlp_conv: mtl.Buffer,
    slot_kv_off: [fmt.num_slots]usize = @splat(0),
    slot_full: [fmt.num_slots]bool = @splat(false),
    slot_visit: [fmt.num_slots]usize = @splat(0),

    ids: mtl.Buffer,
    x: mtl.Buffer,
    prelude: mtl.Buffer,
    passout: mtl.Buffer,
    bank: mtl.Buffer,
    n0: mtl.Buffer,
    n1: mtl.Buffer,
    n2: mtl.Buffer,
    m0: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    r: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    kdyn: mtl.Buffer,
    vdyn: mtl.Buffer,
    sink: mtl.Buffer,
    relbias: mtl.Buffer,
    attn: mtl.Buffer,
    gate: mtl.Buffer,
    projout: mtl.Buffer,
    o: mtl.Buffer,
    aconv: mtl.Buffer,
    moe: mtl.Buffer,
    blocks: mtl.Buffer,
    mixout: mtl.Buffer,
    msg: mtl.Buffer,
    h2in: mtl.Buffer,
    normed: mtl.Buffer,
    router: mtl.Buffer,
    picks: mtl.Buffer,
    wts: mtl.Buffer,
    act: mtl.Buffer,
    down: mtl.Buffer,
    logits: mtl.Buffer,
    parts: mtl.Buffer,
    ctl_in: mtl.Buffer,
    ctl_hid: mtl.Buffer,
    ctl_inv: mtl.Buffer,
    argmax_out: mtl.Buffer,
    argmax_tmp_v: mtl.Buffer,
    argmax_tmp_i: mtl.Buffer,
    token_ring: mtl.Buffer,
    loop_ctrl: mtl.Buffer,
    opt_mask: u32 = 0,

    param_arena: mtl.Buffer,
    icb_pre: ?mtl.IndirectCommandBuffer = null,
    icb_main: ?mtl.IndirectCommandBuffer = null,
    icb_pre_count: usize = 0,
    icb_main_count: usize = 0,
    icb_built_argmax: bool = false,
    dyn_qkv_row_off: [fmt.num_slots]usize = @splat(0),
    dyn_attn_row_off: [fmt.num_slots]usize = @splat(0),
    dyn_attn_split_cmd: [fmt.num_slots]usize = @splat(0),
    dyn_aconv_row_off: [fmt.num_slots]usize = @splat(0),
    dyn_mconv_row_off: [fmt.num_slots]usize = @splat(0),

    pub const OpenOpts = struct {
        capacity: usize = 4096,
        use_mmap: bool = false,
        device_arch_override: ?[]const u8 = null,
        verify_only: bool = false,
        opt_mask: u32 = 0,
    };

    /// Opens and verifies an AOT bundle, returning a heap-allocated `Runtime` owned by the caller.
    /// Release all GPU resources and free the handle via `Runtime.deinit`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, bundle_dir: []const u8, opts: OpenOpts) !*Runtime {
        const t_total = std.Io.Clock.awake.now(io).toNanoseconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();

        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const manifest_path = try std.fs.path.join(arena, &.{ bundle_dir, "manifest.json" });
        const manifest_bytes = std.Io.Dir.cwd().readFileAlloc(io, manifest_path, arena, .limited(4 << 20)) catch return error.MissingFile;
        const parsed_manifest = try std.json.parseFromSlice(fmt.Manifest, arena, manifest_bytes, .{ .ignore_unknown_fields = true });
        const manifest = parsed_manifest.value;
        if (manifest.schema != fmt.schema_version) return error.BadVersion;

        const device = try mtl.Device.init();
        errdefer device.deinit();
        var arch_buf: [32]u8 = undefined;
        const actual_arch = opts.device_arch_override orelse ba.deviceArch(device, &arch_buf);
        const actual_gen: u32 = if (opts.device_arch_override) |ov| mtl.generationOf(ov) else device.generation();
        try fmt.checkDevice(actual_arch, actual_gen, manifest.target_arch, manifest.min_generation);

        var weights_entry: ?fmt.FileEntry = null;
        for (manifest.files) |fe| {
            if (std.mem.eql(u8, fe.name, "weights.aot")) {
                weights_entry = fe;
                continue;
            }
            const p = try std.fs.path.join(arena, &.{ bundle_dir, fe.name });
            const h_ = try sha256File(io, p);
            if (h_.size != fe.size) return error.SizeMismatch;
            const hex = fmt.hexEncode(h_.digest);
            if (!std.mem.eql(u8, &hex, fe.sha256)) return error.DigestMismatch;
        }
        const we = weights_entry orelse return error.MissingFile;

        const sched_path = try std.fs.path.join(arena, &.{ bundle_dir, "schedule.bin" });
        const sched_bytes = try std.Io.Dir.cwd().readFileAlloc(io, sched_path, arena, .limited(4 << 20));
        const sched = try fmt.validateScheduleHeader(sched_bytes);
        try fmt.checkDevice(actual_arch, actual_gen, sched.archName(), sched.min_generation);
        if (sched.weights_size != we.size) return error.SizeMismatch;
        const sched_w_hex = fmt.hexEncode(sched.weights_sha256);
        if (!std.mem.eql(u8, &sched_w_hex, we.sha256)) return error.DigestMismatch;
        if (manifest.context_cap != sched.geom.context_cap or manifest.layout_flags != sched.layout_flags) return error.CorruptArtifact;
        if (opts.capacity > sched.geom.context_cap) return error.ContextTooLarge;
        const t_verify = std.Io.Clock.awake.now(io).toNanoseconds();

        // Metal ICB setKernelBuffer:offset:atIndex: truncates offsets >= 4 GiB on Apple Silicon.
        const weights_path_z = try std.fs.path.joinZ(arena, &.{ bundle_dir, "weights.aot" });
        var mapped_weights: ?mtl.MappedFile = null;
        var weight_chunks: [max_weight_chunks]WeightChunk = undefined;
        var weight_chunk_count: usize = 0;
        if (opts.verify_only) {
            const h_ = try sha256File(io, weights_path_z);
            if (h_.size != we.size) return error.SizeMismatch;
            const hex = fmt.hexEncode(h_.digest);
            if (!std.mem.eql(u8, &hex, we.sha256)) return error.DigestMismatch;
            weight_chunks[0] = .{ .buf = try device.buffer(4096, mtl.ResourceOptions.shared), .start = 0, .end = 4096 };
            weight_chunk_count = 1;
        } else if (opts.use_mmap) {
            const map = try mtl.MappedFile.open(weights_path_z.ptr);
            errdefer map.deinit();
            if (map.size != sched.weights_size) return error.SizeMismatch;
            weight_chunk_count = try loadWeightsMmapChunks(device, map, &sched, &weight_chunks);
            mapped_weights = map;
        } else {
            weight_chunk_count = try loadWeightsPreadChunks(device, weights_path_z, &sched, &weight_chunks);
        }
        errdefer {
            for (0..weight_chunk_count) |ci| weight_chunks[ci].buf.deinit();
            if (mapped_weights) |m| m.deinit();
        }
        const t_weights = std.Io.Clock.awake.now(io).toNanoseconds();

        const metallib_path = try std.fs.path.join(arena, &.{ bundle_dir, "kernels.metallib" });
        const archive_path = try std.fs.path.join(arena, &.{ bundle_dir, "pipelines.mtlarchive" });
        const lib = try mtl.Library.fromFile(device, metallib_path);
        defer lib.deinit();
        const archive = try ba.BinaryArchive.open(device, archive_path);
        defer archive.deinit();
        var pipes = try Pipes.load(device, lib, archive, sched.layout_flags);
        errdefer pipes.deinit();
        const ind = (sched.layout_flags & fmt.layout_non_indirect) == 0;
        var extra_pipes: ?prefill_mod.ExtraPipes = if ((sched.layout_flags & fmt.layout_fused_kernels) != 0)
            try prefill_mod.ExtraPipes.load(device, lib, archive, ind)
        else
            null;
        errdefer if (extra_pipes) |*ep| ep.deinit();
        const t_pipes = std.Io.Clock.awake.now(io).toNanoseconds();

        const queue = try device.queue();
        errdefer queue.deinit();

        const rt = try gpa.create(Runtime);
        errdefer gpa.destroy(rt);

        const g = sched.geom;
        const h: usize = g.hidden;
        const qw: usize = g.heads * g.head_dim;
        const kvw: usize = g.kv_heads * g.head_dim;
        const cap = if (opts.verify_only) 16 else opts.capacity;
        const kv_row_bytes = kvw * 2;
        var slot_full: [fmt.num_slots]bool = @splat(false);
        var slot_visit: [fmt.num_slots]usize = @splat(0);
        for (sched.slots, 0..) |sl, visit_i| {
            slot_full[sl.slot] = sl.is_full != 0;
            slot_visit[sl.slot] = visit_i;
        }
        var slot_kv_off: [fmt.num_slots]usize = @splat(0);
        var kv_total: usize = 0;
        for (0..fmt.num_slots) |sidx| {
            slot_kv_off[sidx] = kv_total;
            const rows: usize = if (slot_full[sidx]) cap else g.window;
            kv_total += rows * kv_row_bytes;
        }

        const priv = mtl.ResourceOptions.private;
        const shr = mtl.ResourceOptions.shared;

        rt.* = undefined;
        rt.gpa = gpa;
        rt.device = device;
        rt.queue = queue;
        rt.sched = sched;
        rt.weight_chunks = weight_chunks;
        rt.weight_chunk_count = weight_chunk_count;
        rt.mapped_weights = mapped_weights;
        rt.pipes = pipes;
        rt.extra_pipes = extra_pipes;
        rt.prefill_scratch = null;
        rt.verify_scratch = null;
        rt.residency = null;
        rt.stats = .{};
        rt.capacity = cap;
        rt.len = 0;
        rt.slot_kv_off = slot_kv_off;
        rt.slot_full = slot_full;
        rt.slot_visit = slot_visit;
        rt.opt_mask = opts.opt_mask;
        rt.icb_pre = null;
        rt.icb_main = null;
        rt.icb_pre_count = 0;
        rt.icb_main_count = 0;
        rt.icb_built_argmax = false;
        rt.dyn_qkv_row_off = @splat(0);
        rt.dyn_attn_row_off = @splat(0);
        rt.dyn_attn_split_cmd = @splat(0);
        rt.dyn_aconv_row_off = @splat(0);
        rt.dyn_mconv_row_off = @splat(0);

        const BufSpec = struct { len: usize, opts: u32 };
        const buf_names = [_][]const u8{
            "keys", "values", "k_conv", "v_conv", "attn_conv", "mlp_conv", "ids", "x", "prelude",
            "passout", "bank", "n0", "n1", "n2", "m0", "q", "k", "v", "r", "kc", "vc", "kdyn",
            "vdyn", "sink", "relbias", "attn", "gate", "projout", "o", "aconv", "moe", "blocks",
            "mixout", "msg", "h2in", "normed", "router", "picks", "wts", "act", "down", "logits",
            "parts", "ctl_in", "ctl_hid", "ctl_inv", "argmax_out", "argmax_tmp_v", "argmax_tmp_i",
            "token_ring", "loop_ctrl", "param_arena",
        };
        const buf_specs = [buf_names.len]BufSpec{
            .{ .len = kv_total, .opts = priv },
            .{ .len = kv_total, .opts = priv },
            .{ .len = fmt.num_slots * conv_ring * kvw * 4, .opts = priv },
            .{ .len = fmt.num_slots * conv_ring * kvw * 4, .opts = priv },
            .{ .len = fmt.num_slots * conv_ring * h * 4, .opts = priv },
            .{ .len = fmt.num_slots * conv_ring * h * 4, .opts = priv },
            .{ .len = 16 * 4, .opts = shr },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = mix_states * h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = qw * 4, .opts = priv },
            .{ .len = kvw * 4, .opts = priv },
            .{ .len = kvw * 4, .opts = priv },
            .{ .len = g.heads * g.d_rel * 4, .opts = priv },
            .{ .len = kvw * 4, .opts = priv },
            .{ .len = kvw * 4, .opts = priv },
            .{ .len = g.kv_heads * 4 * 4, .opts = priv },
            .{ .len = g.kv_heads * 4 * 4, .opts = priv },
            .{ .len = g.heads * 4, .opts = priv },
            .{ .len = g.heads * (g.rel_extent + 1) * 4, .opts = priv },
            .{ .len = qw * 4, .opts = priv },
            .{ .len = qw * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = h * 4, .opts = priv },
            .{ .len = mix_states * h * 4, .opts = priv },
            .{ .len = 4 * h * 4, .opts = priv },
            .{ .len = 4 * h * 4, .opts = priv },
            .{ .len = 2 * h * 4, .opts = priv },
            .{ .len = g.gate_rank * 4, .opts = priv },
            .{ .len = g.experts * 4, .opts = priv },
            .{ .len = g.topk * 4, .opts = priv },
            .{ .len = g.topk * 4, .opts = priv },
            .{ .len = g.topk * g.expert_inter * 4, .opts = priv },
            .{ .len = g.topk * h * 4, .opts = priv },
            .{ .len = g.unpadded_vocab * 4, .opts = shr },
            .{ .len = g.heads * attn_splits * (g.head_dim + 2) * 4, .opts = priv },
            .{ .len = 3 * 2 * h * 4, .opts = priv },
            .{ .len = 3 * g.gate_rank * 4, .opts = priv },
            .{ .len = 16 * 4, .opts = priv },
            .{ .len = 16 * 4, .opts = shr },
            .{ .len = 128 * 4, .opts = priv },
            .{ .len = 128 * 4, .opts = priv },
            .{ .len = 64 * 4, .opts = shr },
            .{ .len = @sizeOf(LoopCtrl), .opts = shr },
            .{ .len = 256 * 1024, .opts = shr },
        };
        var built_bufs: [buf_names.len]mtl.Buffer = undefined;
        var built_count: usize = 0;
        errdefer for (built_bufs[0..built_count]) |b| b.deinit();
        inline for (buf_names, 0..) |name, i| {
            const buf = try device.buffer(buf_specs[i].len, buf_specs[i].opts);
            built_bufs[built_count] = buf;
            built_count += 1;
            @field(rt, name) = buf;
        }

        if (!opts.verify_only) {
            if (device.residencySet(64)) |rs| {
                rt.addAllToResidency(rs);
                rs.commit();
                rs.requestResidency();
                queue.addResidencySet(rs);
                rt.residency = rs;
            } else |err| switch (err) {
                error.Unsupported => {},
                else => return err,
            }
        }
        const t_res = std.Io.Clock.awake.now(io).toNanoseconds();

        rt.stats = .{
            .verify_ms = @as(f64, @floatFromInt(t_verify - t_total)) / 1e6,
            .weights_ms = @as(f64, @floatFromInt(t_weights - t_verify)) / 1e6,
            .pipelines_ms = @as(f64, @floatFromInt(t_pipes - t_weights)) / 1e6,
            .residency_ms = @as(f64, @floatFromInt(t_res - t_pipes)) / 1e6,
            .total_ms = @as(f64, @floatFromInt(t_res - t_total)) / 1e6,
            .weights_bytes = sched.weights_size,
            .use_mmap = opts.use_mmap,
        };
        return rt;
    }

    fn addAllToResidency(rt: *Runtime, rs: mtl.ResidencySet) void {
        for (0..rt.weight_chunk_count) |ci| rs.add(rt.weight_chunks[ci].buf);
        const bufs = [_]mtl.Buffer{
            rt.keys, rt.values, rt.k_conv, rt.v_conv, rt.attn_conv, rt.mlp_conv,
            rt.ids, rt.x, rt.prelude, rt.passout, rt.bank, rt.n0, rt.n1, rt.n2, rt.m0,
            rt.q, rt.k, rt.v, rt.r, rt.kc, rt.vc, rt.kdyn, rt.vdyn, rt.sink, rt.relbias,
            rt.attn, rt.gate, rt.projout, rt.o, rt.aconv, rt.moe, rt.blocks, rt.mixout,
            rt.msg, rt.h2in, rt.normed, rt.router, rt.picks, rt.wts, rt.act, rt.down,
            rt.logits, rt.parts, rt.ctl_in, rt.ctl_hid, rt.ctl_inv, rt.argmax_out,
            rt.argmax_tmp_v, rt.argmax_tmp_i, rt.token_ring, rt.loop_ctrl, rt.param_arena,
        };
        for (bufs) |b| rs.add(b);
    }

    pub inline fn setW(rt: *const Runtime, target: anytype, off: u64, index: usize) void {
        std.debug.assert(rt.weight_chunk_count > 0);
        std.debug.assert(off <= rt.sched.weights_size);
        var i: usize = rt.weight_chunk_count;
        while (i > 0) {
            i -= 1;
            const c = rt.weight_chunks[i];
            if (off >= c.start) {
                target.setBuffer(c.buf, @intCast(off - c.start), index);
                return;
            }
        }
        target.setBuffer(rt.weight_chunks[0].buf, 0, index);
    }

    pub fn slotIsFull(rt: *const Runtime, slot: u32) bool {
        return rt.slot_full[slot];
    }

    pub fn ensurePrefillScratch(rt: *Runtime) !*prefill_mod.PrefillScratch {
        if (rt.prefill_scratch == null) {
            rt.prefill_scratch = try prefill_mod.PrefillScratch.init(rt.device, rt.sched.geom, prefill_mod.default_chunk);
        }
        return &rt.prefill_scratch.?;
    }

    pub fn ensureVerifyScratch(rt: *Runtime) !*prefill_mod.VerifyScratch {
        if (rt.verify_scratch == null) {
            rt.verify_scratch = try prefill_mod.VerifyScratch.init(rt.device, rt.sched.geom);
        }
        return &rt.verify_scratch.?;
    }

    pub fn prefillPrompt(rt: *Runtime, ids: []const u32, want_last_logits: bool, half_inputs: bool, capture_taps: bool) !prefill_mod.PrefillStats {
        return prefill_mod.prefillChunk(rt, ids, want_last_logits, half_inputs, capture_taps);
    }

    pub fn verifyTree(rt: *Runtime, ids: []const u32, parents: ?[]const u8, commit: bool, capture_taps: bool) !prefill_mod.VerifyStats {
        return prefill_mod.verifyTree(rt, ids, parents, commit, capture_taps);
    }

    pub fn convertKvToPerm(rt: *Runtime) !void {
        return prefill_mod.convertKvToPerm(rt);
    }

    pub fn deinit(rt: *Runtime) void {
        if (rt.prefill_scratch) |*ps| ps.deinit();
        if (rt.verify_scratch) |*vs| vs.deinit();
        if (rt.icb_pre) |icb| icb.deinit();
        if (rt.icb_main) |icb| icb.deinit();
        if (rt.residency) |rs| {
            rt.queue.removeResidencySet(rs);
            rs.deinit();
        }
        const bufs = [_]mtl.Buffer{
            rt.keys, rt.values, rt.k_conv, rt.v_conv, rt.attn_conv, rt.mlp_conv,
            rt.ids, rt.x, rt.prelude, rt.passout, rt.bank, rt.n0, rt.n1, rt.n2, rt.m0,
            rt.q, rt.k, rt.v, rt.r, rt.kc, rt.vc, rt.kdyn, rt.vdyn, rt.sink, rt.relbias,
            rt.attn, rt.gate, rt.projout, rt.o, rt.aconv, rt.moe, rt.blocks, rt.mixout,
            rt.msg, rt.h2in, rt.normed, rt.router, rt.picks, rt.wts, rt.act, rt.down,
            rt.logits, rt.parts, rt.ctl_in, rt.ctl_hid, rt.ctl_inv, rt.argmax_out,
            rt.argmax_tmp_v, rt.argmax_tmp_i, rt.token_ring, rt.loop_ctrl, rt.param_arena,
        };
        for (bufs) |b| b.deinit();
        if (rt.extra_pipes) |*ep| ep.deinit();
        rt.pipes.deinit();
        for (0..rt.weight_chunk_count) |ci| rt.weight_chunks[ci].buf.deinit();
        if (rt.mapped_weights) |m| m.deinit();
        rt.queue.deinit();
        rt.device.deinit();
        rt.gpa.destroy(rt);
    }

    pub fn reset(rt: *Runtime) void {
        rt.len = 0;
    }

    pub const StateSnapshot = struct {
        len: usize,
        keys: mtl.Buffer,
        values: mtl.Buffer,
        k_conv: mtl.Buffer,
        v_conv: mtl.Buffer,
        attn_conv: mtl.Buffer,
        mlp_conv: mtl.Buffer,

        pub fn deinit(s: StateSnapshot) void {
            s.keys.deinit();
            s.values.deinit();
            s.k_conv.deinit();
            s.v_conv.deinit();
            s.attn_conv.deinit();
            s.mlp_conv.deinit();
        }
    };

    /// Snapshots KV and convolution caches into owned GPU buffers; release with `StateSnapshot.deinit`.
    pub fn snapshotState(rt: *Runtime) !StateSnapshot {
        const priv = mtl.ResourceOptions.private;
        const keys = try rt.device.buffer(rt.keys.length(), priv);
        errdefer keys.deinit();
        const values = try rt.device.buffer(rt.values.length(), priv);
        errdefer values.deinit();
        const k_conv = try rt.device.buffer(rt.k_conv.length(), priv);
        errdefer k_conv.deinit();
        const v_conv = try rt.device.buffer(rt.v_conv.length(), priv);
        errdefer v_conv.deinit();
        const attn_conv = try rt.device.buffer(rt.attn_conv.length(), priv);
        errdefer attn_conv.deinit();
        const mlp_conv = try rt.device.buffer(rt.mlp_conv.length(), priv);
        errdefer mlp_conv.deinit();
        const s = StateSnapshot{
            .len = rt.len,
            .keys = keys,
            .values = values,
            .k_conv = k_conv,
            .v_conv = v_conv,
            .attn_conv = attn_conv,
            .mlp_conv = mlp_conv,
        };
        const cb = rt.queue.commandBuffer();
        const bl = cb.blit();
        bl.copy(rt.keys, 0, s.keys, 0, rt.keys.length());
        bl.copy(rt.values, 0, s.values, 0, rt.values.length());
        bl.copy(rt.k_conv, 0, s.k_conv, 0, rt.k_conv.length());
        bl.copy(rt.v_conv, 0, s.v_conv, 0, rt.v_conv.length());
        bl.copy(rt.attn_conv, 0, s.attn_conv, 0, rt.attn_conv.length());
        bl.copy(rt.mlp_conv, 0, s.mlp_conv, 0, rt.mlp_conv.length());
        bl.end();
        cb.commit();
        cb.wait();
        return s;
    }

    pub fn restoreState(rt: *Runtime, s: StateSnapshot) void {
        rt.len = s.len;
        const cb = rt.queue.commandBuffer();
        const bl = cb.blit();
        bl.copy(s.keys, 0, rt.keys, 0, rt.keys.length());
        bl.copy(s.values, 0, rt.values, 0, rt.values.length());
        bl.copy(s.k_conv, 0, rt.k_conv, 0, rt.k_conv.length());
        bl.copy(s.v_conv, 0, rt.v_conv, 0, rt.v_conv.length());
        bl.copy(s.attn_conv, 0, rt.attn_conv, 0, rt.attn_conv.length());
        bl.copy(s.mlp_conv, 0, rt.mlp_conv, 0, rt.mlp_conv.length());
        bl.end();
        cb.commit();
        cb.wait();
    }

    inline fn kvRowAddr(rt: *const Runtime, base_buf: mtl.Buffer, slot: u32, pos: usize) u64 {
        const g = rt.sched.geom;
        const kvw: usize = g.kv_heads * g.head_dim;
        const is_full = rt.sched.layers[rt.sched.slots[slotVisitFor(rt, slot)].phys].is_full != 0;
        const row = if (is_full) pos else (pos % g.window);
        return base_buf.gpuAddress() + rt.slot_kv_off[slot] + row * kvw * 2;
    }

    inline fn slotVisitFor(rt: *const Runtime, slot: u32) usize {
        return rt.slot_visit[slot];
    }

    inline fn convAddr(base_buf: mtl.Buffer, slot: u32, channels: u32) u64 {
        return base_buf.gpuAddress() + @as(usize, slot) * conv_ring * channels * 4;
    }

    fn encodeStepBody(
        rt: *Runtime,
        cb0: *mtl.CommandBuffer,
        enc: *mtl.ComputeEncoder,
        cb_first: *?mtl.CommandBuffer,
        pos: u32,
        want_logits: bool,
        gpu_argmax: bool,
        ring_slot: ?u32,
        stop_token: u32,
    ) u32 {
        const g = rt.sched.geom;
        const h: u32 = g.hidden;
        const qw: u32 = g.heads * g.head_dim;
        const kvw: u32 = g.kv_heads * g.head_dim;
        const p = &rt.pipes;
        const scale_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const scale_attn: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(g.head_dim)));
        const norm_post: f32 = 1.0 / @sqrt(@as(f32, 5.0));

        const use_sb = (rt.sched.layout_flags & fmt.layout_interleaved_sb) != 0 and p.aot_dense8_sb != null;
        const use_packed_w13_gf = (rt.sched.layout_flags & fmt.layout_packed_w13_gf) != 0 and p.aot_gu_nv_packed != null and p.aot_ctl_apply3_gf2 != null;
        const opt = rt.opt_mask;
        const opt_d1 = (opt & 1) != 0 and p.aot_combine_sig_mul != null;
        const opt_d2 = ((opt & 2) != 0 or use_packed_w13_gf) and p.aot_gu_nv != null and (rt.sched.layout_flags & fmt.layout_predivided_nvfp4) != 0;
        const opt_d3 = (opt & 4) != 0 and p.aot_ctl_down3 != null;
        const opt_d9 = (opt & 256) != 0 and p.aot_qkv_perm != null and p.aot_split_perm != null;

        var dispatches: u32 = 0;

        if (use_sb) {
            enc.setPipeline(p.aot_embed_sb.?);
            enc.setBuffer(rt.ids, 0, 0);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed.w.off), 1);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed.sb.off), 2);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed_norm.off), 3);
            enc.setValue(EmbedA{ .width = h, .rows = 1, .eps = g.eps }, 4);
            enc.setBuffer(rt.x, 0, 5);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            enc.setPipeline(p.embed);
            enc.setBuffer(rt.ids, 0, 0);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed.w.off), 1);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed.s.off), 2);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed.b.off), 3);
            rt.setW(enc.*, @intCast(rt.sched.globals.embed_norm.off), 4);
            enc.setValue(EmbedA{ .width = h, .rows = 1, .eps = g.eps }, 5);
            enc.setBuffer(rt.x, 0, 6);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
        }
        dispatches += 1;

        for (rt.sched.slots) |sl| {
            const lay = &rt.sched.layers[sl.phys];
            const full = sl.is_full != 0;
            const bank = sl.is_bank != 0;
            const push = sl.push_block != 0;

            if (sl.is_pass_start != 0) {
                enc.setPipeline(p.pass_copy);
                enc.setBuffer(rt.blocks, 0, 0);
                enc.setBuffer(rt.x, 0, 1);
                enc.setBuffer(rt.prelude, 0, 2);
                enc.setBuffer(rt.bank, 0, 3);
                enc.setBuffer(rt.passout, 0, 4);
                enc.setValue(PassCopyA{ .width = h, .stride = mix_states, .start = 2, .count = sl.bank_n }, 5);
                enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            }

            var in0 = rt.x;
            var in1 = rt.x;
            var in2 = rt.x;
            var fused_norm = false;

            if (sl.phys > 0) {
                const qref = rt.sched.globals.attn_queries[sl.phys - 1];
                const kq: u32 = if (full) 3 else 1;
                if (bank) {
                    const at: usize = sl.cross_pass_at;
                    const cpq = rt.sched.globals.cross_pass_q.off;
                    const cpkv = rt.sched.globals.cross_pass_kv.off;
                    const q0_off: usize = @intCast(qref.off);
                    const q1_off: usize = if (full) q0_off + h * 4 else q0_off;
                    const q2_off: usize = if (full) q0_off + 2 * h * 4 else q0_off;
                    const args2 = [2]MixArgs{
                        .{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 0, .scale = scale_h, .eps = g.eps },
                        .{ .width = h, .depth = sl.bank_n + 1, .stride = mix_states, .k = 4, .norm = 0, .scale = scale_h, .eps = g.eps },
                    };
                    const ctl_args = ControlsArgs{
                        .h = h,
                        .rank = g.gate_rank,
                        .rows = 1,
                        .xs = .{ kq, kq, kq },
                        .xb = if (full) .{ 0, 1, 2 } else .{ 0, 0, 0 },
                    };
                    if (opt_d3) {
                        enc.setPipeline(p.aot_mix2_inv.?);
                        enc.setBuffer(rt.blocks, 0, 0);
                        enc.setBuffer(rt.x, 0, 1);
                        rt.setW(enc.*, q0_off, 2);
                        rt.setW(enc.*, q1_off, 3);
                        rt.setW(enc.*, q2_off, 4);
                        rt.setW(enc.*, q0_off, 5);
                        enc.setBuffer(rt.mixout, 0, 6);
                        enc.setBuffer(rt.bank, 0, 7);
                        enc.setBuffer(rt.passout, 0, 8);
                        rt.setW(enc.*, @intCast(cpq + (at * 2) * h * 4), 9);
                        rt.setW(enc.*, @intCast(cpkv + (at * 2) * h * 4), 10);
                        rt.setW(enc.*, @intCast(cpkv + (at * 2 + 1) * h * 4), 11);
                        rt.setW(enc.*, @intCast(cpq + (at * 2 + 1) * h * 4), 12);
                        enc.setBuffer(rt.msg, 0, 13);
                        enc.setBytes(std.mem.sliceAsBytes(&args2), 14);
                        enc.setBuffer(rt.ctl_inv, 0, 15);
                        enc.dispatchGroups(mtl.Size.of(1, 2, 1), mtl.Size.of(256, 1, 1));
                        dispatches += 1;

                        enc.setPipeline(p.aot_ctl_down3.?);
                        enc.setBuffer(rt.mixout, 0, 0);
                        enc.setBuffer(rt.msg, 0, 1);
                        enc.setBuffer(rt.ctl_inv, 0, 2);
                        rt.setW(enc.*, @intCast(lay.controllers[0].down.off), 3);
                        rt.setW(enc.*, @intCast(lay.controllers[1].down.off), 4);
                        rt.setW(enc.*, @intCast(lay.controllers[2].down.off), 5);
                        enc.setValue(ctl_args, 6);
                        enc.setBuffer(rt.ctl_hid, 0, 7);
                        enc.dispatchGroups(mtl.Size.of((g.gate_rank + 1) / 2, 1, 3), mtl.Size.of(64, 1, 1));
                        dispatches += 1;
                    } else {
                        enc.setPipeline(p.batch_mix2);
                        enc.setBuffer(rt.blocks, 0, 0);
                        enc.setBuffer(rt.x, 0, 1);
                        rt.setW(enc.*, q0_off, 2);
                        rt.setW(enc.*, q1_off, 3);
                        rt.setW(enc.*, q2_off, 4);
                        rt.setW(enc.*, q0_off, 5);
                        enc.setBuffer(rt.mixout, 0, 6);
                        enc.setBuffer(rt.bank, 0, 7);
                        enc.setBuffer(rt.passout, 0, 8);
                        rt.setW(enc.*, @intCast(cpq + (at * 2) * h * 4), 9);
                        rt.setW(enc.*, @intCast(cpkv + (at * 2) * h * 4), 10);
                        rt.setW(enc.*, @intCast(cpkv + (at * 2 + 1) * h * 4), 11);
                        rt.setW(enc.*, @intCast(cpq + (at * 2 + 1) * h * 4), 12);
                        enc.setBuffer(rt.msg, 0, 13);
                        enc.setBytes(std.mem.sliceAsBytes(&args2), 14);
                        enc.dispatchGroups(mtl.Size.of(1, 2, 1), mtl.Size.of(256, 1, 1));
                        dispatches += 1;

                        enc.setPipeline(p.ctl_norm);
                        enc.setBuffer(rt.mixout, 0, 0);
                        enc.setBuffer(rt.mixout, 0, 1);
                        enc.setBuffer(rt.mixout, 0, 2);
                        enc.setBuffer(rt.msg, 0, 3);
                        enc.setValue(ctl_args, 4);
                        enc.setBuffer(rt.ctl_in, 0, 5);
                        enc.dispatchGroups(mtl.Size.of(6, 1, 1), mtl.Size.of(256, 1, 1));
                        dispatches += 1;

                        enc.setPipeline(p.ctl_down);
                        enc.setBuffer(rt.ctl_in, 0, 0);
                        rt.setW(enc.*, @intCast(lay.controllers[0].down.off), 1);
                        rt.setW(enc.*, @intCast(lay.controllers[1].down.off), 2);
                        rt.setW(enc.*, @intCast(lay.controllers[2].down.off), 3);
                        enc.setValue(ctl_args, 4);
                        enc.setBuffer(rt.ctl_hid, 0, 5);
                        enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                        dispatches += 1;
                    }

                    if (use_packed_w13_gf) {
                        enc.setPipeline(p.aot_ctl_apply3_gf2.?);
                        enc.setBuffer(rt.ctl_hid, 0, 0);
                        rt.setW(enc.*, @intCast(lay.controllers[0].gain.off), 1);
                        rt.setW(enc.*, @intCast(lay.controllers[1].gain.off), 2);
                        rt.setW(enc.*, @intCast(lay.controllers[2].gain.off), 3);
                        enc.setBuffer(rt.mixout, 0, 4);
                        enc.setBuffer(rt.mixout, 0, 5);
                        enc.setBuffer(rt.mixout, 0, 6);
                        enc.setBuffer(rt.msg, 0, 7);
                        enc.setBuffer(rt.n0, 0, 8);
                        enc.setBuffer(rt.n1, 0, 9);
                        enc.setBuffer(rt.n2, 0, 10);
                        enc.setValue(ctl_args, 11);
                        enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                    } else {
                        enc.setPipeline(p.ctl_apply);
                        enc.setBuffer(rt.ctl_hid, 0, 0);
                        rt.setW(enc.*, @intCast(lay.controllers[0].gain.off), 1);
                        rt.setW(enc.*, @intCast(lay.controllers[1].gain.off), 2);
                        rt.setW(enc.*, @intCast(lay.controllers[2].gain.off), 3);
                        rt.setW(enc.*, @intCast(lay.controllers[0].forget.off), 4);
                        rt.setW(enc.*, @intCast(lay.controllers[1].forget.off), 5);
                        rt.setW(enc.*, @intCast(lay.controllers[2].forget.off), 6);
                        enc.setBuffer(rt.mixout, 0, 7);
                        enc.setBuffer(rt.mixout, 0, 8);
                        enc.setBuffer(rt.mixout, 0, 9);
                        enc.setBuffer(rt.msg, 0, 10);
                        enc.setBuffer(rt.n0, 0, 11);
                        enc.setBuffer(rt.n1, 0, 12);
                        enc.setBuffer(rt.n2, 0, 13);
                        enc.setValue(ctl_args, 14);
                        enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                    }
                    dispatches += 1;

                    enc.setPipeline(p.rms3);
                    enc.setBuffer(rt.n0, 0, 0);
                    enc.setBuffer(rt.n1, 0, 1);
                    enc.setBuffer(rt.n2, 0, 2);
                    rt.setW(enc.*, @intCast(lay.q_norm.off), 3);
                    rt.setW(enc.*, @intCast(lay.k_norm.off), 4);
                    rt.setW(enc.*, @intCast(lay.v_norm.off), 5);
                    enc.setValue(Rms3A{ .width = h, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 }, .eps = g.eps }, 6);
                    enc.setBuffer(rt.n0, 0, 7);
                    enc.setBuffer(rt.n1, 0, 8);
                    enc.setBuffer(rt.n2, 0, 9);
                    enc.dispatchGroups(mtl.Size.of(3, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;

                    in0 = rt.n0;
                    in1 = rt.n1;
                    in2 = rt.n2;
                    fused_norm = true;
                } else {
                    const q0_off: usize = @intCast(qref.off);
                    const q1_off: usize = if (full) q0_off + h * 4 else q0_off;
                    const q2_off: usize = if (full) q0_off + 2 * h * 4 else q0_off;
                    enc.setPipeline(p.batch_mix);
                    enc.setBuffer(rt.blocks, 0, 0);
                    enc.setBuffer(rt.x, 0, 1);
                    rt.setW(enc.*, q0_off, 2);
                    rt.setW(enc.*, q1_off, 3);
                    rt.setW(enc.*, q2_off, 4);
                    rt.setW(enc.*, q0_off, 5);
                    enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
                    enc.setBuffer(rt.mixout, 0, 7);
                    rt.setW(enc.*, @intCast(lay.q_norm.off), 8);
                    rt.setW(enc.*, @intCast(lay.k_norm.off), 9);
                    rt.setW(enc.*, @intCast(lay.v_norm.off), 10);
                    enc.setBuffer(rt.n0, 0, 11);
                    enc.setBuffer(rt.n1, 0, 12);
                    enc.setBuffer(rt.n2, 0, 13);
                    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;

                    in0 = rt.n0;
                    in1 = if (full) rt.n1 else rt.n0;
                    in2 = if (full) rt.n2 else rt.n0;
                    fused_norm = true;
                }
            }

            if (!fused_norm) {
                enc.setPipeline(p.rms);
                enc.setBuffer(rt.x, 0, 0);
                rt.setW(enc.*, @intCast(lay.q_norm.off), 1);
                enc.setValue(RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
                enc.setBuffer(rt.n0, 0, 3);
                enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
                in0 = rt.n0;
                in1 = rt.n0;
                in2 = rt.n0;
            }

            var total_proj: u32 = 0;
            for (lay.proj_ms) |m_| total_proj += m_;
            if (use_sb) {
                enc.setPipeline(p.aot_batch_proj8_sb.?);
                enc.setBuffer(in0, 0, 0);
                enc.setBuffer(in1, 0, 1);
                enc.setBuffer(in2, 0, 2);
                rt.setW(enc.*, @intCast(lay.proj_q.w.off), 3);
                rt.setW(enc.*, @intCast(lay.proj_q.sb.off), 4);
                rt.setW(enc.*, @intCast(lay.proj_k.w.off), 5);
                rt.setW(enc.*, @intCast(lay.proj_k.sb.off), 6);
                rt.setW(enc.*, @intCast(lay.proj_v.w.off), 7);
                rt.setW(enc.*, @intCast(lay.proj_v.sb.off), 8);
                enc.setBuffer(rt.q, 0, 9);
                enc.setBuffer(rt.k, 0, 10);
                enc.setBuffer(rt.v, 0, 11);
                enc.setBuffer(rt.r, 0, 12);
                enc.setBuffer(rt.gate, 0, 13);
                enc.setBuffer(rt.sink, 0, 14);
                enc.setBuffer(rt.kdyn, 0, 15);
                enc.setBuffer(rt.vdyn, 0, 16);
                enc.setValue(AffineProjA{ .k = h, .full = sl.is_full, .m = lay.proj_ms }, 17);
                enc.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            } else {
                enc.setPipeline(p.batch_proj);
                enc.setBuffer(in0, 0, 0);
                enc.setBuffer(in1, 0, 1);
                enc.setBuffer(in2, 0, 2);
                rt.setW(enc.*, @intCast(lay.proj_q.w.off), 3);
                rt.setW(enc.*, @intCast(lay.proj_q.s.off), 4);
                rt.setW(enc.*, @intCast(lay.proj_q.b.off), 5);
                rt.setW(enc.*, @intCast(lay.proj_k.w.off), 6);
                rt.setW(enc.*, @intCast(lay.proj_k.s.off), 7);
                rt.setW(enc.*, @intCast(lay.proj_k.b.off), 8);
                rt.setW(enc.*, @intCast(lay.proj_v.w.off), 9);
                rt.setW(enc.*, @intCast(lay.proj_v.s.off), 10);
                rt.setW(enc.*, @intCast(lay.proj_v.b.off), 11);
                enc.setBuffer(rt.q, 0, 12);
                enc.setBuffer(rt.k, 0, 13);
                enc.setBuffer(rt.v, 0, 14);
                enc.setBuffer(rt.r, 0, 15);
                enc.setBuffer(rt.gate, 0, 16);
                enc.setBuffer(rt.sink, 0, 17);
                enc.setBuffer(rt.kdyn, 0, 18);
                enc.setBuffer(rt.vdyn, 0, 19);
                enc.setValue(AffineProjA{ .k = h, .full = sl.is_full, .m = lay.proj_ms }, 20);
                enc.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
            dispatches += 1;

            const qkv_norm: f32 = 1.0 / @sqrt(@as(f32, if (full) 9.0 else 5.0));
            const kv_stride_rows: u32 = if (full) @intCast(rt.capacity) else g.window;
            const kv_row: u32 = if (full) pos else (pos % g.window);
            if (opt_d9) {
                const qkv_row = BatchQkvRow{
                    .k_ring = convAddr(rt.k_conv, sl.slot, kvw),
                    .v_ring = convAddr(rt.v_conv, sl.slot, kvw),
                    .keys = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot],
                    .values = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot],
                    .pos = pos,
                };
                const qkv_args = QkvPermA{
                    .kv = g.kv_heads, .heads = g.heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent,
                    .layout = sl.is_full, .dyn = sl.is_full, .tau = sl.is_full, .kv_stride_rows = kv_stride_rows, .kv_row = kv_row,
                    .norm = qkv_norm, .q_threshold = g.qk_clip, .rel_threshold = g.rel_clip, .eps = g.eps,
                    .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha,
                };
                enc.setPipeline(p.aot_qkv_perm.?);
                enc.setBuffer(rt.k, 0, 0);
                enc.setBuffer(rt.v, 0, 1);
                rt.setW(enc.*, @intCast(lay.k_conv.off), 2);
                rt.setW(enc.*, @intCast(lay.v_conv.off), 3);
                enc.setBuffer(rt.kdyn, 0, 4);
                enc.setBuffer(rt.vdyn, 0, 5);
                enc.setBuffer(rt.q, 0, 6);
                rt.setW(enc.*, @intCast(lay.q_scale.off), 7);
                enc.setBuffer(rt.r, 0, 8);
                rt.setW(enc.*, @intCast(lay.r_scale.off), 9);
                rt.setW(enc.*, @intCast(lay.rel_bias.off), 10);
                enc.setValue(qkv_args, 11);
                enc.setBuffer(rt.relbias, 0, 12);
                enc.setValue(qkv_row, 13);
                enc.dispatchGroups(mtl.Size.of(g.kv_heads + g.heads + (g.heads * (lay.extent + 1) + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
            } else {
                const kv_at = rt.slot_kv_off[sl.slot] + @as(usize, kv_row) * kvw * 2;
                const qkv_row = BatchQkvRow{
                    .k_ring = convAddr(rt.k_conv, sl.slot, kvw),
                    .v_ring = convAddr(rt.v_conv, sl.slot, kvw),
                    .keys = rt.keys.gpuAddress() + kv_at,
                    .values = rt.values.gpuAddress() + kv_at,
                    .pos = pos,
                };
                const qkv_args = QkvArgs{
                    .kv = g.kv_heads, .heads = g.heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent,
                    .layout = sl.is_full, .dyn = sl.is_full, .tau = sl.is_full, .norm = qkv_norm,
                    .q_threshold = g.qk_clip, .rel_threshold = g.rel_clip, .eps = g.eps,
                    .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha,
                };
                enc.setPipeline(p.qkv);
                enc.setBuffer(rt.k, 0, 0);
                enc.setBuffer(rt.v, 0, 1);
                rt.setW(enc.*, @intCast(lay.k_conv.off), 2);
                rt.setW(enc.*, @intCast(lay.v_conv.off), 3);
                enc.setBuffer(rt.kdyn, 0, 4);
                enc.setBuffer(rt.vdyn, 0, 5);
                enc.setBuffer(rt.q, 0, 6);
                rt.setW(enc.*, @intCast(lay.q_scale.off), 7);
                enc.setBuffer(rt.r, 0, 8);
                rt.setW(enc.*, @intCast(lay.r_scale.off), 9);
                rt.setW(enc.*, @intCast(lay.rel_bias.off), 10);
                enc.setValue(qkv_args, 11);
                enc.setBuffer(rt.relbias, 0, 12);
                enc.setValue(qkv_row, 13);
                enc.setBuffer(rt.kc, 0, 14);
                enc.setBuffer(rt.vc, 0, 15);
                enc.dispatchGroups(mtl.Size.of(g.kv_heads + g.heads + (g.heads * (lay.extent + 1) + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
            }
            dispatches += 1;

            const win: u32 = if (full) 0 else g.window;
            const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
            const part = attnPartition(pos + 1 - lo);
            const attn_row = BatchRow{
                .keys = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot],
                .values = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot],
                .args = .{
                    .heads = g.heads, .kv = g.kv_heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent,
                    .window = win, .pos = pos, .tau = sl.is_full, .splits = part.splits, .chunk = part.chunk,
                    .scale = scale_attn, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha,
                },
            };
            if (opt_d9) {
                const perm_row = SplitPermRow{ .keys = attn_row.keys, .values = attn_row.values, .args = attn_row.args, .kv_stride_rows = kv_stride_rows };
                enc.setPipeline(p.aot_split_perm.?);
                enc.setBuffer(rt.q, 0, 0);
                enc.setBuffer(rt.relbias, 0, 1);
                enc.setBuffer(rt.sink, 0, 2);
                enc.setValue(perm_row, 3);
                enc.setBuffer(rt.parts, 0, 4);
                enc.dispatchGroups(mtl.Size.of(g.kv_heads * part.splits, 1, 1), mtl.Size.of((g.heads / g.kv_heads) * 32, 1, 1));
            } else {
                enc.setPipeline(p.split);
                enc.setBuffer(rt.q, 0, 0);
                enc.setBuffer(rt.relbias, 0, 1);
                enc.setBuffer(rt.sink, 0, 2);
                enc.setValue(attn_row, 3);
                enc.setBuffer(rt.parts, 0, 4);
                enc.dispatchGroups(mtl.Size.of(g.kv_heads * part.splits, 1, 1), mtl.Size.of((g.heads / g.kv_heads) * 32, 1, 1));
            }
            dispatches += 1;

            if (opt_d1) {
                enc.setPipeline(p.aot_combine_sig_mul.?);
                enc.setBuffer(rt.parts, 0, 0);
                enc.setValue(attn_row, 1);
                enc.setBuffer(rt.gate, 0, 2);
                enc.setBuffer(rt.attn, 0, 3);
                enc.dispatchGroups(mtl.Size.of(g.heads, 1, 1), mtl.Size.of(32, 1, 1));
                dispatches += 1;
            } else {
                enc.setPipeline(p.combine);
                enc.setBuffer(rt.parts, 0, 0);
                enc.setValue(attn_row, 1);
                enc.setBuffer(rt.attn, 0, 2);
                enc.dispatchGroups(mtl.Size.of(g.heads, 1, 1), mtl.Size.of(32, 1, 1));
                dispatches += 1;

                enc.setPipeline(p.sig_mul);
                enc.setBuffer(rt.attn, 0, 0);
                enc.setBuffer(rt.gate, 0, 1);
                enc.setValue(PairA{ .count = qw }, 2);
                enc.setBuffer(rt.attn, 0, 3);
                enc.dispatchThreads(mtl.Size.of(qw, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            }

            if (use_sb) {
                enc.setPipeline(p.aot_dense8_sb.?);
                enc.setBuffer(rt.attn, 0, 0);
                rt.setW(enc.*, @intCast(lay.proj_o.w.off), 1);
                rt.setW(enc.*, @intCast(lay.proj_o.sb.off), 2);
                enc.setValue(AffineDenseA{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 3);
                enc.setBuffer(rt.projout, 0, 4);
                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            } else {
                enc.setPipeline(p.dense);
                enc.setBuffer(rt.attn, 0, 0);
                rt.setW(enc.*, @intCast(lay.proj_o.w.off), 1);
                rt.setW(enc.*, @intCast(lay.proj_o.s.off), 2);
                rt.setW(enc.*, @intCast(lay.proj_o.b.off), 3);
                enc.setValue(AffineDenseA{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 4);
                enc.setBuffer(rt.projout, 0, 5);
                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
            dispatches += 1;

            const aconv_row = ConvRow{ .ring = convAddr(rt.attn_conv, sl.slot, h), .pos = pos };
            const mlp_q = rt.sched.globals.mlp_queries[sl.phys];
            const mq_off: usize = @intCast(mlp_q.off);
            enc.setPipeline(p.batch_conv);
            enc.setBuffer(rt.projout, 0, 0);
            rt.setW(enc.*, @intCast(lay.attn_post.off), 1);
            const aconv_args = DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = @intFromBool(!push), .topk = 0 };
            enc.setValue(aconv_args, 2);
            enc.setValue(aconv_args, 3);
            enc.setBuffer(if (!push) rt.x else rt.o, 0, 4);
            enc.setValue(aconv_row, 5);
            enc.setBuffer(rt.projout, 0, 6);
            enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
            dispatches += 1;

            if (push) {
                enc.setPipeline(p.scatter);
                enc.setBuffer(rt.x, 0, 0);
                enc.setValue(ScatterA{ .width = h, .stride = mix_states, .slot = sl.block_count_before }, 1);
                enc.setBuffer(rt.blocks, 0, 2);
                enc.setBuffer(rt.o, 0, 3);
                enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            }

            if (bank and opt_d3) {
                enc.setPipeline(p.aot_mix_inv.?);
                enc.setBuffer(rt.blocks, 0, 0);
                enc.setBuffer(rt.x, 0, 1);
                rt.setW(enc.*, mq_off, 2);
                enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = 0, .scale = scale_h, .eps = g.eps }, 3);
                enc.setBuffer(rt.mixout, 0, 4);
                enc.setBuffer(rt.ctl_inv, 0, 5);
                enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;

                const ctl = lay.controllers[3];
                enc.setPipeline(p.aot_ctl_down1.?);
                enc.setBuffer(rt.mixout, 0, 0);
                enc.setBuffer(rt.msg, 0, 1);
                enc.setBuffer(rt.ctl_inv, 0, 2);
                rt.setW(enc.*, @intCast(ctl.down.off), 3);
                enc.setValue(AffineDenseA{ .n = g.gate_rank, .k = 2 * h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 4);
                enc.setBuffer(rt.normed, 0, 5);
                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 1) / 2, 1, 1), mtl.Size.of(64, 1, 1));
                dispatches += 1;

                if (use_packed_w13_gf) {
                    enc.setPipeline(p.aot_ctl_apply1_gf2.?);
                    enc.setBuffer(rt.normed, 0, 0);
                    rt.setW(enc.*, @intCast(ctl.gain.off), 1);
                    enc.setBuffer(rt.mixout, 0, 2);
                    enc.setBuffer(rt.msg, 0, 3);
                    enc.setBuffer(rt.mixout, 0, 4);
                    enc.setValue(AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 5);
                    enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                } else {
                    enc.setPipeline(p.batch_float_ctl);
                    enc.setBuffer(rt.normed, 0, 0);
                    rt.setW(enc.*, @intCast(ctl.gain.off), 1);
                    rt.setW(enc.*, @intCast(ctl.forget.off), 2);
                    enc.setBuffer(rt.mixout, 0, 3);
                    enc.setBuffer(rt.msg, 0, 4);
                    enc.setBuffer(rt.mixout, 0, 5);
                    enc.setValue(AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 6);
                    enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                }
                dispatches += 1;

                enc.setPipeline(p.rms);
                enc.setBuffer(rt.mixout, 0, 0);
                rt.setW(enc.*, @intCast(lay.mlp_norm.off), 1);
                enc.setValue(RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
                enc.setBuffer(rt.m0, 0, 3);
                enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            } else {
                enc.setPipeline(p.batch_mix);
                enc.setBuffer(rt.blocks, 0, 0);
                enc.setBuffer(rt.x, 0, 1);
                rt.setW(enc.*, mq_off, 2);
                rt.setW(enc.*, mq_off, 3);
                rt.setW(enc.*, mq_off, 4);
                rt.setW(enc.*, mq_off, 5);
                enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = @intFromBool(!bank), .scale = scale_h, .eps = g.eps }, 6);
                enc.setBuffer(rt.mixout, 0, 7);
                rt.setW(enc.*, @intCast(if (!bank) lay.mlp_norm.off else mq_off), 8);
                rt.setW(enc.*, mq_off, 9);
                rt.setW(enc.*, mq_off, 10);
                enc.setBuffer(if (!bank) rt.m0 else rt.n0, 0, 11);
                enc.setBuffer(if (!bank) rt.m0 else rt.n1, 0, 12);
                enc.setBuffer(if (!bank) rt.m0 else rt.n2, 0, 13);
                enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;

                if (bank) {
                    const ctl = lay.controllers[3];
                    enc.setPipeline(p.rms_pair);
                    enc.setBuffer(rt.mixout, 0, 0);
                    enc.setBuffer(rt.msg, 0, 1);
                    enc.setValue(RmsPairA{ .width = h, .xs0 = 1, .xb0 = 0, .xs1 = 4, .xb1 = 3, .eps = g.eps }, 2);
                    enc.setBuffer(rt.h2in, 0, 3);
                    enc.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;

                    enc.setPipeline(p.float_mv);
                    enc.setBuffer(rt.h2in, 0, 0);
                    rt.setW(enc.*, @intCast(ctl.down.off), 1);
                    enc.setValue(AffineDenseA{ .n = g.gate_rank, .k = 2 * h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
                    enc.setBuffer(rt.normed, 0, 3);
                    enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;

                    if (use_packed_w13_gf) {
                        enc.setPipeline(p.aot_ctl_apply1_gf2.?);
                        enc.setBuffer(rt.normed, 0, 0);
                        rt.setW(enc.*, @intCast(ctl.gain.off), 1);
                        enc.setBuffer(rt.mixout, 0, 2);
                        enc.setBuffer(rt.msg, 0, 3);
                        enc.setBuffer(rt.mixout, 0, 4);
                        enc.setValue(AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 5);
                        enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    } else {
                        enc.setPipeline(p.batch_float_ctl);
                        enc.setBuffer(rt.normed, 0, 0);
                        rt.setW(enc.*, @intCast(ctl.gain.off), 1);
                        rt.setW(enc.*, @intCast(ctl.forget.off), 2);
                        enc.setBuffer(rt.mixout, 0, 3);
                        enc.setBuffer(rt.msg, 0, 4);
                        enc.setBuffer(rt.mixout, 0, 5);
                        enc.setValue(AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 6);
                        enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    }
                    dispatches += 1;

                    enc.setPipeline(p.rms);
                    enc.setBuffer(rt.mixout, 0, 0);
                    rt.setW(enc.*, @intCast(lay.mlp_norm.off), 1);
                    enc.setValue(RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
                    enc.setBuffer(rt.m0, 0, 3);
                    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;
                }
            }

            enc.setPipeline(p.float_mv);
            enc.setBuffer(rt.m0, 0, 0);
            rt.setW(enc.*, @intCast(lay.router.off), 1);
            enc.setValue(AffineDenseA{ .n = g.experts, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
            enc.setBuffer(rt.router, 0, 3);
            enc.dispatchGroups(mtl.Size.of((g.experts + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            dispatches += 1;

            enc.setPipeline(p.routepick);
            enc.setBuffer(rt.router, 0, 0);
            rt.setW(enc.*, @intCast(lay.router_bias.off), 1);
            enc.setValue(PickA{ .experts = g.experts, .topk = g.topk, .rows = 1, .route_scale = g.route_scale }, 2);
            enc.setBuffer(rt.picks, 0, 3);
            enc.setBuffer(rt.wts, 0, 4);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
            dispatches += 1;

            if (lay.is_affine4 != 0) {
                enc.setPipeline(p.aff_up);
                enc.setBuffer(rt.m0, 0, 0);
                rt.setW(enc.*, @intCast(lay.w13.off), 1);
                rt.setW(enc.*, @intCast(lay.w13_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w13_bias.off), 3);
                enc.setBuffer(rt.picks, 0, 4);
                enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
                enc.setBuffer(rt.act, 0, 6);
                enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;

                enc.setPipeline(p.aff_down);
                enc.setBuffer(rt.act, 0, 0);
                rt.setW(enc.*, @intCast(lay.w2.off), 1);
                rt.setW(enc.*, @intCast(lay.w2_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w2_bias.off), 3);
                enc.setBuffer(rt.picks, 0, 4);
                enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
                enc.setBuffer(rt.down, 0, 6);
                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            } else if (opt_d2) {
                enc.setPipeline(if (use_packed_w13_gf) p.aot_gu_nv_packed.? else p.aot_gu_nv.?);
                enc.setBuffer(rt.m0, 0, 0);
                rt.setW(enc.*, @intCast(lay.w13.off), 1);
                rt.setW(enc.*, @intCast(lay.w13_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w13_global.off), 3);
                enc.setBuffer(rt.picks, 0, 4);
                enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
                enc.setBuffer(rt.act, 0, 6);
                rt.setW(enc.*, @intCast(rt.sched.globals.t16.off), 7);
                rt.setW(enc.*, @intCast(rt.sched.globals.t256.off), 8);
                enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;

                enc.setPipeline(p.aot_down1_nv.?);
                enc.setBuffer(rt.act, 0, 0);
                rt.setW(enc.*, @intCast(lay.w2.off), 1);
                rt.setW(enc.*, @intCast(lay.w2_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w2_global.off), 3);
                enc.setBuffer(rt.picks, 0, 4);
                enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
                enc.setBuffer(rt.down, 0, 6);
                rt.setW(enc.*, @intCast(rt.sched.globals.t16.off), 7);
                rt.setW(enc.*, @intCast(rt.sched.globals.t256.off), 8);
                enc.dispatchGroups(mtl.Size.of((h + 31) / 32, g.topk, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            } else {
                enc.setPipeline(p.gu_nv);
                enc.setBuffer(rt.m0, 0, 0);
                rt.setW(enc.*, @intCast(lay.w13.off), 1);
                rt.setW(enc.*, @intCast(lay.w13_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w13_global.off), 3);
                rt.setW(enc.*, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                enc.setBuffer(rt.picks, 0, 5);
                enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 6);
                enc.setBuffer(rt.act, 0, 7);
                rt.setW(enc.*, @intCast(rt.sched.globals.t16.off), 8);
                enc.dispatchGroups(mtl.Size.of(g.expert_inter, g.topk, 1), mtl.Size.of(32, 1, 1));
                dispatches += 1;

                enc.setPipeline(p.down1_nv);
                enc.setBuffer(rt.act, 0, 0);
                rt.setW(enc.*, @intCast(lay.w2.off), 1);
                rt.setW(enc.*, @intCast(lay.w2_scale.off), 2);
                rt.setW(enc.*, @intCast(lay.w2_global.off), 3);
                rt.setW(enc.*, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                enc.setBuffer(rt.picks, 0, 5);
                enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 6);
                enc.setBuffer(rt.down, 0, 7);
                rt.setW(enc.*, @intCast(rt.sched.globals.t16.off), 8);
                enc.dispatchGroups(mtl.Size.of((h + 31) / 32, g.topk, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;
            }

            const mconv_row = ConvRow{ .ring = convAddr(rt.mlp_conv, sl.slot, h), .pos = pos };
            const mconv_args = DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = 1, .topk = g.topk };
            enc.setPipeline(p.batch_conv);
            enc.setBuffer(rt.down, 0, 0);
            rt.setW(enc.*, @intCast(lay.mlp_post.off), 1);
            enc.setValue(mconv_args, 2);
            enc.setValue(mconv_args, 3);
            enc.setBuffer(rt.x, 0, 4);
            enc.setValue(mconv_row, 5);
            enc.setBuffer(rt.wts, 0, 6);
            enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
            dispatches += 1;

            if (sl.is_prelude_end != 0) {
                enc.setPipeline(p.copy);
                enc.setBuffer(rt.x, 0, 0);
                enc.setBuffer(rt.prelude, 0, 1);
                enc.setValue(PairA{ .count = h }, 2);
                enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                dispatches += 1;

                if (ring_slot == null) {
                    enc.end();
                    cb0.commit();
                    cb_first.* = cb0.*;
                    cb0.* = rt.queue.commandBuffer();
                    enc.* = cb0.compute(.serial);
                }
            }
        }

        if (want_logits) {
            const oq_off: usize = @intCast(rt.sched.globals.output_query.off);
            enc.setPipeline(p.batch_mix);
            enc.setBuffer(rt.blocks, 0, 0);
            enc.setBuffer(rt.x, 0, 1);
            rt.setW(enc.*, oq_off, 2);
            rt.setW(enc.*, oq_off, 3);
            rt.setW(enc.*, oq_off, 4);
            rt.setW(enc.*, oq_off, 5);
            enc.setValue(MixArgs{ .width = h, .depth = 5, .stride = mix_states, .k = 1, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
            enc.setBuffer(rt.mixout, 0, 7);
            rt.setW(enc.*, @intCast(rt.sched.globals.norm.off), 8);
            rt.setW(enc.*, oq_off, 9);
            rt.setW(enc.*, oq_off, 10);
            enc.setBuffer(rt.n0, 0, 11);
            enc.setBuffer(rt.n0, 0, 12);
            enc.setBuffer(rt.n0, 0, 13);
            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            dispatches += 1;

            if (use_sb) {
                enc.setPipeline(p.aot_dense8_sb.?);
                enc.setBuffer(rt.n0, 0, 0);
                rt.setW(enc.*, @intCast(rt.sched.globals.unembed.w.off), 1);
                rt.setW(enc.*, @intCast(rt.sched.globals.unembed.sb.off), 2);
                enc.setValue(AffineDenseA{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 3);
                enc.setBuffer(rt.logits, 0, 4);
                enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            } else {
                enc.setPipeline(p.dense);
                enc.setBuffer(rt.n0, 0, 0);
                rt.setW(enc.*, @intCast(rt.sched.globals.unembed.w.off), 1);
                rt.setW(enc.*, @intCast(rt.sched.globals.unembed.s.off), 2);
                rt.setW(enc.*, @intCast(rt.sched.globals.unembed.b.off), 3);
                enc.setValue(AffineDenseA{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 4);
                enc.setBuffer(rt.logits, 0, 5);
                enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            }
            dispatches += 1;

            if (gpu_argmax or ring_slot != null) {
                if (p.aot_argmax_p1 != null and p.aot_argmax_p2 != null) {
                    const am_args = Argmax2A{
                        .vocab = g.unpadded_vocab,
                        .step_idx = ring_slot orelse 0,
                        .stop_token = stop_token,
                        .write_loop = @intFromBool(ring_slot != null),
                    };
                    enc.setPipeline(p.aot_argmax_p1.?);
                    enc.setBuffer(rt.logits, 0, 0);
                    enc.setValue(am_args, 1);
                    enc.setBuffer(rt.argmax_tmp_v, 0, 2);
                    enc.setBuffer(rt.argmax_tmp_i, 0, 3);
                    enc.dispatchGroups(mtl.Size.of(128, 1, 1), mtl.Size.of(256, 1, 1));
                    dispatches += 1;

                    enc.setPipeline(p.aot_argmax_p2.?);
                    enc.setBuffer(rt.argmax_tmp_v, 0, 0);
                    enc.setBuffer(rt.argmax_tmp_i, 0, 1);
                    enc.setValue(am_args, 2);
                    enc.setBuffer(rt.ids, 0, 3);
                    enc.setBuffer(if (ring_slot != null) rt.token_ring else rt.argmax_out, 0, 4);
                    enc.setBuffer(rt.loop_ctrl, 0, 5);
                    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(128, 1, 1));
                    dispatches += 1;
                } else {
                    enc.setPipeline(p.argmax);
                    enc.setBuffer(rt.logits, 0, 0);
                    enc.setValue(ArgmaxA{ .vocab = g.unpadded_vocab }, 1);
                    enc.setBuffer(rt.argmax_out, 0, 2);
                    enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
                    dispatches += 1;
                }
            }
        }
        return dispatches;
    }

    pub fn stepDirect(rt: *Runtime, token_id: u32, want_logits: bool, gpu_argmax: bool) !StepStats {
        if (rt.len >= rt.capacity) return error.ContextFull;
        const t0 = nowNs();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();

        rt.ids.slice(u32, 1)[0] = token_id;
        var cb0 = rt.queue.commandBuffer();
        var enc = cb0.compute(.serial);
        var cb_first: ?mtl.CommandBuffer = null;
        const dispatches = rt.encodeStepBody(&cb0, &enc, &cb_first, @intCast(rt.len), want_logits, gpu_argmax, null, 0xffffffff);

        enc.end();
        cb0.commit();
        cb0.wait();
        if (cb_first) |c0| {
            if (c0.failure()) |msg| {
                std.log.err("aot cb0 failed: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        if (cb0.failure()) |msg| {
            std.log.err("aot cb1 failed: {s}", .{msg});
            return error.GpuFailed;
        }
        const gpu_ms = (if (cb_first) |c0| c0.gpuSeconds() else 0) * 1000.0 + cb0.gpuSeconds() * 1000.0;
        const wall_ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        rt.len += 1;
        const tok: u32 = if (!want_logits) 0 else if (gpu_argmax) rt.argmax_out.slice(u32, 1)[0] else cpuArgmax(rt.logits.slice(f32, rt.sched.geom.unpadded_vocab));
        return .{ .wall_ms = wall_ms, .gpu_ms = gpu_ms, .dispatches = dispatches, .command_buffers = if (cb_first != null) 2 else 1, .syncs = 1, .token = tok };
    }

    pub const BurstStats = struct {
        produced: usize = 0,
        stopped: bool = false,
        cancelled: bool = false,
        wall_ms: f64 = 0,
        gpu_ms: f64 = 0,
        dispatches: u32 = 0,
        syncs: u32 = 1,
    };

    pub fn requestCancel(rt: *Runtime) void {
        const ctrl: *volatile LoopCtrl = @ptrCast(@alignCast(rt.loop_ctrl.contents()));
        ctrl.cancelled = 1;
    }

    pub fn generateBurst(rt: *Runtime, start_token: u32, count: usize, stop_token: u32, out_tokens: []u32) !BurstStats {
        if (count == 0 or count > 64 or count > out_tokens.len) return error.InvalidBurst;
        if (rt.len + count > rt.capacity) return error.ContextFull;
        const t0 = nowNs();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();

        rt.ids.slice(u32, 1)[0] = start_token;
        const ctrl: *LoopCtrl = @ptrCast(@alignCast(rt.loop_ctrl.contents()));
        ctrl.* = .{};

        var cbs: [64]mtl.CommandBuffer = undefined;
        var dispatches: u32 = 0;
        for (0..count) |s| {
            var cb = rt.queue.commandBuffer();
            var enc = cb.compute(.serial);
            var unused_first: ?mtl.CommandBuffer = null;
            dispatches = rt.encodeStepBody(&cb, &enc, &unused_first, @intCast(rt.len + s), true, true, @intCast(s), stop_token);
            enc.end();
            cb.commit();
            cbs[s] = cb;
        }
        cbs[count - 1].wait();
        var gpu_ms: f64 = 0;
        for (0..count) |s| {
            if (cbs[s].failure()) |msg| {
                std.log.err("aot burst cb[{d}] failed: {s}", .{ s, msg });
                return error.GpuFailed;
            }
            gpu_ms += cbs[s].gpuSeconds() * 1000.0;
        }
        const wall_ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        const produced: usize = @min(@as(usize, ctrl.produced), count);
        const ring = rt.token_ring.slice(u32, count);
        @memcpy(out_tokens[0..produced], ring[0..produced]);
        rt.len += produced;
        const was_cancelled = ctrl.cancelled != 0;
        return .{
            .produced = produced,
            .stopped = ctrl.stopped != 0 and !was_cancelled,
            .cancelled = was_cancelled,
            .wall_ms = wall_ms,
            .gpu_ms = gpu_ms,
            .dispatches = dispatches,
            .syncs = 1,
        };
    }

    pub fn allocParam(arena_bytes: []u8, cursor: *usize, val: anytype) !usize {
        const off = (cursor.* + 15) & ~@as(usize, 15);
        const bytes = std.mem.asBytes(&val);
        if (off > arena_bytes.len or bytes.len > arena_bytes.len - off) return error.ParamArenaFull;
        @memcpy(arena_bytes[off .. off + bytes.len], bytes);
        cursor.* = off + bytes.len;
        return off;
    }

    pub fn buildIcb(rt: *Runtime, gpu_argmax: bool) !void {
        if (rt.icb_pre != null and rt.icb_built_argmax == gpu_argmax) return;
        if (rt.icb_pre) |icb| icb.deinit();
        if (rt.icb_main) |icb| icb.deinit();

        const icb_pre = try mtl.IndirectCommandBuffer.init(rt.device, 128, 24);
        errdefer icb_pre.deinit();
        const icb_main = try mtl.IndirectCommandBuffer.init(rt.device, 896, 24);
        errdefer icb_main.deinit();

        const arena = rt.param_arena.contents()[0..rt.param_arena.length()];
        var cur: usize = 0;
        const param_arena = rt.param_arena;
        const p = &rt.pipes;
        const g = rt.sched.geom;
        const h: u32 = g.hidden;
        const qw: u32 = g.heads * g.head_dim;
        const kvw: u32 = g.kv_heads * g.head_dim;
        const scale_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const norm_post: f32 = 1.0 / @sqrt(@as(f32, 5.0));

        var active_icb = icb_pre;
        var idx: usize = 0;

        {
            const cmd = active_icb.command(idx);
            cmd.setPipeline(p.embed);
            cmd.setBuffer(rt.ids, 0, 0);
            rt.setW(cmd, @intCast(rt.sched.globals.embed.w.off), 1);
            rt.setW(cmd, @intCast(rt.sched.globals.embed.s.off), 2);
            rt.setW(cmd, @intCast(rt.sched.globals.embed.b.off), 3);
            rt.setW(cmd, @intCast(rt.sched.globals.embed_norm.off), 4);
            cmd.setBuffer(param_arena, try allocParam(arena, &cur, EmbedA{ .width = h, .rows = 1, .eps = g.eps }), 5);
            cmd.setBuffer(rt.x, 0, 6);
            cmd.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            idx += 1;
        }

        for (rt.sched.slots, 0..) |sl, visit_i| {
            const lay = &rt.sched.layers[sl.phys];
            const full = sl.is_full != 0;
            const bank = sl.is_bank != 0;
            const push = sl.push_block != 0;

            if (sl.is_pass_start != 0) {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.pass_copy);
                cmd.setBuffer(rt.blocks, 0, 0);
                cmd.setBuffer(rt.x, 0, 1);
                cmd.setBuffer(rt.prelude, 0, 2);
                cmd.setBuffer(rt.bank, 0, 3);
                cmd.setBuffer(rt.passout, 0, 4);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, PassCopyA{ .width = h, .stride = mix_states, .start = 2, .count = sl.bank_n }), 5);
                cmd.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            var in0 = rt.x;
            var in1 = rt.x;
            var in2 = rt.x;
            var fused_norm = false;

            if (sl.phys > 0) {
                const qref = rt.sched.globals.attn_queries[sl.phys - 1];
                const kq: u32 = if (full) 3 else 1;
                if (bank) {
                    const at: usize = sl.cross_pass_at;
                    const cpq = rt.sched.globals.cross_pass_q.off;
                    const cpkv = rt.sched.globals.cross_pass_kv.off;
                    const q0_off: usize = @intCast(qref.off);
                    const q1_off: usize = if (full) q0_off + h * 4 else q0_off;
                    const q2_off: usize = if (full) q0_off + 2 * h * 4 else q0_off;
                    const args2 = [2]MixArgs{
                        .{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 0, .scale = scale_h, .eps = g.eps },
                        .{ .width = h, .depth = sl.bank_n + 1, .stride = mix_states, .k = 4, .norm = 0, .scale = scale_h, .eps = g.eps },
                    };
                    const c0 = active_icb.command(idx);
                    c0.setBarrier();
                    c0.setPipeline(p.batch_mix2);
                    c0.setBuffer(rt.blocks, 0, 0);
                    c0.setBuffer(rt.x, 0, 1);
                    rt.setW(c0, q0_off, 2);
                    rt.setW(c0, q1_off, 3);
                    rt.setW(c0, q2_off, 4);
                    rt.setW(c0, q0_off, 5);
                    c0.setBuffer(rt.mixout, 0, 6);
                    c0.setBuffer(rt.bank, 0, 7);
                    c0.setBuffer(rt.passout, 0, 8);
                    rt.setW(c0, @intCast(cpq + (at * 2) * h * 4), 9);
                    rt.setW(c0, @intCast(cpkv + (at * 2) * h * 4), 10);
                    rt.setW(c0, @intCast(cpkv + (at * 2 + 1) * h * 4), 11);
                    rt.setW(c0, @intCast(cpq + (at * 2 + 1) * h * 4), 12);
                    c0.setBuffer(rt.msg, 0, 13);
                    c0.setBuffer(param_arena, try allocParam(arena, &cur, args2), 14);
                    c0.dispatchGroups(mtl.Size.of(1, 2, 1), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    const ctl_off = try allocParam(arena, &cur, ControlsArgs{
                        .h = h,
                        .rank = g.gate_rank,
                        .rows = 1,
                        .xs = .{ kq, kq, kq },
                        .xb = if (full) .{ 0, 1, 2 } else .{ 0, 0, 0 },
                    });

                    const c1 = active_icb.command(idx);
                    c1.setBarrier();
                    c1.setPipeline(p.ctl_norm);
                    c1.setBuffer(rt.mixout, 0, 0);
                    c1.setBuffer(rt.mixout, 0, 1);
                    c1.setBuffer(rt.mixout, 0, 2);
                    c1.setBuffer(rt.msg, 0, 3);
                    c1.setBuffer(param_arena, ctl_off, 4);
                    c1.setBuffer(rt.ctl_in, 0, 5);
                    c1.dispatchGroups(mtl.Size.of(6, 1, 1), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    const c2 = active_icb.command(idx);
                    c2.setBarrier();
                    c2.setPipeline(p.ctl_down);
                    c2.setBuffer(rt.ctl_in, 0, 0);
                    rt.setW(c2, @intCast(lay.controllers[0].down.off), 1);
                    rt.setW(c2, @intCast(lay.controllers[1].down.off), 2);
                    rt.setW(c2, @intCast(lay.controllers[2].down.off), 3);
                    c2.setBuffer(param_arena, ctl_off, 4);
                    c2.setBuffer(rt.ctl_hid, 0, 5);
                    c2.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    const c3 = active_icb.command(idx);
                    c3.setBarrier();
                    c3.setPipeline(p.ctl_apply);
                    c3.setBuffer(rt.ctl_hid, 0, 0);
                    rt.setW(c3, @intCast(lay.controllers[0].gain.off), 1);
                    rt.setW(c3, @intCast(lay.controllers[1].gain.off), 2);
                    rt.setW(c3, @intCast(lay.controllers[2].gain.off), 3);
                    rt.setW(c3, @intCast(lay.controllers[0].forget.off), 4);
                    rt.setW(c3, @intCast(lay.controllers[1].forget.off), 5);
                    rt.setW(c3, @intCast(lay.controllers[2].forget.off), 6);
                    c3.setBuffer(rt.mixout, 0, 7);
                    c3.setBuffer(rt.mixout, 0, 8);
                    c3.setBuffer(rt.mixout, 0, 9);
                    c3.setBuffer(rt.msg, 0, 10);
                    c3.setBuffer(rt.n0, 0, 11);
                    c3.setBuffer(rt.n1, 0, 12);
                    c3.setBuffer(rt.n2, 0, 13);
                    c3.setBuffer(param_arena, ctl_off, 14);
                    c3.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    const c4 = active_icb.command(idx);
                    c4.setBarrier();
                    c4.setPipeline(p.rms3);
                    c4.setBuffer(rt.n0, 0, 0);
                    c4.setBuffer(rt.n1, 0, 1);
                    c4.setBuffer(rt.n2, 0, 2);
                    rt.setW(c4, @intCast(lay.q_norm.off), 3);
                    rt.setW(c4, @intCast(lay.k_norm.off), 4);
                    rt.setW(c4, @intCast(lay.v_norm.off), 5);
                    c4.setBuffer(param_arena, try allocParam(arena, &cur, Rms3A{ .width = h, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 }, .eps = g.eps }), 6);
                    c4.setBuffer(rt.n0, 0, 7);
                    c4.setBuffer(rt.n1, 0, 8);
                    c4.setBuffer(rt.n2, 0, 9);
                    c4.dispatchGroups(mtl.Size.of(3, 1, 1), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    in0 = rt.n0;
                    in1 = rt.n1;
                    in2 = rt.n2;
                    fused_norm = true;
                } else {
                    const q0_off: usize = @intCast(qref.off);
                    const q1_off: usize = if (full) q0_off + h * 4 else q0_off;
                    const q2_off: usize = if (full) q0_off + 2 * h * 4 else q0_off;
                    const cmd = active_icb.command(idx);
                    cmd.setBarrier();
                    cmd.setPipeline(p.batch_mix);
                    cmd.setBuffer(rt.blocks, 0, 0);
                    cmd.setBuffer(rt.x, 0, 1);
                    rt.setW(cmd, q0_off, 2);
                    rt.setW(cmd, q1_off, 3);
                    rt.setW(cmd, q2_off, 4);
                    rt.setW(cmd, q0_off, 5);
                    cmd.setBuffer(param_arena, try allocParam(arena, &cur, MixArgs{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = kq, .norm = 1, .scale = scale_h, .eps = g.eps }), 6);
                    cmd.setBuffer(rt.mixout, 0, 7);
                    rt.setW(cmd, @intCast(lay.q_norm.off), 8);
                    rt.setW(cmd, @intCast(lay.k_norm.off), 9);
                    rt.setW(cmd, @intCast(lay.v_norm.off), 10);
                    cmd.setBuffer(rt.n0, 0, 11);
                    cmd.setBuffer(rt.n1, 0, 12);
                    cmd.setBuffer(rt.n2, 0, 13);
                    cmd.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                    idx += 1;

                    in0 = rt.n0;
                    in1 = if (full) rt.n1 else rt.n0;
                    in2 = if (full) rt.n2 else rt.n0;
                    fused_norm = true;
                }
            }

            if (!fused_norm) {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.rms);
                cmd.setBuffer(rt.x, 0, 0);
                rt.setW(cmd, @intCast(lay.q_norm.off), 1);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }), 2);
                cmd.setBuffer(rt.n0, 0, 3);
                cmd.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
                in0 = rt.n0;
                in1 = rt.n0;
                in2 = rt.n0;
            }

            {
                var total_proj: u32 = 0;
                for (lay.proj_ms) |m_| total_proj += m_;
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.batch_proj);
                cmd.setBuffer(in0, 0, 0);
                cmd.setBuffer(in1, 0, 1);
                cmd.setBuffer(in2, 0, 2);
                rt.setW(cmd, @intCast(lay.proj_q.w.off), 3);
                rt.setW(cmd, @intCast(lay.proj_q.s.off), 4);
                rt.setW(cmd, @intCast(lay.proj_q.b.off), 5);
                rt.setW(cmd, @intCast(lay.proj_k.w.off), 6);
                rt.setW(cmd, @intCast(lay.proj_k.s.off), 7);
                rt.setW(cmd, @intCast(lay.proj_k.b.off), 8);
                rt.setW(cmd, @intCast(lay.proj_v.w.off), 9);
                rt.setW(cmd, @intCast(lay.proj_v.s.off), 10);
                rt.setW(cmd, @intCast(lay.proj_v.b.off), 11);
                cmd.setBuffer(rt.q, 0, 12);
                cmd.setBuffer(rt.k, 0, 13);
                cmd.setBuffer(rt.v, 0, 14);
                cmd.setBuffer(rt.r, 0, 15);
                cmd.setBuffer(rt.gate, 0, 16);
                cmd.setBuffer(rt.sink, 0, 17);
                cmd.setBuffer(rt.kdyn, 0, 18);
                cmd.setBuffer(rt.vdyn, 0, 19);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, AffineProjA{ .k = h, .full = sl.is_full, .m = lay.proj_ms }), 20);
                cmd.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const qkv_norm: f32 = 1.0 / @sqrt(@as(f32, if (full) 9.0 else 5.0));
                const qkv_args = QkvArgs{
                    .kv = g.kv_heads,
                    .heads = g.heads,
                    .dim = g.head_dim,
                    .rel = g.d_rel,
                    .extent = lay.extent,
                    .layout = sl.is_full,
                    .dyn = sl.is_full,
                    .tau = sl.is_full,
                    .norm = qkv_norm,
                    .q_threshold = g.qk_clip,
                    .rel_threshold = g.rel_clip,
                    .eps = g.eps,
                    .floor = @floatFromInt(g.log_floor),
                    .alpha = g.log_alpha,
                };
                const row_off = try allocParam(arena, &cur, BatchQkvRow{
                    .k_ring = convAddr(rt.k_conv, sl.slot, kvw),
                    .v_ring = convAddr(rt.v_conv, sl.slot, kvw),
                    .keys = 0,
                    .values = 0,
                    .pos = 0,
                });
                rt.dyn_qkv_row_off[visit_i] = row_off;

                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.qkv);
                cmd.setBuffer(rt.k, 0, 0);
                cmd.setBuffer(rt.v, 0, 1);
                rt.setW(cmd, @intCast(lay.k_conv.off), 2);
                rt.setW(cmd, @intCast(lay.v_conv.off), 3);
                cmd.setBuffer(rt.kdyn, 0, 4);
                cmd.setBuffer(rt.vdyn, 0, 5);
                cmd.setBuffer(rt.q, 0, 6);
                rt.setW(cmd, @intCast(lay.q_scale.off), 7);
                cmd.setBuffer(rt.r, 0, 8);
                rt.setW(cmd, @intCast(lay.r_scale.off), 9);
                rt.setW(cmd, @intCast(lay.rel_bias.off), 10);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, qkv_args), 11);
                cmd.setBuffer(rt.relbias, 0, 12);
                cmd.setBuffer(param_arena, row_off, 13);
                cmd.setBuffer(rt.kc, 0, 14);
                cmd.setBuffer(rt.vc, 0, 15);
                cmd.dispatchGroups(mtl.Size.of(g.kv_heads + g.heads + (g.heads * (lay.extent + 1) + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const arow_off = try allocParam(arena, &cur, BatchRow{
                    .keys = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot],
                    .values = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot],
                    .args = .{
                        .heads = g.heads,
                        .kv = g.kv_heads,
                        .dim = g.head_dim,
                        .rel = g.d_rel,
                        .extent = lay.extent,
                        .window = if (full) 0 else g.window,
                        .pos = 0,
                        .tau = sl.is_full,
                        .splits = 1,
                        .chunk = 1,
                        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(g.head_dim))),
                        .floor = @floatFromInt(g.log_floor),
                        .alpha = g.log_alpha,
                    },
                });
                rt.dyn_attn_row_off[visit_i] = arow_off;
                rt.dyn_attn_split_cmd[visit_i] = idx;

                const cs = active_icb.command(idx);
                cs.setBarrier();
                cs.setPipeline(p.split);
                cs.setBuffer(rt.q, 0, 0);
                cs.setBuffer(rt.relbias, 0, 1);
                cs.setBuffer(rt.sink, 0, 2);
                cs.setBuffer(param_arena, arow_off, 3);
                cs.setBuffer(rt.parts, 0, 4);
                cs.dispatchGroups(mtl.Size.of(g.kv_heads * attn_splits, 1, 1), mtl.Size.of((g.heads / g.kv_heads) * 32, 1, 1));
                idx += 1;

                const cc = active_icb.command(idx);
                cc.setBarrier();
                cc.setPipeline(p.combine);
                cc.setBuffer(rt.parts, 0, 0);
                cc.setBuffer(param_arena, arow_off, 1);
                cc.setBuffer(rt.attn, 0, 2);
                cc.dispatchGroups(mtl.Size.of(g.heads, 1, 1), mtl.Size.of(32, 1, 1));
                idx += 1;
            }

            {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.sig_mul);
                cmd.setBuffer(rt.attn, 0, 0);
                cmd.setBuffer(rt.gate, 0, 1);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, PairA{ .count = qw }), 2);
                cmd.setBuffer(rt.attn, 0, 3);
                cmd.dispatchThreads(mtl.Size.of(qw, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.dense);
                cmd.setBuffer(rt.attn, 0, 0);
                rt.setW(cmd, @intCast(lay.proj_o.w.off), 1);
                rt.setW(cmd, @intCast(lay.proj_o.s.off), 2);
                rt.setW(cmd, @intCast(lay.proj_o.b.off), 3);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, AffineDenseA{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }), 4);
                cmd.setBuffer(rt.projout, 0, 5);
                cmd.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const arow_off = try allocParam(arena, &cur, ConvRow{ .ring = convAddr(rt.attn_conv, sl.slot, h), .pos = 0 });
                rt.dyn_aconv_row_off[visit_i] = arow_off;
                const ca_off = try allocParam(arena, &cur, DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = @intFromBool(!push), .topk = 0 });
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.batch_conv);
                cmd.setBuffer(rt.projout, 0, 0);
                rt.setW(cmd, @intCast(lay.attn_post.off), 1);
                cmd.setBuffer(param_arena, ca_off, 2);
                cmd.setBuffer(param_arena, ca_off, 3);
                cmd.setBuffer(if (!push) rt.x else rt.o, 0, 4);
                cmd.setBuffer(param_arena, arow_off, 5);
                cmd.setBuffer(rt.projout, 0, 6);
                cmd.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            if (push) {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.scatter);
                cmd.setBuffer(rt.x, 0, 0);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, ScatterA{ .width = h, .stride = mix_states, .slot = sl.block_count_before }), 1);
                cmd.setBuffer(rt.blocks, 0, 2);
                cmd.setBuffer(rt.o, 0, 3);
                cmd.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const mq_off: usize = @intCast(rt.sched.globals.mlp_queries[sl.phys].off);
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.batch_mix);
                cmd.setBuffer(rt.blocks, 0, 0);
                cmd.setBuffer(rt.x, 0, 1);
                rt.setW(cmd, mq_off, 2);
                rt.setW(cmd, mq_off, 3);
                rt.setW(cmd, mq_off, 4);
                rt.setW(cmd, mq_off, 5);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = @intFromBool(!bank), .scale = scale_h, .eps = g.eps }), 6);
                cmd.setBuffer(rt.mixout, 0, 7);
                rt.setW(cmd, @intCast(if (!bank) lay.mlp_norm.off else mq_off), 8);
                rt.setW(cmd, mq_off, 9);
                rt.setW(cmd, mq_off, 10);
                cmd.setBuffer(if (!bank) rt.m0 else rt.n0, 0, 11);
                cmd.setBuffer(if (!bank) rt.m0 else rt.n1, 0, 12);
                cmd.setBuffer(if (!bank) rt.m0 else rt.n2, 0, 13);
                cmd.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            if (bank) {
                const ctl = lay.controllers[3];
                const c0 = active_icb.command(idx);
                c0.setBarrier();
                c0.setPipeline(p.rms_pair);
                c0.setBuffer(rt.mixout, 0, 0);
                c0.setBuffer(rt.msg, 0, 1);
                c0.setBuffer(param_arena, try allocParam(arena, &cur, RmsPairA{ .width = h, .xs0 = 1, .xb0 = 0, .xs1 = 4, .xb1 = 3, .eps = g.eps }), 2);
                c0.setBuffer(rt.h2in, 0, 3);
                c0.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                const c1 = active_icb.command(idx);
                c1.setBarrier();
                c1.setPipeline(p.float_mv);
                c1.setBuffer(rt.h2in, 0, 0);
                rt.setW(c1, @intCast(ctl.down.off), 1);
                c1.setBuffer(param_arena, try allocParam(arena, &cur, AffineDenseA{ .n = g.gate_rank, .k = 2 * h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }), 2);
                c1.setBuffer(rt.normed, 0, 3);
                c1.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                const c2 = active_icb.command(idx);
                c2.setBarrier();
                c2.setPipeline(p.batch_float_ctl);
                c2.setBuffer(rt.normed, 0, 0);
                rt.setW(c2, @intCast(ctl.gain.off), 1);
                rt.setW(c2, @intCast(ctl.forget.off), 2);
                c2.setBuffer(rt.mixout, 0, 3);
                c2.setBuffer(rt.msg, 0, 4);
                c2.setBuffer(rt.mixout, 0, 5);
                c2.setBuffer(param_arena, try allocParam(arena, &cur, AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }), 6);
                c2.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                const c3 = active_icb.command(idx);
                c3.setBarrier();
                c3.setPipeline(p.rms);
                c3.setBuffer(rt.mixout, 0, 0);
                rt.setW(c3, @intCast(lay.mlp_norm.off), 1);
                c3.setBuffer(param_arena, try allocParam(arena, &cur, RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }), 2);
                c3.setBuffer(rt.m0, 0, 3);
                c3.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const c0 = active_icb.command(idx);
                c0.setBarrier();
                c0.setPipeline(p.float_mv);
                c0.setBuffer(rt.m0, 0, 0);
                rt.setW(c0, @intCast(lay.router.off), 1);
                c0.setBuffer(param_arena, try allocParam(arena, &cur, AffineDenseA{ .n = g.experts, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }), 2);
                c0.setBuffer(rt.router, 0, 3);
                c0.dispatchGroups(mtl.Size.of((g.experts + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                const c1 = active_icb.command(idx);
                c1.setBarrier();
                c1.setPipeline(p.routepick);
                c1.setBuffer(rt.router, 0, 0);
                rt.setW(c1, @intCast(lay.router_bias.off), 1);
                c1.setBuffer(param_arena, try allocParam(arena, &cur, PickA{ .experts = g.experts, .topk = g.topk, .rows = 1, .route_scale = g.route_scale }), 2);
                c1.setBuffer(rt.picks, 0, 3);
                c1.setBuffer(rt.wts, 0, 4);
                c1.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
                idx += 1;
            }

            if (lay.is_affine4 != 0) {
                const c0 = active_icb.command(idx);
                c0.setBarrier();
                c0.setPipeline(p.aff_up);
                c0.setBuffer(rt.m0, 0, 0);
                rt.setW(c0, @intCast(lay.w13.off), 1);
                rt.setW(c0, @intCast(lay.w13_scale.off), 2);
                rt.setW(c0, @intCast(lay.w13_bias.off), 3);
                c0.setBuffer(rt.picks, 0, 4);
                c0.setBuffer(param_arena, try allocParam(arena, &cur, MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }), 5);
                c0.setBuffer(rt.act, 0, 6);
                c0.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                const c1 = active_icb.command(idx);
                c1.setBarrier();
                c1.setPipeline(p.aff_down);
                c1.setBuffer(rt.act, 0, 0);
                rt.setW(c1, @intCast(lay.w2.off), 1);
                rt.setW(c1, @intCast(lay.w2_scale.off), 2);
                rt.setW(c1, @intCast(lay.w2_bias.off), 3);
                c1.setBuffer(rt.picks, 0, 4);
                c1.setBuffer(param_arena, try allocParam(arena, &cur, MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }), 5);
                c1.setBuffer(rt.down, 0, 6);
                c1.dispatchGroups(mtl.Size.of((h + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            } else {
                const c0 = active_icb.command(idx);
                c0.setBarrier();
                c0.setPipeline(p.gu_nv);
                c0.setBuffer(rt.m0, 0, 0);
                rt.setW(c0, @intCast(lay.w13.off), 1);
                rt.setW(c0, @intCast(lay.w13_scale.off), 2);
                rt.setW(c0, @intCast(lay.w13_global.off), 3);
                rt.setW(c0, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                c0.setBuffer(rt.picks, 0, 5);
                c0.setBuffer(param_arena, try allocParam(arena, &cur, MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }), 6);
                c0.setBuffer(rt.act, 0, 7);
                rt.setW(c0, @intCast(rt.sched.globals.t16.off), 8);
                c0.dispatchGroups(mtl.Size.of(g.expert_inter, g.topk, 1), mtl.Size.of(32, 1, 1));
                idx += 1;

                const c1 = active_icb.command(idx);
                c1.setBarrier();
                c1.setPipeline(p.down1_nv);
                c1.setBuffer(rt.act, 0, 0);
                rt.setW(c1, @intCast(lay.w2.off), 1);
                rt.setW(c1, @intCast(lay.w2_scale.off), 2);
                rt.setW(c1, @intCast(lay.w2_global.off), 3);
                rt.setW(c1, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                c1.setBuffer(rt.picks, 0, 5);
                c1.setBuffer(param_arena, try allocParam(arena, &cur, MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }), 6);
                c1.setBuffer(rt.down, 0, 7);
                rt.setW(c1, @intCast(rt.sched.globals.t16.off), 8);
                c1.dispatchGroups(mtl.Size.of((h + 31) / 32, g.topk, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            {
                const mrow_off = try allocParam(arena, &cur, ConvRow{ .ring = convAddr(rt.mlp_conv, sl.slot, h), .pos = 0 });
                rt.dyn_mconv_row_off[visit_i] = mrow_off;
                const ma_off = try allocParam(arena, &cur, DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = 1, .topk = g.topk });
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.batch_conv);
                cmd.setBuffer(rt.down, 0, 0);
                rt.setW(cmd, @intCast(lay.mlp_post.off), 1);
                cmd.setBuffer(param_arena, ma_off, 2);
                cmd.setBuffer(param_arena, ma_off, 3);
                cmd.setBuffer(rt.x, 0, 4);
                cmd.setBuffer(param_arena, mrow_off, 5);
                cmd.setBuffer(rt.wts, 0, 6);
                cmd.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;
            }

            if (sl.is_prelude_end != 0) {
                const cmd = active_icb.command(idx);
                cmd.setBarrier();
                cmd.setPipeline(p.copy);
                cmd.setBuffer(rt.x, 0, 0);
                cmd.setBuffer(rt.prelude, 0, 1);
                cmd.setBuffer(param_arena, try allocParam(arena, &cur, PairA{ .count = h }), 2);
                cmd.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                idx += 1;

                rt.icb_pre_count = idx;
                active_icb = icb_main;
                idx = 0;
            }
        }

        {
            const oq_off: usize = @intCast(rt.sched.globals.output_query.off);
            const c0 = active_icb.command(idx);
            c0.setBarrier();
            c0.setPipeline(p.batch_mix);
            c0.setBuffer(rt.blocks, 0, 0);
            c0.setBuffer(rt.x, 0, 1);
            rt.setW(c0, oq_off, 2);
            rt.setW(c0, oq_off, 3);
            rt.setW(c0, oq_off, 4);
            rt.setW(c0, oq_off, 5);
            c0.setBuffer(param_arena, try allocParam(arena, &cur, MixArgs{ .width = h, .depth = 5, .stride = mix_states, .k = 1, .norm = 1, .scale = scale_h, .eps = g.eps }), 6);
            c0.setBuffer(rt.mixout, 0, 7);
            rt.setW(c0, @intCast(rt.sched.globals.norm.off), 8);
            rt.setW(c0, oq_off, 9);
            rt.setW(c0, oq_off, 10);
            c0.setBuffer(rt.n0, 0, 11);
            c0.setBuffer(rt.n0, 0, 12);
            c0.setBuffer(rt.n0, 0, 13);
            c0.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
            idx += 1;

            const c1 = active_icb.command(idx);
            c1.setBarrier();
            c1.setPipeline(p.dense);
            c1.setBuffer(rt.n0, 0, 0);
            rt.setW(c1, @intCast(rt.sched.globals.unembed.w.off), 1);
            rt.setW(c1, @intCast(rt.sched.globals.unembed.s.off), 2);
            rt.setW(c1, @intCast(rt.sched.globals.unembed.b.off), 3);
            c1.setBuffer(param_arena, try allocParam(arena, &cur, AffineDenseA{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }), 4);
            c1.setBuffer(rt.logits, 0, 5);
            c1.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
            idx += 1;

            if (gpu_argmax) {
                const c2 = active_icb.command(idx);
                c2.setBarrier();
                c2.setPipeline(p.argmax);
                c2.setBuffer(rt.logits, 0, 0);
                c2.setBuffer(param_arena, try allocParam(arena, &cur, ArgmaxA{ .vocab = g.unpadded_vocab }), 1);
                c2.setBuffer(rt.argmax_out, 0, 2);
                c2.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
                idx += 1;
            }
        }
        rt.icb_pre = icb_pre;
        rt.icb_main = icb_main;
        rt.icb_main_count = idx;
        rt.icb_built_argmax = gpu_argmax;
    }

    pub fn stepIcb(rt: *Runtime, token_id: u32, gpu_argmax: bool) !StepStats {
        if (rt.len >= rt.capacity) return error.ContextFull;
        try rt.buildIcb(gpu_argmax);
        const t0 = nowNs();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();

        rt.ids.slice(u32, 1)[0] = token_id;
        const pos: u32 = @intCast(rt.len);
        const g = rt.sched.geom;
        const kvw: usize = g.kv_heads * g.head_dim;
        const arena = rt.param_arena.contents()[0..rt.param_arena.length()];
        const local_lo: u32 = if (pos + 1 > g.window) pos + 1 - g.window else 0;
        const local_part = attnPartition(pos + 1 - local_lo);
        const full_part = attnPartition(pos + 1);

        for (rt.sched.slots, 0..) |sl, vi| {
            const full = sl.is_full != 0;
            const kv_at = rt.slot_kv_off[sl.slot] + (if (full) @as(usize, pos) else @as(usize, pos % g.window)) * kvw * 2;
            const qr: *BatchQkvRow = @ptrCast(@alignCast(arena.ptr + rt.dyn_qkv_row_off[vi]));
            qr.keys = rt.keys.gpuAddress() + kv_at;
            qr.values = rt.values.gpuAddress() + kv_at;
            qr.pos = pos;

            const part = if (full) full_part else local_part;
            const ar: *BatchRow = @ptrCast(@alignCast(arena.ptr + rt.dyn_attn_row_off[vi]));
            ar.args.pos = pos;
            ar.args.splits = part.splits;
            ar.args.chunk = part.chunk;

            const ac: *ConvRow = @ptrCast(@alignCast(arena.ptr + rt.dyn_aconv_row_off[vi]));
            ac.pos = pos;
            const mc: *ConvRow = @ptrCast(@alignCast(arena.ptr + rt.dyn_mconv_row_off[vi]));
            mc.pos = pos;
        }

        const cb = rt.queue.commandBuffer();
        const enc = cb.compute(.concurrent);
        var ro_ids: [max_weight_chunks + 2]mtl.objc.Id = undefined;
        for (0..rt.weight_chunk_count) |ci| ro_ids[ci] = rt.weight_chunks[ci].buf.id;
        ro_ids[rt.weight_chunk_count] = rt.param_arena.id;
        ro_ids[rt.weight_chunk_count + 1] = rt.ids.id;
        const rw_ids = [_]mtl.objc.Id{
            rt.keys.id, rt.values.id, rt.k_conv.id, rt.v_conv.id, rt.attn_conv.id, rt.mlp_conv.id,
            rt.x.id, rt.prelude.id, rt.passout.id, rt.bank.id, rt.n0.id, rt.n1.id, rt.n2.id, rt.m0.id,
            rt.q.id, rt.k.id, rt.v.id, rt.r.id, rt.kc.id, rt.vc.id, rt.kdyn.id, rt.vdyn.id, rt.sink.id, rt.relbias.id,
            rt.attn.id, rt.gate.id, rt.projout.id, rt.o.id, rt.aconv.id, rt.moe.id, rt.blocks.id, rt.mixout.id,
            rt.msg.id, rt.h2in.id, rt.normed.id, rt.router.id, rt.picks.id, rt.wts.id, rt.act.id, rt.down.id,
            rt.logits.id, rt.parts.id, rt.ctl_in.id, rt.ctl_hid.id, rt.argmax_out.id,
        };
        enc.useResources(ro_ids[0 .. rt.weight_chunk_count + 2], mtl.ResourceUsage.read);
        enc.useResources(&rw_ids, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        enc.execute(rt.icb_pre.?, 0, rt.icb_pre_count);
        enc.barrier();
        enc.execute(rt.icb_main.?, 0, rt.icb_main_count);
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("aot icb failed: {s}", .{msg});
            return error.GpuFailed;
        }
        const gpu_ms = cb.gpuSeconds() * 1000.0;
        const wall_ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        rt.len += 1;
        const tok: u32 = if (gpu_argmax) rt.argmax_out.slice(u32, 1)[0] else cpuArgmax(rt.logits.slice(f32, g.unpadded_vocab));
        return .{
            .wall_ms = wall_ms,
            .gpu_ms = gpu_ms,
            .dispatches = @intCast(rt.icb_pre_count + rt.icb_main_count),
            .command_buffers = 1,
            .syncs = 1,
            .token = tok,
        };
    }

    /// Returns a borrowed view of the most recent step's output logits, valid until the next step, prefill, or `deinit`.
    pub fn logitsSlice(rt: *const Runtime) []const f32 {
        return rt.logits.slice(f32, rt.sched.geom.unpadded_vocab);
    }

    pub fn profileClasses(rt: *Runtime, reps: usize) !void {
        const g = rt.sched.geom;
        const h: u32 = g.hidden;
        const qw: u32 = g.heads * g.head_dim;
        const kvw: u32 = g.kv_heads * g.head_dim;
        const p = &rt.pipes;
        const use_sb = (rt.sched.layout_flags & fmt.layout_interleaved_sb) != 0 and p.aot_dense8_sb != null;
        const pos: u32 = @intCast(if (rt.len > 0) rt.len - 1 else 0);
        const scale_h: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(h)));
        const scale_attn: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(g.head_dim)));
        const norm_post: f32 = 1.0 / @sqrt(@as(f32, 5.0));
        const opt_d1 = (rt.opt_mask & 1) != 0 and p.aot_combine_sig_mul != null;
        const opt_d2 = (rt.opt_mask & 2) != 0 and p.aot_gu_nv != null;
        const opt_d3 = (rt.opt_mask & 4) != 0 and p.aot_ctl_down3 != null;

        const classes = [_][]const u8{
            "attn_proj_48", "qkv_48", "split_48", "combine_sig_48", "o_proj_48",
            "attn_post_48", "mix_norms_all", "controllers_24", "router_pick_48",
            "gu_nv_44", "down1_nv_44", "aff4_exp_4", "mlp_post_48", "lm_head_1",
        };
        var totals: [classes.len]f64 = @splat(0);

        for (0..classes.len) |ci| {
            var sum_ms: f64 = 0;
            for (0..reps) |_| {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = rt.queue.commandBuffer();
                const enc = cb.compute(.serial);
                for (rt.sched.slots) |sl| {
                    const lay = &rt.sched.layers[sl.phys];
                    const full = sl.is_full != 0;
                    const bank = sl.is_bank != 0;
                    switch (ci) {
                        0 => {
                            var total_proj: u32 = 0;
                            for (lay.proj_ms) |m_| total_proj += m_;
                            if (use_sb) {
                                enc.setPipeline(p.aot_batch_proj8_sb.?);
                                enc.setBuffer(rt.n0, 0, 0);
                                enc.setBuffer(rt.n1, 0, 1);
                                enc.setBuffer(rt.n2, 0, 2);
                                rt.setW(enc, @intCast(lay.proj_q.w.off), 3);
                                rt.setW(enc, @intCast(lay.proj_q.sb.off), 4);
                                rt.setW(enc, @intCast(lay.proj_k.w.off), 5);
                                rt.setW(enc, @intCast(lay.proj_k.sb.off), 6);
                                rt.setW(enc, @intCast(lay.proj_v.w.off), 7);
                                rt.setW(enc, @intCast(lay.proj_v.sb.off), 8);
                                enc.setBuffer(rt.q, 0, 9);
                                enc.setBuffer(rt.k, 0, 10);
                                enc.setBuffer(rt.v, 0, 11);
                                enc.setBuffer(rt.r, 0, 12);
                                enc.setBuffer(rt.gate, 0, 13);
                                enc.setBuffer(rt.sink, 0, 14);
                                enc.setBuffer(rt.kdyn, 0, 15);
                                enc.setBuffer(rt.vdyn, 0, 16);
                                enc.setValue(AffineProjA{ .k = h, .full = sl.is_full, .m = lay.proj_ms }, 17);
                                enc.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            } else {
                                enc.setPipeline(p.batch_proj);
                                enc.setBuffer(rt.n0, 0, 0);
                                enc.setBuffer(rt.n1, 0, 1);
                                enc.setBuffer(rt.n2, 0, 2);
                                rt.setW(enc, @intCast(lay.proj_q.w.off), 3);
                                rt.setW(enc, @intCast(lay.proj_q.s.off), 4);
                                rt.setW(enc, @intCast(lay.proj_q.b.off), 5);
                                rt.setW(enc, @intCast(lay.proj_k.w.off), 6);
                                rt.setW(enc, @intCast(lay.proj_k.s.off), 7);
                                rt.setW(enc, @intCast(lay.proj_k.b.off), 8);
                                rt.setW(enc, @intCast(lay.proj_v.w.off), 9);
                                rt.setW(enc, @intCast(lay.proj_v.s.off), 10);
                                rt.setW(enc, @intCast(lay.proj_v.b.off), 11);
                                enc.setBuffer(rt.q, 0, 12);
                                enc.setBuffer(rt.k, 0, 13);
                                enc.setBuffer(rt.v, 0, 14);
                                enc.setBuffer(rt.r, 0, 15);
                                enc.setBuffer(rt.gate, 0, 16);
                                enc.setBuffer(rt.sink, 0, 17);
                                enc.setBuffer(rt.kdyn, 0, 18);
                                enc.setBuffer(rt.vdyn, 0, 19);
                                enc.setValue(AffineProjA{ .k = h, .full = sl.is_full, .m = lay.proj_ms }, 20);
                                enc.dispatchGroups(mtl.Size.of((total_proj + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            }
                        },
                        1 => {
                            const kv_at = rt.slot_kv_off[sl.slot] + (if (full) @as(usize, pos) else @as(usize, pos % g.window)) * kvw * 2;
                            const qkv_row = BatchQkvRow{ .k_ring = convAddr(rt.k_conv, sl.slot, kvw), .v_ring = convAddr(rt.v_conv, sl.slot, kvw), .keys = rt.keys.gpuAddress() + kv_at, .values = rt.values.gpuAddress() + kv_at, .pos = pos };
                            const qkv_args = QkvArgs{ .kv = g.kv_heads, .heads = g.heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent, .layout = sl.is_full, .dyn = sl.is_full, .tau = sl.is_full, .norm = 1.0 / @sqrt(@as(f32, if (full) 9.0 else 5.0)), .q_threshold = g.qk_clip, .rel_threshold = g.rel_clip, .eps = g.eps, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha };
                            enc.setPipeline(p.qkv);
                            enc.setBuffer(rt.k, 0, 0);
                            enc.setBuffer(rt.v, 0, 1);
                            rt.setW(enc, @intCast(lay.k_conv.off), 2);
                            rt.setW(enc, @intCast(lay.v_conv.off), 3);
                            enc.setBuffer(rt.kdyn, 0, 4);
                            enc.setBuffer(rt.vdyn, 0, 5);
                            enc.setBuffer(rt.q, 0, 6);
                            rt.setW(enc, @intCast(lay.q_scale.off), 7);
                            enc.setBuffer(rt.r, 0, 8);
                            rt.setW(enc, @intCast(lay.r_scale.off), 9);
                            rt.setW(enc, @intCast(lay.rel_bias.off), 10);
                            enc.setValue(qkv_args, 11);
                            enc.setBuffer(rt.relbias, 0, 12);
                            enc.setValue(qkv_row, 13);
                            enc.setBuffer(rt.kc, 0, 14);
                            enc.setBuffer(rt.vc, 0, 15);
                            enc.dispatchGroups(mtl.Size.of(g.kv_heads + g.heads + (g.heads * (lay.extent + 1) + 255) / 256, 1, 1), mtl.Size.of(256, 1, 1));
                        },
                        2 => {
                            const win: u32 = if (full) 0 else g.window;
                            const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
                            const part = attnPartition(pos + 1 - lo);
                            const attn_row = BatchRow{ .keys = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot], .values = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot], .args = .{ .heads = g.heads, .kv = g.kv_heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent, .window = win, .pos = pos, .tau = sl.is_full, .splits = part.splits, .chunk = part.chunk, .scale = scale_attn, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha } };
                            enc.setPipeline(p.split);
                            enc.setBuffer(rt.q, 0, 0);
                            enc.setBuffer(rt.relbias, 0, 1);
                            enc.setBuffer(rt.sink, 0, 2);
                            enc.setValue(attn_row, 3);
                            enc.setBuffer(rt.parts, 0, 4);
                            enc.dispatchGroups(mtl.Size.of(g.kv_heads * part.splits, 1, 1), mtl.Size.of((g.heads / g.kv_heads) * 32, 1, 1));
                        },
                        3 => {
                            const win: u32 = if (full) 0 else g.window;
                            const lo: u32 = if (win != 0 and pos + 1 > win) pos + 1 - win else 0;
                            const part = attnPartition(pos + 1 - lo);
                            const attn_row = BatchRow{ .keys = rt.keys.gpuAddress() + rt.slot_kv_off[sl.slot], .values = rt.values.gpuAddress() + rt.slot_kv_off[sl.slot], .args = .{ .heads = g.heads, .kv = g.kv_heads, .dim = g.head_dim, .rel = g.d_rel, .extent = lay.extent, .window = win, .pos = pos, .tau = sl.is_full, .splits = part.splits, .chunk = part.chunk, .scale = scale_attn, .floor = @floatFromInt(g.log_floor), .alpha = g.log_alpha } };
                            if (opt_d1) {
                                enc.setPipeline(p.aot_combine_sig_mul.?);
                                enc.setBuffer(rt.parts, 0, 0);
                                enc.setValue(attn_row, 1);
                                enc.setBuffer(rt.gate, 0, 2);
                                enc.setBuffer(rt.attn, 0, 3);
                                enc.dispatchGroups(mtl.Size.of(g.heads, 1, 1), mtl.Size.of(32, 1, 1));
                            } else {
                                enc.setPipeline(p.combine);
                                enc.setBuffer(rt.parts, 0, 0);
                                enc.setValue(attn_row, 1);
                                enc.setBuffer(rt.attn, 0, 2);
                                enc.dispatchGroups(mtl.Size.of(g.heads, 1, 1), mtl.Size.of(32, 1, 1));
                                enc.setPipeline(p.sig_mul);
                                enc.setBuffer(rt.attn, 0, 0);
                                enc.setBuffer(rt.gate, 0, 1);
                                enc.setValue(PairA{ .count = qw }, 2);
                                enc.setBuffer(rt.attn, 0, 3);
                                enc.dispatchThreads(mtl.Size.of(qw, 1, 1), mtl.Size.of(256, 1, 1));
                            }
                        },
                        4 => {
                            if (use_sb) {
                                enc.setPipeline(p.aot_dense8_sb.?);
                                enc.setBuffer(rt.attn, 0, 0);
                                rt.setW(enc, @intCast(lay.proj_o.w.off), 1);
                                rt.setW(enc, @intCast(lay.proj_o.sb.off), 2);
                                enc.setValue(AffineDenseA{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 3);
                                enc.setBuffer(rt.projout, 0, 4);
                                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            } else {
                                enc.setPipeline(p.dense);
                                enc.setBuffer(rt.attn, 0, 0);
                                rt.setW(enc, @intCast(lay.proj_o.w.off), 1);
                                rt.setW(enc, @intCast(lay.proj_o.s.off), 2);
                                rt.setW(enc, @intCast(lay.proj_o.b.off), 3);
                                enc.setValue(AffineDenseA{ .n = h, .k = qw, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = 0 }, 4);
                                enc.setBuffer(rt.projout, 0, 5);
                                enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            }
                        },
                        5 => {
                            const aconv_row = ConvRow{ .ring = convAddr(rt.attn_conv, sl.slot, h), .pos = pos };
                            const aconv_args = DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = 1, .topk = 0 };
                            enc.setPipeline(p.batch_conv);
                            enc.setBuffer(rt.projout, 0, 0);
                            rt.setW(enc, @intCast(lay.attn_post.off), 1);
                            enc.setValue(aconv_args, 2);
                            enc.setValue(aconv_args, 3);
                            enc.setBuffer(rt.o, 0, 4);
                            enc.setValue(aconv_row, 5);
                            enc.setBuffer(rt.projout, 0, 6);
                            enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                        },
                        6 => {
                            const mq_off: usize = @intCast(rt.sched.globals.mlp_queries[sl.phys].off);
                            enc.setPipeline(p.batch_mix);
                            enc.setBuffer(rt.blocks, 0, 0);
                            enc.setBuffer(rt.x, 0, 1);
                            rt.setW(enc, mq_off, 2);
                            rt.setW(enc, mq_off, 3);
                            rt.setW(enc, mq_off, 4);
                            rt.setW(enc, mq_off, 5);
                            enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_after_attn + 1, .stride = mix_states, .k = 1, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
                            enc.setBuffer(rt.mixout, 0, 7);
                            rt.setW(enc, @intCast(lay.mlp_norm.off), 8);
                            rt.setW(enc, mq_off, 9);
                            rt.setW(enc, mq_off, 10);
                            enc.setBuffer(rt.m0, 0, 11);
                            enc.setBuffer(rt.m0, 0, 12);
                            enc.setBuffer(rt.m0, 0, 13);
                            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                            if (sl.phys > 0) {
                                const q0_off: usize = @intCast(rt.sched.globals.attn_queries[sl.phys - 1].off);
                                enc.setPipeline(p.batch_mix);
                                enc.setBuffer(rt.blocks, 0, 0);
                                enc.setBuffer(rt.x, 0, 1);
                                rt.setW(enc, q0_off, 2);
                                rt.setW(enc, q0_off, 3);
                                rt.setW(enc, q0_off, 4);
                                rt.setW(enc, q0_off, 5);
                                enc.setValue(MixArgs{ .width = h, .depth = sl.block_count_before + 1, .stride = mix_states, .k = if (full) 3 else 1, .norm = 1, .scale = scale_h, .eps = g.eps }, 6);
                                enc.setBuffer(rt.mixout, 0, 7);
                                rt.setW(enc, @intCast(lay.q_norm.off), 8);
                                rt.setW(enc, @intCast(lay.k_norm.off), 9);
                                rt.setW(enc, @intCast(lay.v_norm.off), 10);
                                enc.setBuffer(rt.n0, 0, 11);
                                enc.setBuffer(rt.n1, 0, 12);
                                enc.setBuffer(rt.n2, 0, 13);
                                enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                            }
                        },
                        7 => if (bank) {
                            const kq: u32 = if (full) 3 else 1;
                            const ctl_args = ControlsArgs{ .h = h, .rank = g.gate_rank, .rows = 1, .xs = .{ kq, kq, kq }, .xb = if (full) .{ 0, 1, 2 } else .{ 0, 0, 0 } };
                            if (opt_d3) {
                                enc.setPipeline(p.aot_ctl_down3.?);
                                enc.setBuffer(rt.mixout, 0, 0);
                                enc.setBuffer(rt.msg, 0, 1);
                                enc.setBuffer(rt.ctl_inv, 0, 2);
                                rt.setW(enc, @intCast(lay.controllers[0].down.off), 3);
                                rt.setW(enc, @intCast(lay.controllers[1].down.off), 4);
                                rt.setW(enc, @intCast(lay.controllers[2].down.off), 5);
                                enc.setValue(ctl_args, 6);
                                enc.setBuffer(rt.ctl_hid, 0, 7);
                                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 1) / 2, 1, 3), mtl.Size.of(64, 1, 1));
                            } else {
                                enc.setPipeline(p.ctl_norm);
                                enc.setBuffer(rt.mixout, 0, 0);
                                enc.setBuffer(rt.mixout, 0, 1);
                                enc.setBuffer(rt.mixout, 0, 2);
                                enc.setBuffer(rt.msg, 0, 3);
                                enc.setValue(ctl_args, 4);
                                enc.setBuffer(rt.ctl_in, 0, 5);
                                enc.dispatchGroups(mtl.Size.of(6, 1, 1), mtl.Size.of(256, 1, 1));
                                enc.setPipeline(p.ctl_down);
                                enc.setBuffer(rt.ctl_in, 0, 0);
                                rt.setW(enc, @intCast(lay.controllers[0].down.off), 1);
                                rt.setW(enc, @intCast(lay.controllers[1].down.off), 2);
                                rt.setW(enc, @intCast(lay.controllers[2].down.off), 3);
                                enc.setValue(ctl_args, 4);
                                enc.setBuffer(rt.ctl_hid, 0, 5);
                                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                            }
                            enc.setPipeline(p.ctl_apply);
                            enc.setBuffer(rt.ctl_hid, 0, 0);
                            rt.setW(enc, @intCast(lay.controllers[0].gain.off), 1);
                            rt.setW(enc, @intCast(lay.controllers[1].gain.off), 2);
                            rt.setW(enc, @intCast(lay.controllers[2].gain.off), 3);
                            rt.setW(enc, @intCast(lay.controllers[0].forget.off), 4);
                            rt.setW(enc, @intCast(lay.controllers[1].forget.off), 5);
                            rt.setW(enc, @intCast(lay.controllers[2].forget.off), 6);
                            enc.setBuffer(rt.mixout, 0, 7);
                            enc.setBuffer(rt.mixout, 0, 8);
                            enc.setBuffer(rt.mixout, 0, 9);
                            enc.setBuffer(rt.msg, 0, 10);
                            enc.setBuffer(rt.n0, 0, 11);
                            enc.setBuffer(rt.n1, 0, 12);
                            enc.setBuffer(rt.n2, 0, 13);
                            enc.setValue(ctl_args, 14);
                            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 3), mtl.Size.of(256, 1, 1));
                            enc.setPipeline(p.rms3);
                            enc.setBuffer(rt.n0, 0, 0);
                            enc.setBuffer(rt.n1, 0, 1);
                            enc.setBuffer(rt.n2, 0, 2);
                            rt.setW(enc, @intCast(lay.q_norm.off), 3);
                            rt.setW(enc, @intCast(lay.k_norm.off), 4);
                            rt.setW(enc, @intCast(lay.v_norm.off), 5);
                            enc.setValue(Rms3A{ .width = h, .xs = .{ 1, 1, 1 }, .xb = .{ 0, 0, 0 }, .eps = g.eps }, 6);
                            enc.setBuffer(rt.n0, 0, 7);
                            enc.setBuffer(rt.n1, 0, 8);
                            enc.setBuffer(rt.n2, 0, 9);
                            enc.dispatchGroups(mtl.Size.of(3, 1, 1), mtl.Size.of(256, 1, 1));
                            const ctl = lay.controllers[3];
                            if (opt_d3) {
                                enc.setPipeline(p.aot_ctl_down1.?);
                                enc.setBuffer(rt.mixout, 0, 0);
                                enc.setBuffer(rt.msg, 0, 1);
                                enc.setBuffer(rt.ctl_inv, 0, 2);
                                rt.setW(enc, @intCast(ctl.down.off), 3);
                                enc.setValue(AffineDenseA{ .n = g.gate_rank, .k = 2 * h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 4);
                                enc.setBuffer(rt.normed, 0, 5);
                                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 1) / 2, 1, 1), mtl.Size.of(64, 1, 1));
                            } else {
                                enc.setPipeline(p.rms_pair);
                                enc.setBuffer(rt.mixout, 0, 0);
                                enc.setBuffer(rt.msg, 0, 1);
                                enc.setValue(RmsPairA{ .width = h, .xs0 = 1, .xb0 = 0, .xs1 = 4, .xb1 = 3, .eps = g.eps }, 2);
                                enc.setBuffer(rt.h2in, 0, 3);
                                enc.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(256, 1, 1));
                                enc.setPipeline(p.float_mv);
                                enc.setBuffer(rt.h2in, 0, 0);
                                rt.setW(enc, @intCast(ctl.down.off), 1);
                                enc.setValue(AffineDenseA{ .n = g.gate_rank, .k = 2 * h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
                                enc.setBuffer(rt.normed, 0, 3);
                                enc.dispatchGroups(mtl.Size.of((g.gate_rank + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            }
                            enc.setPipeline(p.batch_float_ctl);
                            enc.setBuffer(rt.normed, 0, 0);
                            rt.setW(enc, @intCast(ctl.gain.off), 1);
                            rt.setW(enc, @intCast(ctl.forget.off), 2);
                            enc.setBuffer(rt.mixout, 0, 3);
                            enc.setBuffer(rt.msg, 0, 4);
                            enc.setBuffer(rt.mixout, 0, 5);
                            enc.setValue(AffineBatchCtlA{ .k = g.gate_rank, .m0 = h, .m1 = h, .h = h, .cxs = 1, .cxb = 0, .mxs = 4, .mxb = 3, .ys = 1, .yb = 0 }, 6);
                            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            enc.setPipeline(p.rms);
                            enc.setBuffer(rt.mixout, 0, 0);
                            rt.setW(enc, @intCast(lay.mlp_norm.off), 1);
                            enc.setValue(RmsBA{ .width = h, .rows = 1, .xs = 1, .xb = 0, .ys = 1, .yb = 0, .wb = 1, .eps = g.eps }, 2);
                            enc.setBuffer(rt.m0, 0, 3);
                            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(256, 1, 1));
                        },
                        8 => {
                            enc.setPipeline(p.float_mv);
                            enc.setBuffer(rt.m0, 0, 0);
                            rt.setW(enc, @intCast(lay.router.off), 1);
                            enc.setValue(AffineDenseA{ .n = g.experts, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 0, .cap = 0 }, 2);
                            enc.setBuffer(rt.router, 0, 3);
                            enc.dispatchGroups(mtl.Size.of((g.experts + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                            enc.setPipeline(p.routepick);
                            enc.setBuffer(rt.router, 0, 0);
                            rt.setW(enc, @intCast(lay.router_bias.off), 1);
                            enc.setValue(PickA{ .experts = g.experts, .topk = g.topk, .rows = 1, .route_scale = g.route_scale }, 2);
                            enc.setBuffer(rt.picks, 0, 3);
                            enc.setBuffer(rt.wts, 0, 4);
                            enc.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
                        },
                        9 => if (lay.is_affine4 == 0) {
                            if (opt_d2) {
                                enc.setPipeline(p.aot_gu_nv.?);
                                enc.setBuffer(rt.m0, 0, 0);
                                rt.setW(enc, @intCast(lay.w13.off), 1);
                                rt.setW(enc, @intCast(lay.w13_scale.off), 2);
                                rt.setW(enc, @intCast(lay.w13_global.off), 3);
                                enc.setBuffer(rt.picks, 0, 4);
                                enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
                                enc.setBuffer(rt.act, 0, 6);
                                rt.setW(enc, @intCast(rt.sched.globals.t16.off), 7);
                                rt.setW(enc, @intCast(rt.sched.globals.t256.off), 8);
                                enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                            } else {
                                enc.setPipeline(p.gu_nv);
                                enc.setBuffer(rt.m0, 0, 0);
                                rt.setW(enc, @intCast(lay.w13.off), 1);
                                rt.setW(enc, @intCast(lay.w13_scale.off), 2);
                                rt.setW(enc, @intCast(lay.w13_global.off), 3);
                                rt.setW(enc, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                                enc.setBuffer(rt.picks, 0, 5);
                                enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 6);
                                enc.setBuffer(rt.act, 0, 7);
                                rt.setW(enc, @intCast(rt.sched.globals.t16.off), 8);
                                enc.dispatchGroups(mtl.Size.of(g.expert_inter, g.topk, 1), mtl.Size.of(32, 1, 1));
                            }
                        },
                        10 => if (lay.is_affine4 == 0) {
                            if (opt_d2) {
                                enc.setPipeline(p.aot_down1_nv.?);
                                enc.setBuffer(rt.act, 0, 0);
                                rt.setW(enc, @intCast(lay.w2.off), 1);
                                rt.setW(enc, @intCast(lay.w2_scale.off), 2);
                                rt.setW(enc, @intCast(lay.w2_global.off), 3);
                                enc.setBuffer(rt.picks, 0, 4);
                                enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
                                enc.setBuffer(rt.down, 0, 6);
                                rt.setW(enc, @intCast(rt.sched.globals.t16.off), 7);
                                rt.setW(enc, @intCast(rt.sched.globals.t256.off), 8);
                                enc.dispatchGroups(mtl.Size.of((h + 31) / 32, g.topk, 1), mtl.Size.of(256, 1, 1));
                            } else {
                                enc.setPipeline(p.down1_nv);
                                enc.setBuffer(rt.act, 0, 0);
                                rt.setW(enc, @intCast(lay.w2.off), 1);
                                rt.setW(enc, @intCast(lay.w2_scale.off), 2);
                                rt.setW(enc, @intCast(lay.w2_global.off), 3);
                                rt.setW(enc, @intCast(rt.sched.globals.nvfp4_den.off), 4);
                                enc.setBuffer(rt.picks, 0, 5);
                                enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 6);
                                enc.setBuffer(rt.down, 0, 7);
                                rt.setW(enc, @intCast(rt.sched.globals.t16.off), 8);
                                enc.dispatchGroups(mtl.Size.of((h + 31) / 32, g.topk, 1), mtl.Size.of(256, 1, 1));
                            }
                        },
                        11 => if (lay.is_affine4 != 0) {
                            enc.setPipeline(p.aff_up);
                            enc.setBuffer(rt.m0, 0, 0);
                            rt.setW(enc, @intCast(lay.w13.off), 1);
                            rt.setW(enc, @intCast(lay.w13_scale.off), 2);
                            rt.setW(enc, @intCast(lay.w13_bias.off), 3);
                            enc.setBuffer(rt.picks, 0, 4);
                            enc.setValue(MoeBA{ .k = h, .n = g.expert_inter, .slots = g.topk, .topk = g.topk, .gate_cap = g.gate_cap, .up_cap = g.up_cap }, 5);
                            enc.setBuffer(rt.act, 0, 6);
                            enc.dispatchGroups(mtl.Size.of((g.expert_inter + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                            enc.setPipeline(p.aff_down);
                            enc.setBuffer(rt.act, 0, 0);
                            rt.setW(enc, @intCast(lay.w2.off), 1);
                            rt.setW(enc, @intCast(lay.w2_scale.off), 2);
                            rt.setW(enc, @intCast(lay.w2_bias.off), 3);
                            enc.setBuffer(rt.picks, 0, 4);
                            enc.setValue(MoeBA{ .k = g.expert_inter, .n = h, .slots = g.topk, .topk = g.topk, .gate_cap = 0, .up_cap = 0 }, 5);
                            enc.setBuffer(rt.down, 0, 6);
                            enc.dispatchGroups(mtl.Size.of((h + 7) / 8, g.topk, 1), mtl.Size.of(256, 1, 1));
                        },
                        12 => {
                            const mconv_row = ConvRow{ .ring = convAddr(rt.mlp_conv, sl.slot, h), .pos = pos };
                            const mconv_args = DecodeConvA{ .channels = h, .heads = 0, .ring = conv_ring, .pos = 0, .layout = 0, .norm = norm_post, .residual = 1, .topk = g.topk };
                            enc.setPipeline(p.batch_conv);
                            enc.setBuffer(rt.down, 0, 0);
                            rt.setW(enc, @intCast(lay.mlp_post.off), 1);
                            enc.setValue(mconv_args, 2);
                            enc.setValue(mconv_args, 3);
                            enc.setBuffer(rt.o, 0, 4);
                            enc.setValue(mconv_row, 5);
                            enc.setBuffer(rt.wts, 0, 6);
                            enc.dispatchThreads(mtl.Size.of(h, 1, 1), mtl.Size.of(256, 1, 1));
                        },
                        13 => {},
                        else => unreachable,
                    }
                }
                if (ci == 13) {
                    if (use_sb) {
                        enc.setPipeline(p.aot_dense8_sb.?);
                        enc.setBuffer(rt.n0, 0, 0);
                        rt.setW(enc, @intCast(rt.sched.globals.unembed.w.off), 1);
                        rt.setW(enc, @intCast(rt.sched.globals.unembed.sb.off), 2);
                        enc.setValue(AffineDenseA{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 3);
                        enc.setBuffer(rt.logits, 0, 4);
                        enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    } else {
                        enc.setPipeline(p.dense);
                        enc.setBuffer(rt.n0, 0, 0);
                        rt.setW(enc, @intCast(rt.sched.globals.unembed.w.off), 1);
                        rt.setW(enc, @intCast(rt.sched.globals.unembed.s.off), 2);
                        rt.setW(enc, @intCast(rt.sched.globals.unembed.b.off), 3);
                        enc.setValue(AffineDenseA{ .n = g.unpadded_vocab, .k = h, .rows = 1, .xs = 1, .xb = 0, .bits = 8, .cap = g.logit_cap }, 4);
                        enc.setBuffer(rt.logits, 0, 5);
                        enc.dispatchGroups(mtl.Size.of((g.unpadded_vocab + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                    }
                }
                enc.end();
                cb.commit();
                cb.wait();
                sum_ms += cb.gpuSeconds() * 1000.0;
            }
            totals[ci] = sum_ms / @as(f64, @floatFromInt(reps));
        }
        var grand: f64 = 0;
        for (classes, 0..) |name, ci| {
            grand += totals[ci];
            std.debug.print("profile {s:<16}: {d:6.3} ms\n", .{ name, totals[ci] });
        }
        std.debug.print("profile SUM             : {d:6.3} ms (opt={d}, use_sb={})\n", .{ grand, rt.opt_mask, use_sb });
    }
};

pub fn cpuArgmax(x: []const f32) u32 {
    var best: usize = 0;
    for (x, 0..) |v, i| {
        if (v > x[best]) best = i;
    }
    return @intCast(best);
}

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

test "allocParam enforces alignment and fails closed on arena overflow" {
    var buf: [32]u8 = @splat(0);
    var cur: usize = 0;
    const o0 = try Runtime.allocParam(&buf, &cur, @as(u64, 0x1122334455667788));
    try std.testing.expectEqual(@as(usize, 0), o0);
    try std.testing.expectEqual(@as(usize, 8), cur);
    const o1 = try Runtime.allocParam(&buf, &cur, @as(u64, 0x99aabbccddeeff00));
    try std.testing.expectEqual(@as(usize, 16), o1);
    try std.testing.expectEqual(@as(usize, 24), cur);
    try std.testing.expectError(error.ParamArenaFull, Runtime.allocParam(&buf, &cur, @as(u64, 1)));
}

test "Runtime.open fails closed on missing or invalid-schema bundle" {
    const io = std.testing.io;
    try std.testing.expectError(error.MissingFile, Runtime.open(std.testing.allocator, io, "/nonexistent-tf-aot-bundle-path", .{ .verify_only = true }));
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const manifest_bad = "{\"schema\":99,\"target_arch\":\"applegpu_g17s\",\"min_generation\":17,\"context_cap\":32768,\"layout_flags\":0,\"files\":[]}";
    try tmp.dir.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest_bad });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &path_buf);
    try std.testing.expectError(error.BadVersion, Runtime.open(std.testing.allocator, io, path_buf[0..n], .{ .verify_only = true }));
}

test "verifyTree rejects empty rows and short parent arrays before GPU access" {
    var fake_rt: Runtime = undefined;
    try std.testing.expectError(error.InvalidVerifyRows, prefill_mod.verifyTree(&fake_rt, &.{}, null, false, false));
    const ids = [_]u32{ 1, 2, 3, 4 };
    const short_parents = [_]u8{ 0, 1 };
    try std.testing.expectError(error.BadDraftParents, prefill_mod.verifyTree(&fake_rt, &ids, &short_parents, false, false));
}
