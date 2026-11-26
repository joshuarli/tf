const std = @import("std");

pub const magic = [8]u8{ 'T', 'F', 'A', 'O', 'T', '0', '1', 0 };
pub const schema_version: u32 = 1;
pub const page_align: u64 = 16384;
pub const num_layers: usize = 24;
pub const num_slots: usize = 48;
pub const num_core_layers: usize = 12;
pub const core_start: u32 = 6;
pub const core_end: u32 = 18;
pub const num_passes: u32 = 3;

pub const layout_interleaved_sb: u32 = 1 << 0;
pub const layout_predivided_nvfp4: u32 = 1 << 1;
pub const layout_trimmed_lm_head: u32 = 1 << 2;
pub const layout_colocated_nvfp4: u32 = 1 << 3;
pub const layout_fused_kernels: u32 = 1 << 4;
pub const layout_non_indirect: u32 = 1 << 5;
pub const layout_packed_w13_gf: u32 = 1 << 6;
pub const max_weight_chunks: usize = 8;
pub const max_chunk_bytes: u64 = 3_500_000_000;

pub const ChunkPlan = struct {
    starts: [max_weight_chunks]u64 = @splat(0),
    ends: [max_weight_chunks]u64 = @splat(0),
    count: usize = 0,
};

pub fn planWeightChunks(sched: *const Schedule) ChunkPlan {
    var plan = ChunkPlan{};
    var cur_start: u64 = 0;
    for (0..num_layers) |li| {
        const layer_start = sched.layers[li].q_norm.off;
        const layer_end = if (li + 1 < num_layers) sched.layers[li + 1].q_norm.off else sched.weights_size;
        if (layer_end - cur_start > max_chunk_bytes and layer_start > cur_start) {
            plan.starts[plan.count] = cur_start;
            plan.ends[plan.count] = layer_start;
            plan.count += 1;
            cur_start = layer_start;
        }
    }
    plan.starts[plan.count] = cur_start;
    plan.ends[plan.count] = sched.weights_size;
    plan.count += 1;
    return plan;
}

pub const TensorRef = extern struct {
    off: u64 = 0,
    bytes: u64 = 0,
};

pub const AffineRef = extern struct {
    w: TensorRef = .{},
    s: TensorRef = .{},
    b: TensorRef = .{},
    sb: TensorRef = .{},
    rows: u32 = 0,
    cols: u32 = 0,
    bits: u32 = 8,
    pad: u32 = 0,
};

pub const ControllerRef = extern struct {
    down: TensorRef = .{},
    gain: TensorRef = .{},
    forget: TensorRef = .{},
};

pub const LayerPlan = extern struct {
    is_full: u32 = 0,
    is_affine4: u32 = 0,
    has_controllers: u32 = 0,
    extent: u32 = 1024,
    q_norm: TensorRef = .{},
    k_norm: TensorRef = .{},
    v_norm: TensorRef = .{},
    proj_q: AffineRef = .{},
    proj_k: AffineRef = .{},
    proj_v: AffineRef = .{},
    proj_o: AffineRef = .{},
    proj_ms: [8]u32 = @splat(0),
    q_scale: TensorRef = .{},
    r_scale: TensorRef = .{},
    rel_bias: TensorRef = .{},
    k_conv: TensorRef = .{},
    v_conv: TensorRef = .{},
    attn_post: TensorRef = .{},
    mlp_norm: TensorRef = .{},
    mlp_post: TensorRef = .{},
    router: TensorRef = .{},
    router_bias: TensorRef = .{},
    w13: TensorRef = .{},
    w13_scale: TensorRef = .{},
    w13_global: TensorRef = .{},
    w13_bias: TensorRef = .{},
    w2: TensorRef = .{},
    w2_scale: TensorRef = .{},
    w2_global: TensorRef = .{},
    w2_bias: TensorRef = .{},
    controllers: [4]ControllerRef = @splat(.{}),
};

pub const SlotPlan = extern struct {
    slot: u32 = 0,
    phys: u32 = 0,
    visit: u32 = 0,
    pass: u32 = 0,
    is_full: u32 = 0,
    is_bank: u32 = 0,
    push_block: u32 = 0,
    block_count_before: u32 = 0,
    block_count_after_attn: u32 = 0,
    bank_n: u32 = 0,
    cross_pass_at: u32 = 0,
    is_pass_start: u32 = 0,
    is_prelude_end: u32 = 0,
    is_flush_after: u32 = 0,
    local_index: u32 = 0,
    global_index: u32 = 0,
};

pub const Geometry = extern struct {
    hidden: u32 = 2304,
    vocab: u32 = 201024,
    unpadded_vocab: u32 = 200058,
    heads: u32 = 24,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    d_rel: u32 = 16,
    window: u32 = 1024,
    rel_extent: u32 = 1024,
    log_floor: u32 = 8192,
    experts: u32 = 128,
    topk: u32 = 8,
    expert_inter: u32 = 1152,
    gate_rank: u32 = 64,
    block_layers: u32 = 6,
    context_cap: u32 = 32768,
    eos_token_id: u32 = 200006,
    pad0: u32 = 0,
    eps: f32 = 1e-6,
    qk_clip: f32 = 22.627416997969522,
    rel_clip: f32 = 8.0,
    log_alpha: f32 = 0.1,
    route_scale: f32 = 8.0,
    gate_cap: f32 = 4.0,
    up_cap: f32 = 25.0,
    logit_cap: f32 = 30.0,
};

pub const GlobalWeights = extern struct {
    embed: AffineRef = .{},
    embed_norm: TensorRef = .{},
    norm: TensorRef = .{},
    unembed: AffineRef = .{},
    output_query: TensorRef = .{},
    cross_pass_q: TensorRef = .{},
    cross_pass_kv: TensorRef = .{},
    attn_queries: [num_layers - 1]TensorRef = @splat(.{}),
    mlp_queries: [num_layers]TensorRef = @splat(.{}),
    t16: TensorRef = .{},
    t256: TensorRef = .{},
    nvfp4_den: TensorRef = .{},
};

pub const Schedule = extern struct {
    magic: [8]u8 = magic,
    version: u32 = schema_version,
    layout_flags: u32 = 0,
    min_generation: u32 = 17,
    target_arch_len: u32 = 0,
    target_arch: [32]u8 = @splat(0),
    weights_size: u64 = 0,
    weights_sha256: [32]u8 = @splat(0),
    chunk_count: u32 = 0,
    chunk_pad: u32 = 0,
    chunk_starts: [max_weight_chunks]u64 = @splat(0),
    chunk_ends: [max_weight_chunks]u64 = @splat(0),
    chunk_sha256: [max_weight_chunks][32]u8 = @splat(@splat(0)),
    geom: Geometry = .{},
    globals: GlobalWeights = .{},
    layers: [num_layers]LayerPlan = @splat(.{}),
    slots: [num_slots]SlotPlan = @splat(.{}),

    pub fn archName(self: *const Schedule) []const u8 {
        const n = @min(@as(usize, self.target_arch_len), self.target_arch.len);
        return self.target_arch[0..n];
    }

    pub fn setArchName(self: *Schedule, name: []const u8) void {
        const n = @min(name.len, self.target_arch.len);
        self.target_arch = @splat(0);
        @memcpy(self.target_arch[0..n], name[0..n]);
        self.target_arch_len = @intCast(n);
    }
};

pub const FileEntry = struct {
    name: []const u8,
    size: u64,
    sha256: []const u8,
};

pub const Manifest = struct {
    schema: u32 = 1,
    model_id: []const u8 = "Inkling-2-Mini-oQ4e-mtp@3eea042751cd",
    target_arch: []const u8 = "applegpu_g17s",
    min_generation: u32 = 17,
    context_cap: u32 = 32768,
    layout_flags: u32 = 0,
    quant_contract: []const u8 = "oq4e-nvfp4-0..19-affine4-20..23-affine8-dense-f16-kv-d18-exact",
    files: []const FileEntry,
};

pub const VerifyError = error{
    BadMagic,
    BadVersion,
    UnsupportedDevice,
    CorruptArtifact,
    MissingFile,
    SizeMismatch,
    DigestMismatch,
};

pub fn hexEncode(digest: [32]u8) [64]u8 {
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn checkDevice(actual_arch: []const u8, actual_gen: u32, expected_arch: []const u8, min_gen: u32) VerifyError!void {
    if (actual_gen < min_gen) return error.UnsupportedDevice;
    if (expected_arch.len > 0 and !std.mem.eql(u8, actual_arch, expected_arch)) return error.UnsupportedDevice;
}

pub fn validateScheduleHeader(bytes: []const u8) VerifyError!*const Schedule {
    if (bytes.len != @sizeOf(Schedule)) return error.CorruptArtifact;
    const s: *const Schedule = @ptrCast(@alignCast(bytes.ptr));
    if (!std.mem.eql(u8, &s.magic, &magic)) return error.BadMagic;
    if (s.version != schema_version) return error.BadVersion;
    if (s.geom.hidden != 2304 or s.geom.context_cap > 32768) return error.CorruptArtifact;
    return s;
}

pub fn buildSlots(layers: *const [num_layers]LayerPlan) [num_slots]SlotPlan {
    var out: [num_slots]SlotPlan = undefined;
    var visit: u32 = 0;
    var count: u32 = 0;
    var locals: u32 = 0;
    var globals: u32 = 0;
    var idx: usize = 0;

    for (0..core_start) |i| {
        const phys: u32 = @intCast(i);
        const full = layers[i].is_full != 0;
        const push = (phys % 6) == 0;
        const after_attn = count + @intFromBool(push);
        out[idx] = .{
            .slot = phys,
            .phys = phys,
            .visit = visit,
            .pass = 0,
            .is_full = @intFromBool(full),
            .is_bank = 0,
            .push_block = @intFromBool(push),
            .block_count_before = count,
            .block_count_after_attn = after_attn,
            .is_prelude_end = @intFromBool(i + 1 == core_start),
            .is_flush_after = @intFromBool(phys % 6 == 5),
            .local_index = locals,
            .global_index = globals,
        };
        if (full) globals += 1 else locals += 1;
        count = after_attn;
        visit += 1;
        idx += 1;
    }

    const prelude = count;
    const span: u32 = core_end - core_start;
    for (0..num_passes) |p| {
        const pass: u32 = @intCast(p);
        var bank_n: u32 = 0;
        if (pass > 0) {
            bank_n = count - (prelude + 1);
            count = prelude;
        }
        for (core_start..core_end) |ip_usize| {
            const ip: u32 = @intCast(ip_usize);
            const slot: u32 = if (pass == 0) ip else @as(u32, num_layers) + (pass - 1) * span + (ip - core_start);
            const full = layers[ip_usize].is_full != 0;
            const push = (ip % 6) == 0;
            const after_attn = count + @intFromBool(push);
            out[idx] = .{
                .slot = slot,
                .phys = ip,
                .visit = visit,
                .pass = pass,
                .is_full = @intFromBool(full),
                .is_bank = @intFromBool(pass > 0),
                .push_block = @intFromBool(push),
                .block_count_before = count,
                .block_count_after_attn = after_attn,
                .bank_n = bank_n,
                .cross_pass_at = ip - core_start,
                .is_pass_start = @intFromBool(pass > 0 and ip == core_start),
                .is_flush_after = @intFromBool(slot % 6 == 5),
                .local_index = locals,
                .global_index = globals,
            };
            if (full) globals += 1 else locals += 1;
            count = after_attn;
            visit += 1;
            idx += 1;
        }
    }

    for (core_end..num_layers) |i| {
        const phys: u32 = @intCast(i);
        const full = layers[i].is_full != 0;
        const push = (phys % 6) == 0;
        const after_attn = count + @intFromBool(push);
        out[idx] = .{
            .slot = phys,
            .phys = phys,
            .visit = visit,
            .pass = num_passes,
            .is_full = @intFromBool(full),
            .is_bank = 0,
            .push_block = @intFromBool(push),
            .block_count_before = count,
            .block_count_after_attn = after_attn,
            .local_index = locals,
            .global_index = globals,
        };
        if (full) globals += 1 else locals += 1;
        count = after_attn;
        visit += 1;
        idx += 1;
    }

    std.debug.assert(idx == num_slots);
    std.debug.assert(locals == 40 and globals == 8);
    return out;
}

pub fn alignUp(val: u64, alignment: u64) u64 {
    return (val + alignment - 1) & ~(alignment - 1);
}

test "schedule slot topology matches inkling2_loop 48-slot layout" {
    var layers: [num_layers]LayerPlan = @splat(.{});
    for (&layers, 0..) |*l, i| {
        l.is_full = @intFromBool(i % 6 == 5);
        l.is_affine4 = @intFromBool(i >= 20);
        l.has_controllers = @intFromBool(i >= 6 and i < 18);
    }
    const slots = buildSlots(&layers);
    try std.testing.expectEqual(@as(u32, 0), slots[0].slot);
    try std.testing.expectEqual(@as(u32, 1), slots[5].is_prelude_end);
    try std.testing.expectEqual(@as(u32, 1), slots[18].is_pass_start);
    try std.testing.expectEqual(@as(u32, 24), slots[18].slot);
    try std.testing.expectEqual(@as(u32, 1), slots[18].bank_n);
    try std.testing.expectEqual(@as(u32, 36), slots[30].slot);
    try std.testing.expectEqual(@as(u32, 18), slots[42].slot);
    try std.testing.expectEqual(@as(u32, 23), slots[47].slot);
    try std.testing.expectEqual(@as(u32, 4), slots[47].block_count_after_attn);
}

test "device and schedule checks fail closed on mismatch or corruption" {
    try checkDevice("applegpu_g17s", 17, "applegpu_g17s", 17);
    try std.testing.expectError(error.UnsupportedDevice, checkDevice("applegpu_g13g", 13, "applegpu_g17s", 17));
    try std.testing.expectError(error.UnsupportedDevice, checkDevice("applegpu_g16s", 17, "applegpu_g17s", 17));
    var s = Schedule{};
    s.setArchName("applegpu_g17s");
    try std.testing.expectEqualStrings("applegpu_g17s", s.archName());
    var bytes = std.mem.toBytes(s);
    _ = try validateScheduleHeader(&bytes);
    bytes[0] = 'X';
    try std.testing.expectError(error.BadMagic, validateScheduleHeader(&bytes));
    try std.testing.expectError(error.CorruptArtifact, validateScheduleHeader(bytes[0..16]));
}
