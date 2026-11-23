#include <metal_stdlib>
using namespace metal;

// The checkpoint's MTP head (mtp.safetensors): a three-layer block drafter over the target's tapped hidden states and
// a recurrent Markov head. Activations are f32; the head's bf16 weights are widened per read. Rotations follow MLX's
// non-traditional RoPE: pairs (i, i + dim / 2), angle position * 2^(-i / (dim / 2) * log2(theta)).

// OUT[mix][row] = rms_norm(sum_t alpha[mix][t] * TAPS[t][row0 + row], W + norms[mix]) for every mix: the context row
// each draft layer projects, then the Markov head's. TAPS is tap-major [taps][tap_rows][hidden].
struct i2m_mix_args { uint hidden, rows, taps, tap_rows, row0, mixes; float eps; uint norms[4]; float alpha[32]; };

[[kernel]] void i2m_mix(const device float* TAPS [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2m_mix_args& a [[buffer(2)]], device float* OUT [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint2 nth2 [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint row = tg.x, mix = tg.y, nth = nth2.x;
  if (row >= a.rows || mix >= a.mixes) return;
  device float* out = OUT + ((size_t)mix * a.rows + row) * a.hidden;
  float sum = 0.0f;
  for (uint i = tid; i < a.hidden; i += nth) {
    float v = 0.0f;
    for (uint t = 0; t < a.taps; ++t) v += TAPS[((size_t)t * a.tap_rows + a.row0 + row) * a.hidden + i] * a.alpha[mix * 8u + t];
    out[i] = v;
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.hidden) + a.eps);
  const device bfloat* w = W + a.norms[mix];
  for (uint i = tid; i < a.hidden; i += nth) out[i] = out[i] * inv * float(w[i]);
}

// Context keys and values: per head, K = rotate(rms_norm(K, k_norm_ctx), pos) and V as projected, stored f16 at the
// row's stream position in every draft layer. K and V are [layers][rows][kv][dim]; a row with keep 0 is not stored.
struct i2m_ctx_row { device half* keys; device half* values; uint pos, capacity, keep, pad; };
struct i2m_ctx_args { uint kv, dim, rows, layers; float eps, log_theta; uint norms[8]; };

[[kernel]] void i2m_ctx_store(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    const device bfloat* W [[buffer(2)]], constant i2m_ctx_args& a [[buffer(3)]],
    constant i2m_ctx_row* rows [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  const uint row = tg.x, head = tg.y, layer = tg.z;
  const i2m_ctx_row r = rows[row];
  if (r.keep == 0u) return;
  const uint hd = a.dim / 2u;
  const size_t at = (((size_t)layer * a.rows + row) * a.kv + head) * a.dim;
  const float k0 = K[at + tid], k1 = K[at + tid + hd];
  const float sum = simd_sum(fma(k0, k0, k1 * k1));
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < (hd + 31u) / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.dim) + a.eps);
  const device bfloat* w = W + a.norms[layer];
  const float x0 = k0 * inv * float(w[tid]), x1 = k1 * inv * float(w[tid + hd]);
  const float theta = float(r.pos) * exp2(-float(tid) / float(hd) * a.log_theta);
  const float c = precise::cos(theta), s = precise::sin(theta);
  const size_t dst = (((size_t)layer * r.capacity + r.pos) * a.kv + head) * a.dim;
  r.keys[dst + tid] = half(x0 * c - x1 * s);
  r.keys[dst + tid + hd] = half(x0 * s + x1 * c);
  r.values[dst + tid] = half(V[at + tid]);
  r.values[dst + tid + hd] = half(V[at + tid + hd]);
}

// H[slot * block + j]: the anchor's normalized embedding at j = 0, the trained mask embedding at every later row.
struct i2m_block_args { uint hidden, block, rows; };

[[kernel]] void i2m_block(const device float* ANCHOR [[buffer(0)]], const device bfloat* MASK [[buffer(1)]],
    constant i2m_block_args& a [[buffer(2)]], device float* H [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.hidden || p.y >= a.rows) return;
  const uint slot = p.y / a.block, j = p.y % a.block;
  H[(size_t)p.y * a.hidden + p.x] = j == 0u ? ANCHOR[(size_t)slot * a.hidden + p.x] : float(MASK[p.x]);
}

// Block queries and keys in place: per head rms_norm with q_norm or k_norm, then rotation at POS[slot] + j.
// Q is [rows][q_heads][dim] and K [rows][kv_heads][dim]; one threadgroup a (row, head), dim / 2 threads.
struct i2m_qk_args { uint q_heads, kv_heads, dim, block; float eps, log_theta; uint q_norm, k_norm; };

[[kernel]] void i2m_qk(device float* Q [[buffer(0)]], device float* K [[buffer(1)]], const device bfloat* W [[buffer(2)]],
    const device uint* POS [[buffer(3)]], constant i2m_qk_args& a [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  const uint row = tg.x, head = tg.y;
  const bool is_q = head < a.q_heads;
  device float* x = is_q ? Q + ((size_t)row * a.q_heads + head) * a.dim
                         : K + ((size_t)row * a.kv_heads + (head - a.q_heads)) * a.dim;
  const uint hd = a.dim / 2u;
  const float x0 = x[tid], x1 = x[tid + hd];
  const float sum = simd_sum(fma(x0, x0, x1 * x1));
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < (hd + 31u) / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.dim) + a.eps);
  const device bfloat* w = W + (is_q ? a.q_norm : a.k_norm);
  const float y0 = x0 * inv * float(w[tid]), y1 = x1 * inv * float(w[tid + hd]);
  const uint pos = POS[row / a.block] + row % a.block;
  const float theta = float(pos) * exp2(-float(tid) / float(hd) * a.log_theta);
  const float c = precise::cos(theta), s = precise::sin(theta);
  x[tid] = y0 * c - y1 * s;
  x[tid + hd] = y0 * s + y1 * c;
}

// Block attention, unmasked: every block row of a slot attends the slot's context positions [lo, hi) and the block's
// own keys. One threadgroup a (split, kv head, slot) with a simdgroup per (row, head of the group); splits below the
// slot's `splits` walk `chunk` context positions, split `splits` the block, later splits are empty.
struct i2m_attn_slot { const device half* keys; const device half* values; uint lo, hi, splits, chunk; };
struct i2m_attn_args { uint q_heads, kv_heads, dim, block, max_splits; float scale; };

[[kernel]] void i2m_attn_split(const device float* Q [[buffer(0)]], const device float* BK [[buffer(1)]],
    const device float* BV [[buffer(2)]], constant i2m_attn_args& a [[buffer(3)]],
    constant i2m_attn_slot* slots [[buffer(4)]], device float* PART [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint3 nth3 [[threads_per_threadgroup]]) {
  const uint split = tg.x, kv = tg.y, slot = tg.z, nth = nth3.x;
  constant i2m_attn_slot& sl = slots[slot];
  const uint G = a.q_heads / a.kv_heads;
  const uint j = sg / G, g = sg % G;
  const uint h = kv * G + g;
  const uint row = slot * a.block + j;
  const uint D = a.dim, per = max(1u, D / 32u);
  const uint S = a.max_splits + 1u;
  const size_t base = ((((size_t)slot * a.q_heads + h) * a.block + j) * S + split) * (D + 2u);
  const bool block_split = split == sl.splits;
  const uint js = block_split ? 0u : sl.lo + split * sl.chunk;
  const uint je = block_split ? a.block : (split < sl.splits ? min(js + sl.chunk, sl.hi) : js);
  if (js >= je) {
    if (lane == 0u) {
      PART[base] = -INFINITY;
      PART[base + 1u] = 0.0f;
    }
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = 0.0f;
    return;
  }
  float qv[4];
  for (uint u = 0; u < per; ++u) qv[u] = (lane + 32u * u < D) ? Q[((size_t)row * a.q_heads + h) * D + lane + 32u * u] * a.scale : 0.0f;
  threadgroup float4 KB4[32];
  threadgroup float4 VB4[32];
  threadgroup float* KB = (threadgroup float*)KB4;
  threadgroup float* VB = (threadgroup float*)VB4;
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint p = js; p < je; ++p) {
    if (block_split) {
      const size_t at = (((size_t)slot * a.block + p) * a.kv_heads + kv) * D;
      for (uint v = tid; v < D; v += nth) {
        KB[v] = BK[at + v];
        VB[v] = BV[at + v];
      }
    } else {
      const size_t at = ((size_t)p * a.kv_heads + kv) * D;
      for (uint v = tid; v < D / 4u; v += nth) {
        KB4[v] = float4(*reinterpret_cast<const device packed_half4*>(sl.keys + at + v * 4u));
        VB4[v] = float4(*reinterpret_cast<const device packed_half4*>(sl.values + at + v * 4u));
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) dot = fma(qv[u], KB[lane + 32u * u], dot);
    const float score = simd_sum(dot);
    const float peak = max(top, score);
    const float rescale = precise::exp(top - peak);
    const float w = precise::exp(score - peak);
    mass = mass * rescale + w;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * rescale + w * VB[lane + 32u * u];
    top = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = acc[u];
}

// O[row][h]: the slot's splits merged. One simdgroup a (head, row).
[[kernel]] void i2m_attn_combine(const device float* PART [[buffer(0)]], constant i2m_attn_args& a [[buffer(1)]],
    device float* O [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, row = tg.y;
  const uint slot = row / a.block, j = row % a.block;
  const uint D = a.dim, S = a.max_splits + 1u, per = max(1u, D / 32u);
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint s = 0; s < S; ++s) {
    const size_t base = ((((size_t)slot * a.q_heads + h) * a.block + j) * S + s) * (D + 2u);
    const float ptop = PART[base];
    const float pmass = PART[base + 1u];
    if (pmass == 0.0f) continue;
    const float peak = max(top, ptop);
    const float r0 = precise::exp(top - peak);
    const float r1 = precise::exp(ptop - peak);
    mass = mass * r0 + pmass * r1;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * r0 + r1 * PART[base + 2u + lane + 32u * u];
    top = peak;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) O[((size_t)row * a.q_heads + h) * D + lane + 32u * u] = acc[u] / mass;
}

// G = silu(G) * U, elementwise.
struct i2m_count_args { uint count; };

[[kernel]] void i2m_silu_mul(device float* G [[buffer(0)]], const device float* U [[buffer(1)]],
    constant i2m_count_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float g = G[i];
  G[i] = g / (1.0f + precise::exp(-g)) * U[i];
}

// The recurrent projection's step-independent part: CTX[slot * steps + s] = rnn_proj[:, 2 rank:] applied to the step's
// draft row (block row s + 1) and the slot's mixed Markov tap, plus the bias. One simdgroup an output column.
struct i2m_rnn_ctx_args { uint hidden, rank, block, steps, rows; };

[[kernel]] void i2m_rnn_ctx(const device float* XN [[buffer(0)]], const device float* FRNN [[buffer(1)]],
    const device bfloat* W [[buffer(2)]], const device bfloat* BIAS [[buffer(3)]],
    constant i2m_rnn_ctx_args& a [[buffer(4)]], device float* CTX [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = tg.x * 8u + sg, r = tg.y;
  const uint outs = 3u * a.rank;
  if (col >= outs || r >= a.rows) return;
  const uint slot = r / a.steps, s = r % a.steps;
  const uint width = 2u * a.rank + 2u * a.hidden;
  const device bfloat* w = W + (size_t)col * width + 2u * a.rank;
  const device float* x = XN + ((size_t)slot * a.block + 1u + s) * a.hidden;
  const device float* f = FRNN + (size_t)slot * a.hidden;
  float acc = 0.0f;
  for (uint k = lane; k < a.hidden; k += 32u) acc = fma(x[k], float(w[k]), acc);
  for (uint k = lane; k < a.hidden; k += 32u) acc = fma(f[k], float(w[a.hidden + k]), acc);
  acc = simd_sum(acc);
  if (lane == 0u) CTX[(size_t)r * outs + col] = acc + float(BIAS[col]);
}

// One recurrent step for every slot: raw = rnn_proj[:, :2 rank] @ [state, markov_w1[prev]] + CTX[step]; the state
// keeps sigmoid(gate) of itself and takes the rest from tanh(candidate); the latent is tanh(output).
struct i2m_step_args { uint rank, width, steps, step; };

[[kernel]] void i2m_rnn_step(const device bfloat* W [[buffer(0)]], const device bfloat* W1 [[buffer(1)]],
    const device float* CTX [[buffer(2)]], const device uint* PREV [[buffer(3)]], device float* STATE [[buffer(4)]],
    device float* LATENT [[buffer(5)]], constant i2m_step_args& a [[buffer(6)]],
    uint slot [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float raw[768];
  threadgroup float input[512];
  const uint R = a.rank, nsg = nth / 32u;
  const device bfloat* emb = W1 + (size_t)PREV[slot] * R;
  for (uint k = tid; k < R; k += nth) {
    input[k] = STATE[(size_t)slot * R + k];
    input[R + k] = float(emb[k]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = sg; o < 3u * R; o += nsg) {
    const device bfloat* w = W + (size_t)o * a.width;
    float acc = 0.0f;
    for (uint k = lane; k < 2u * R; k += 32u) acc = fma(input[k], float(w[k]), acc);
    acc = simd_sum(acc);
    if (lane == 0u) raw[o] = acc + CTX[((size_t)slot * a.steps + a.step) * 3u * R + o];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = tid; k < R; k += nth) {
    const float gate = 1.0f / (1.0f + precise::exp(-raw[k]));
    const float cand = precise::tanh(raw[R + k]);
    STATE[(size_t)slot * R + k] = gate * input[k] + (1.0f - gate) * cand;
    LATENT[(size_t)slot * R + k] = precise::tanh(raw[2u * R + k]);
  }
}

// One step's draft logits, cap * tanh((base + fold_bias + fold @ latent) / cap), for every slot over 32 vocabulary
// rows a threadgroup; each threadgroup keeps its best (value, index) per slot, the lowest index on ties. With `dump`,
// LOGITS[slot][step][v] receives the logits too.
struct i2m_fold_args { uint vocab, rank, slots, block, steps, step, tiles, dump; float cap; };

[[kernel]] void i2m_fold(const device float* BASE [[buffer(0)]], const device bfloat* FW [[buffer(1)]],
    const device bfloat* FB [[buffer(2)]], const device float* LATENT [[buffer(3)]],
    constant i2m_fold_args& a [[buffer(4)]], device float2* PART [[buffer(5)]], device float* LOGITS [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float best_v[8][8];
  threadgroup uint best_i[8][8];
  float lat[8][8];
  for (uint s = 0; s < 8u; ++s) for (uint u = 0; u < 8u; ++u)
    lat[s][u] = s < a.slots ? LATENT[(size_t)s * a.rank + lane * 8u + u] : 0.0f;
  float bv[8];
  uint bi[8];
  for (uint s = 0; s < 8u; ++s) {
    bv[s] = -INFINITY;
    bi[s] = 0xffffffffu;
  }
  for (uint q = 0; q < 4u; ++q) {
    const uint v = tg * 32u + sg * 4u + q;
    if (v >= a.vocab) break;
    const device bfloat* w = FW + (size_t)v * a.rank + lane * 8u;
    float wv[8];
    for (uint u = 0; u < 8u; ++u) wv[u] = float(w[u]);
    const float fb = float(FB[v]);
    for (uint s = 0; s < a.slots; ++s) {
      float acc = 0.0f;
      for (uint u = 0; u < 8u; ++u) acc = fma(lat[s][u], wv[u], acc);
      acc = simd_sum(acc);
      const float x = BASE[((size_t)s * a.block + 1u + a.step) * a.vocab + v] + (fb + acc);
      const float y = a.cap > 0.0f ? a.cap * precise::tanh(x / a.cap) : x;
      if (a.dump != 0u && lane == 0u) LOGITS[((size_t)s * a.steps + a.step) * a.vocab + v] = y;
      if (y > bv[s]) {
        bv[s] = y;
        bi[s] = v;
      }
    }
  }
  if (lane == 0u) for (uint s = 0; s < a.slots; ++s) {
    best_v[s][sg] = bv[s];
    best_i[s][sg] = bi[s];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0u && lane < a.slots) {
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint g = 0; g < 8u; ++g) {
      const float x = best_v[lane][g];
      const uint xi = best_i[lane][g];
      if (x > v || (x == v && xi < i)) {
        v = x;
        i = xi;
      }
    }
    PART[(size_t)lane * a.tiles + tg] = float2(v, as_type<float>(i));
  }
}

// The slot's draft for the step: its best over every tile, the lowest index on ties; also the next step's previous
// token. One threadgroup a slot.
struct i2m_pick_args { uint tiles, steps, step; };

[[kernel]] void i2m_pick(const device float2* PART [[buffer(0)]], constant i2m_pick_args& a [[buffer(1)]],
    device uint* PREV [[buffer(2)]], device uint* DRAFT [[buffer(3)]],
    uint slot [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tv[32];
  threadgroup uint ti[32];
  float v = -INFINITY;
  uint i = 0xffffffffu;
  for (uint t = tid; t < a.tiles; t += nth) {
    const float2 p = PART[(size_t)slot * a.tiles + t];
    const uint pi = as_type<uint>(p.y);
    if (p.x > v || (p.x == v && pi < i)) {
      v = p.x;
      i = pi;
    }
  }
  for (uint off = 16u; off > 0u; off /= 2u) {
    const float ov = simd_shuffle_xor(v, off);
    const uint oi = simd_shuffle_xor(i, off);
    if (ov > v || (ov == v && oi < i)) {
      v = ov;
      i = oi;
    }
  }
  if (lane == 0u) {
    tv[sg] = v;
    ti[sg] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint g = 1; g < nth / 32u; ++g) {
      if (tv[g] > v || (tv[g] == v && ti[g] < i)) {
        v = tv[g];
        i = ti[g];
      }
    }
    if (i == 0xffffffffu) i = 0u;
    PREV[slot] = i;
    DRAFT[(size_t)slot * a.steps + a.step] = i;
  }
}

// STATE = 0 for every slot before a block's first step.
[[kernel]] void i2m_zero(device float* X [[buffer(0)]], constant i2m_count_args& a [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i < a.count) X[i] = 0.0f;
}

// The fold, once: FW[v][j] = sum_k U[v][k] * HO[k][j] and FB[v] = sum_k U[v][k] * HOB[k], rounded to bf16, over the
// target's unembedding U (affine codes with bf16 scales and biases when `bits`, else bf16 rows UF). Affine rows are
// dequantized to bf16 first, as both reference heads fold them. One threadgroup of `rank` threads covers 8 vocabulary
// rows, staging them 256 columns at a time.
struct i2m_build_args { uint vocab, hidden, rank, bits; };

inline float i2m_affine(const device uint* w, const device bfloat* s, const device bfloat* b, size_t row, uint k,
                        uint width, uint bits) {
  const uint per = 32u / bits;
  const uint code = (w[row * (width / per) + k / per] >> ((k % per) * bits)) & ((1u << bits) - 1u);
  const size_t group = row * (width / 64u) + k / 64u;
  return float(code) * float(s[group]) + float(b[group]);
}

[[kernel]] void i2m_fold_build(const device uint* UQ [[buffer(0)]], const device bfloat* US [[buffer(1)]],
    const device bfloat* UB [[buffer(2)]], const device bfloat* UF [[buffer(3)]], const device bfloat* HO [[buffer(4)]],
    const device bfloat* HOB [[buffer(5)]], constant i2m_build_args& a [[buffer(6)]], device bfloat* FW [[buffer(7)]],
    device bfloat* FB [[buffer(8)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float rows[8][256];
  threadgroup float part[8][8];
  float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  float accb[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
  const uint v0 = tg * 8u;
  for (uint k0 = 0; k0 < a.hidden; k0 += 256u) {
    for (uint e = tid; e < 8u * 256u; e += a.rank) {
      const uint r = e / 256u, k = k0 + e % 256u, v = v0 + r;
      float x = 0.0f;
      if (v < a.vocab && k < a.hidden) x = a.bits != 0u ? float(bfloat(i2m_affine(UQ, US, UB, v, k, a.hidden, a.bits))) : float(UF[(size_t)v * a.hidden + k]);
      rows[r][e % 256u] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint kn = min(256u, a.hidden - k0);
    for (uint k = 0; k < kn; ++k) {
      const float p = float(HO[(size_t)(k0 + k) * a.rank + tid]);
      for (uint r = 0; r < 8u; ++r) acc[r] = fma(rows[r][k], p, acc[r]);
    }
    for (uint k = tid; k < kn; k += a.rank) {
      const float hb = float(HOB[k0 + k]);
      for (uint r = 0; r < 8u; ++r) accb[r] = fma(rows[r][k], hb, accb[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < 8u; ++r) if (v0 + r < a.vocab) FW[(size_t)(v0 + r) * a.rank + tid] = bfloat(acc[r]);
  for (uint r = 0; r < 8u; ++r) {
    const float s = simd_sum(accb[r]);
    if (tid % 32u == 0u) part[r][tid / 32u] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 8u && v0 + tid < a.vocab) {
    float s = 0.0f;
    for (uint g = 0; g < a.rank / 32u; ++g) s += part[tid][g];
    FB[v0 + tid] = bfloat(s);
  }
}

// The proposals' base logits through the target's 8-bit affine unembedding, for few rows: BASE[s * block + 1 + j][v]
// for every slot s and step j < steps, from the final-normed block rows XN. Each simdgroup reads its 4 vocabulary rows
// once, coalesced, for every input row; the input rows are staged 256 columns at a time.
struct i2m_head_args { uint vocab, hidden, block, steps, rows; };

[[kernel]] void i2m_head(const device float* XN [[buffer(0)]], const device uint* W [[buffer(1)]],
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], constant i2m_head_args& a [[buffer(4)]],
    device float* BASE [[buffer(5)]], uint tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float X[16][256];
  float acc[4][16];
  for (uint q = 0; q < 4u; ++q) for (uint r = 0; r < 16u; ++r) acc[q][r] = 0.0f;
  const uint words = a.hidden / 4u;
  for (uint k0 = 0; k0 < a.hidden; k0 += 256u) {
    for (uint e = tid; e < a.rows * 256u; e += 256u) {
      const uint r = e / 256u, k = k0 + e % 256u;
      const uint row = (r / a.steps) * a.block + 1u + r % a.steps;
      X[r][e % 256u] = k < a.hidden ? XN[(size_t)row * a.hidden + k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k = k0 + lane * 8u;
    if (k < a.hidden) {
      for (uint q = 0; q < 4u; ++q) {
        const uint v = tg * 32u + sg * 4u + q;
        if (v >= a.vocab) break;
        const uint2 packed = *reinterpret_cast<const device uint2*>(W + (size_t)v * words + k / 4u);
        const size_t group = (size_t)v * (a.hidden / 64u) + k / 64u;
        const float scale = float(S[group]), bias = float(B[group]);
        float w[8];
        for (uint u = 0; u < 4u; ++u) {
          w[u] = float((packed.x >> (8u * u)) & 0xffu) * scale + bias;
          w[4u + u] = float((packed.y >> (8u * u)) & 0xffu) * scale + bias;
        }
        for (uint r = 0; r < 16u; ++r) {
          if (r >= a.rows) break;
          float sum = acc[q][r];
          for (uint u = 0; u < 8u; ++u) sum = fma(w[u], X[r][lane * 8u + u], sum);
          acc[q][r] = sum;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint q = 0; q < 4u; ++q) {
    const uint v = tg * 32u + sg * 4u + q;
    if (v >= a.vocab) break;
    for (uint r = 0; r < 16u; ++r) {
      if (r >= a.rows) break;
      const float total = simd_sum(acc[q][r]);
      const uint row = (r / a.steps) * a.block + 1u + r % a.steps;
      if (lane == 0u) BASE[(size_t)row * a.vocab + v] = total;
    }
  }
}
