#include <metal_stdlib>
using namespace metal;

struct i2_aot_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2_aot_expert_args { uint k, n, slots, topk; float gate_cap, up_cap; };
struct i2_aot_embed_args { uint width, rows; float eps; };
struct i2_aot_projection_args { uint k, full; uint m[8]; };
struct i2_aot_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };
struct i2_aot_batch_row { device half* keys; device half* values; i2_aot_attn_parts args; };
struct i2_aot_mix_args { uint width, depth, stride, k, norm; float scale, eps; };
struct i2_aot_controls_args { uint h, rank, rows; uint xs[3], xb[3]; uint msg0, cols; };
struct i2_aot_conv_row { device float* ring; uint pos, pad; };
struct i2_aot_conv_args { uint channels, heads, ring, pos, layout; float norm; uint residual, topk, push_slot; };

inline float i2_aot_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

inline float i2_aot_weight8_sb(const device uint* w, const device bfloat2* sb, size_t row, uint k, uint width) {
  const uint code = (w[row * (width / 4u) + k / 4u] >> ((k % 4u) * 8u)) & 255u;
  const bfloat2 pair = sb[row * (width / 64u) + k / 64u];
  return float(code) * float(pair.x) + float(pair.y);
}

inline float i2_aot_dot8_sb(const device float* x, const device uint* w, const device bfloat2* sb,
                            size_t row, uint width, uint lane) {
  const uint words = width / 4u;
  const uint groups = width / 64u;
  const uint4 shift = uint4(0u, 8u, 16u, 24u);
  const device uint* wr = w + row * words;
  const device bfloat2* sbr = sb + row * groups;
  float acc = 0.0f;
  uint word = lane;
  for (; word + 96u < words; word += 128u) {
    uint packed[4];
    bfloat2 pair[4];
    float4 input0[4];
#pragma unroll
    for (uint i = 0; i < 4u; ++i) {
      const uint at = word + 32u * i;
      packed[i] = wr[at];
      pair[i] = sbr[at >> 4u];
      input0[i] = float4(*reinterpret_cast<const device packed_float4*>(x + at * 4u));
    }
#pragma unroll
    for (uint i = 0; i < 4u; ++i) {
      const float4 code0 = float4((uint4(packed[i]) >> shift) & 255u);
      const float4 value0 = fma(code0, float4(float(pair[i].x)), float4(float(pair[i].y)));
      acc = fma(input0[i].x, value0.x, acc);
      acc = fma(input0[i].y, value0.y, acc);
      acc = fma(input0[i].z, value0.z, acc);
      acc = fma(input0[i].w, value0.w, acc);
    }
  }
  for (; word < words; word += 32u) {
    const uint packed = wr[word];
    const bfloat2 pair = sbr[word >> 4u];
    const float4 code0 = float4((uint4(packed) >> shift) & 255u);
    const float4 input0 = float4(*reinterpret_cast<const device packed_float4*>(x + word * 4u));
    const float4 value0 = fma(code0, float4(float(pair.x)), float4(float(pair.y)));
    acc = fma(input0.x, value0.x, acc);
    acc = fma(input0.y, value0.y, acc);
    acc = fma(input0.z, value0.z, acc);
    acc = fma(input0.w, value0.w, acc);
  }
  return simd_sum(acc);
}

inline float i2_aot_dot4_sb(const device float* x, const device uint* w, const device bfloat2* sb,
                            size_t row, uint width, uint lane) {
  const uint words = width / 8u;
  const uint groups = width / 64u;
  const uint4 shift = uint4(0u, 4u, 8u, 12u);
  const device uint* wr = w + row * words;
  const device bfloat2* sbr = sb + row * groups;
  float acc = 0.0f;
  uint word = lane;
  for (; word + 96u < words; word += 128u) {
    uint packed[4];
    bfloat2 pair[4];
    float4 input0[4], input1[4];
#pragma unroll
    for (uint i = 0; i < 4u; ++i) {
      const uint at = word + 32u * i;
      packed[i] = wr[at];
      pair[i] = sbr[at >> 3u];
      input0[i] = float4(*reinterpret_cast<const device packed_float4*>(x + at * 8u));
      input1[i] = float4(*reinterpret_cast<const device packed_float4*>(x + at * 8u + 4u));
    }
#pragma unroll
    for (uint i = 0; i < 4u; ++i) {
      const float sc = float(pair[i].x), bi = float(pair[i].y);
      const float4 code0 = float4((uint4(packed[i]) >> shift) & 15u);
      const float4 value0 = fma(code0, float4(sc), float4(bi));
      acc = fma(input0[i].x, value0.x, acc);
      acc = fma(input0[i].y, value0.y, acc);
      acc = fma(input0[i].z, value0.z, acc);
      acc = fma(input0[i].w, value0.w, acc);
      const float4 code1 = float4((uint4(packed[i] >> 16u) >> shift) & 15u);
      const float4 value1 = fma(code1, float4(sc), float4(bi));
      acc = fma(input1[i].x, value1.x, acc);
      acc = fma(input1[i].y, value1.y, acc);
      acc = fma(input1[i].z, value1.z, acc);
      acc = fma(input1[i].w, value1.w, acc);
    }
  }
  for (; word < words; word += 32u) {
    const uint packed = wr[word];
    const bfloat2 pair = sbr[word >> 3u];
    const float sc = float(pair.x), bi = float(pair.y);
    const float4 code0 = float4((uint4(packed) >> shift) & 15u);
    const float4 input0 = float4(*reinterpret_cast<const device packed_float4*>(x + word * 8u));
    const float4 value0 = fma(code0, float4(sc), float4(bi));
    acc = fma(input0.x, value0.x, acc);
    acc = fma(input0.y, value0.y, acc);
    acc = fma(input0.z, value0.z, acc);
    acc = fma(input0.w, value0.w, acc);
    const float4 code1 = float4((uint4(packed >> 16u) >> shift) & 15u);
    const float4 input1 = float4(*reinterpret_cast<const device packed_float4*>(x + word * 8u + 4u));
    const float4 value1 = fma(code1, float4(sc), float4(bi));
    acc = fma(input1.x, value1.x, acc);
    acc = fma(input1.y, value1.y, acc);
    acc = fma(input1.z, value1.z, acc);
    acc = fma(input1.w, value1.w, acc);
  }
  return simd_sum(acc);
}

[[kernel]] void i2_aot_embed_sb(const device uint* ids [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat2* sb [[buffer(2)]], const device bfloat* norm [[buffer(3)]],
    constant i2_aot_embed_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint id = ids[row];
  float sum = 0.0f;
  for (uint k = tid; k < a.width; k += nth) {
    const float v = i2_aot_weight8_sb(w, sb, id, k, a.width);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < nth / 32u; ++i) total += part[i];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint k = tid; k < a.width; k += nth)
    y[(size_t)row * a.width + k] = i2_aot_weight8_sb(w, sb, id, k, a.width) * inv * float(norm[k]);
}

[[kernel]] void i2_aot_dense8_sb(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat2* sb [[buffer(2)]], constant i2_aot_dense_args& a [[buffer(3)]],
    device float* y [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint row = group.y, lane = tid % 32u;
  if (col >= a.n || row >= a.rows) return;
  const device float* input = x + (size_t)(row * a.xs + a.xb) * a.k;
  const float acc = i2_aot_dot8_sb(input, w, sb, col, a.k, lane);
  if (lane == 0u) y[(size_t)row * a.n + col] = a.cap > 0.0f ? a.cap * precise::tanh(acc / a.cap) : acc;
}

[[kernel]] void i2_aot_batch_proj8_sb(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]], const device bfloat2* sbq [[buffer(4)]],
    const device uint* wk [[buffer(5)]], const device bfloat2* sbk [[buffer(6)]],
    const device uint* wv [[buffer(7)]], const device bfloat2* sbv [[buffer(8)]],
    device float* oq [[buffer(9)]], device float* ok [[buffer(10)]], device float* ov [[buffer(11)]],
    device float* orl [[buffer(12)]], device float* og [[buffer(13)]], device float* os [[buffer(14)]],
    device float* okd [[buffer(15)]], device float* ovd [[buffer(16)]],
    constant i2_aot_projection_args& a [[buffer(17)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint row = group.x * 8u + tid / 32u;
  const uint lane = tid % 32u;
  uint which = 0u;
  for (; which < 8u; ++which) {
    if (row < a.m[which]) break;
    row -= a.m[which];
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  const device float* x = (key ? xk : val ? xv : xq) + (size_t)group.y * a.k;
  const device uint* w = a.full && key ? wk : a.full && val ? wv : wq;
  const device bfloat2* sb = a.full && key ? sbk : a.full && val ? sbv : sbq;
  uint offset = 0u;
  if (which == 1u) offset = a.full ? 0u : a.m[0];
  if (which == 2u) offset = a.full ? 0u : a.m[0] + a.m[1];
  if (which == 3u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]);
  if (which == 4u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3];
  if (which == 5u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3] + a.m[4];
  if (which == 6u) offset = a.m[1];
  if (which == 7u) offset = a.m[2];
  device float* out = which == 0u ? oq : which == 1u ? ok : which == 2u ? ov : which == 3u ? orl
      : which == 4u ? og : which == 5u ? os : which == 6u ? okd : ovd;
  const float result = i2_aot_dot8_sb(x, w, sb, row + offset, a.k, lane);
  if (lane == 0u) out[(size_t)group.y * a.m[which] + row] = result;
}

[[kernel]] void i2_aot_expert_up4_sb(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat2* sb [[buffer(2)]], const device int* picks [[buffer(3)]],
    constant i2_aot_expert_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint slot = group.y, lane = tid % 32u;
  if (col >= a.n || slot >= a.slots) return;
  const size_t expert = uint(picks[slot]);
  const size_t base = expert * 2u * a.n + 2u * col;
  const device float* input = x + (size_t)(slot / a.topk) * a.k;
  const float gate = i2_aot_dot4_sb(input, w, sb, base, a.k, lane);
  const float up = i2_aot_dot4_sb(input, w, sb, base + 1u, a.k, lane);
  if (lane == 0u) {
    const float g = a.gate_cap * precise::tanh(gate / a.gate_cap) / (1.0f + precise::exp(-gate));
    const float u = a.up_cap > 0.0f ? a.up_cap * precise::tanh(up / a.up_cap) : up;
    y[(size_t)slot * a.n + col] = g * u;
  }
}

[[kernel]] void i2_aot_expert_down4_sb(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat2* sb [[buffer(2)]], const device int* picks [[buffer(3)]],
    constant i2_aot_expert_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint slot = group.y, lane = tid % 32u;
  if (col >= a.n || slot >= a.slots) return;
  const size_t row = (size_t)uint(picks[slot]) * a.n + col;
  const float acc = i2_aot_dot4_sb(x + (size_t)slot * a.k, w, sb, row, a.k, lane);
  if (lane == 0u) y[(size_t)slot * a.n + col] = acc;
}

[[kernel]] void i2_aot_combine_sig_mul(const device float* PART [[buffer(0)]],
    constant i2_aot_batch_row* rows [[buffer(1)]], const device float* GATE [[buffer(2)]],
    device float* O [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constant i2_aot_attn_parts& a = rows[tg.y].args;
  constexpr uint D = 128u;
  const uint h = tg.x, S = a.splits;
  const device float* part = PART + size_t(tg.y) * a.heads * 16u * (D + 2u);
  const device float* gate = GATE + size_t(tg.y) * a.heads * D;
  device float* out = O + size_t(tg.y) * a.heads * D;
  const size_t first = (size_t)h * S * (D + 2u);
  const size_t mine = first + (size_t)min(lane, S - 1u) * (D + 2u);
  const float tops = part[mine], masses = part[mine + 1u];
  float pv[16][4];
#pragma unroll
  for (uint s = 0; s < 16u; ++s) {
    const size_t base = first + (size_t)min(s, S - 1u) * (D + 2u) + 2u + lane;
    pv[s][0] = part[base];
    pv[s][1] = part[base + 32u];
    pv[s][2] = part[base + 64u];
    pv[s][3] = part[base + 96u];
  }
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
  for (uint s = 0; s < 16u; ++s) {
    if (s >= S) break;
    const float ptop = simd_shuffle(tops, ushort(s));
    const float pmass = simd_shuffle(masses, ushort(s));
    if (pmass == 0.0f) continue;
    const float peak = max(top, ptop);
    const float r0 = precise::exp(top - peak);
    const float r1 = precise::exp(ptop - peak);
    mass = mass * r0 + pmass * r1;
    for (uint u = 0; u < 4u; ++u) acc[u] = acc[u] * r0 + r1 * pv[s][u];
    top = peak;
  }
  const uint i0 = h * D + lane;
  out[i0] = (acc[0] / mass) / (1.0f + precise::exp(-gate[i0]));
  out[i0 + 32u] = (acc[1] / mass) / (1.0f + precise::exp(-gate[i0 + 32u]));
  out[i0 + 64u] = (acc[2] / mass) / (1.0f + precise::exp(-gate[i0 + 64u]));
  out[i0 + 96u] = (acc[3] / mass) / (1.0f + precise::exp(-gate[i0 + 96u]));
}

inline float i2_aot_e2m1(uint c) {
  const uint sign = (c & 8u) << 28u;
  const uint mag = c & 7u;
  const uint bits = mag >= 2u
      ? (sign | (((mag >> 1u) + 126u) << 23u) | ((mag & 1u) << 22u))
      : (sign | (mag != 0u ? (126u << 23u) : 0u));
  return as_type<float>(bits);
}

inline float i2_aot_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3u) & 15u;
  const uint m = uint(c) & 7u;
  if (e != 0u) {
    const uint bits = ((uint(c) & 128u) << 24u) | ((e + 120u) << 23u) | (m << 20u);
    return as_type<float>(bits);
  }
  const float v = float(m) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

[[kernel]] void i2_aot_gu_nv(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_expert_args& a [[buffer(5)]],
    device float* ACT [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lut[16];
  threadgroup float scales[256];
  if (tid < 16u) lut[tid] = E2M1[tid];
  scales[tid] = E4M3[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 8u + sg;
  const uint slot = tg.y;
  if (c >= a.n || slot >= a.slots) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = X + (size_t)(slot / a.topk) * a.k;
  const size_t stride = a.k / 2u;
  const size_t groups = a.k / 16u;
  const size_t row = (size_t)e * a.n * 2u + 2u * c;
  const device uchar* wg = W + row * stride;
  const device uchar* wu = wg + stride;
  const device uchar* sg_ptr = S + row * groups;
  const device uchar* su_ptr = sg_ptr + groups;
  float dg = 0.0f, du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * 16u;
    const float4 xv0 = *reinterpret_cast<const device float4*>(x + k0);
    const float4 xv1 = *reinterpret_cast<const device float4*>(x + k0 + 4u);
    const float4 xv2 = *reinterpret_cast<const device float4*>(x + k0 + 8u);
    const float4 xv3 = *reinterpret_cast<const device float4*>(x + k0 + 12u);
    const float xv[16] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w,
                           xv2.x, xv2.y, xv2.z, xv2.w, xv3.x, xv3.y, xv3.z, xv3.w };
    const ulong ag = *(const device ulong*)(wg + g * 8u);
    const ulong au = *(const device ulong*)(wu + g * 8u);
    float pg = 0.0f, pu = 0.0f;
#pragma unroll
    for (uint j = 0; j < 8u; ++j) {
      const uint sh = 8u * j;
      const uint ab = uint(ag >> sh) & 255u;
      const uint ub = uint(au >> sh) & 255u;
      const float x0 = xv[2u * j], x1 = xv[2u * j + 1u];
      pg = fma(lut[ab & 15u], x0, pg);
      pg = fma(lut[ab >> 4u], x1, pg);
      pu = fma(lut[ub & 15u], x0, pu);
      pu = fma(lut[ub >> 4u], x1, pu);
    }
    dg = fma(scales[sg_ptr[g]], pg, dg);
    du = fma(scales[su_ptr[g]], pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
    const float global = gscale[e];
    ACT[(size_t)slot * a.n + c] = i2_aot_situ(dg * global, du * global, a.gate_cap, a.up_cap);
  }
}

[[kernel]] void i2_aot_down1_nv(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_expert_args& a [[buffer(5)]],
    device float* OUT [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lut[16];
  threadgroup float scales[256];
  if (tid < 16u) lut[tid] = E2M1[tid];
  scales[tid] = E4M3[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  constexpr uint PER = 4u;
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  const uint e = uint(PICKS[slot]);
  const uint first = tg.x * 32u + sg * PER;
  const size_t stride = a.k / 2u;
  const size_t groups = a.k / 16u;
  const device float* x = ACT + (size_t)slot * a.k;
  const device uchar* We = W + (size_t)e * a.n * stride;
  const device uchar* Se = S + (size_t)e * a.n * groups;
  size_t ra[PER];
  float d[PER] = {};
#pragma unroll
  for (uint j = 0; j < PER; ++j) ra[j] = min(first + j, a.n - 1u);
  for (uint g = lane; g < groups; g += 32u) {
    const float4 q0 = *reinterpret_cast<const device float4*>(x + g * 16u);
    const float4 q1 = *reinterpret_cast<const device float4*>(x + g * 16u + 4u);
    const float4 q2 = *reinterpret_cast<const device float4*>(x + g * 16u + 8u);
    const float4 q3 = *reinterpret_cast<const device float4*>(x + g * 16u + 12u);
    const float xv[16] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w,
                           q2.x, q2.y, q2.z, q2.w, q3.x, q3.y, q3.z, q3.w };
#pragma unroll
    for (uint j = 0; j < PER; ++j) {
      const ulong word = *(const device ulong*)(We + ra[j] * stride + g * 8u);
      float p = 0.0f;
#pragma unroll
      for (uint i = 0; i < 8u; ++i) {
        const uint byte = uint(word >> (8u * i)) & 255u;
        p = fma(lut[byte >> 4u], xv[2u * i + 1u], fma(lut[byte & 15u], xv[2u * i], p));
      }
      d[j] = fma(scales[Se[ra[j] * groups + g]], p, d[j]);
    }
  }
  const float global = gscale[e];
#pragma unroll
  for (uint j = 0; j < PER; ++j) {
    if (first + j >= a.n) break;
    const float v = simd_sum(d[j]);
    if (lane == 0u) OUT[(size_t)slot * a.n + first + j] = v * global;
  }
}

inline void i2_aot_mix_row_inv(const device float* BLK, const device float* CUR, const device float* Q0,
    const device float* Q1, const device float* Q2, const device float* Q3, i2_aot_mix_args a,
    device float* MIX, device float* INV_OUT, uint inv_base, uint row, uint t, uint sg, uint lane,
    threadgroup float (&part)[4][5][8], threadgroup float (&w)[4][5], threadgroup float (&inv)[5]) {
  float src[5][9];
  for (uint j = 0; j < a.depth; ++j) {
    const device float* s = j + 1u < a.depth ? BLK + ((size_t)row * a.stride + (a.depth - 2u - j)) * a.width
                                             : CUR + (size_t)row * a.width;
    float sum = 0.0f;
    for (uint i = 0; i < 9u; ++i) {
      src[j][i] = s[t + i * 256u];
      sum = fma(src[j][i], src[j][i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) part[0][j][sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.depth) {
    float total = 0.0f;
    for (uint s = 0; s < 8u; ++s) total += part[0][t][s];
    inv[t] = 1.0f / sqrt(total / float(a.width) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < a.depth; ++j) for (uint i = 0; i < 9u; ++i) src[j][i] = src[j][i] * inv[j];
  for (uint k = 0; k < a.k; ++k) {
    const device float* q = k == 0u ? Q0 : k == 1u ? Q1 : k == 2u ? Q2 : Q3;
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = 0; i < 9u; ++i) sum = fma(src[j][i], q[t + i * 256u], sum);
      sum = simd_sum(sum);
      if (lane == 0u) part[k][j][sg] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.k) {
    for (uint j = 0; j < a.depth; ++j) {
      float total = 0.0f;
      for (uint s = 0; s < 8u; ++s) total += part[t][j][s];
      w[t][j] = total * a.scale;
    }
    float peak = w[t][0];
    for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[t][j]);
    float total = 0.0f;
    for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[t][j] - peak);
    for (uint j = 0; j < a.depth; ++j) w[t][j] = precise::exp(w[t][j] - peak) / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < a.k; ++k) {
    device float* mix = MIX + ((size_t)row * a.k + k) * a.width;
    float sum2 = 0.0f;
    for (uint i = 0; i < 9u; ++i) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], src[j][i], sum);
      mix[t + i * 256u] = sum;
      sum2 = fma(sum, sum, sum2);
    }
    sum2 = simd_sum(sum2);
    if (lane == 0u) part[0][k][sg] = sum2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.k) {
    float total = 0.0f;
    for (uint s = 0; s < 8u; ++s) total += part[0][t][s];
    INV_OUT[inv_base + t] = 1.0f / sqrt(total / float(a.width) + 1e-6f);
  }
}

[[kernel]] void i2_aot_mix2_inv(const device float* BLK0 [[buffer(0)]], const device float* CUR0 [[buffer(1)]],
    const device float* Q00 [[buffer(2)]], const device float* Q01 [[buffer(3)]], const device float* Q02 [[buffer(4)]],
    const device float* Q03 [[buffer(5)]], device float* MIX0 [[buffer(6)]], const device float* BLK1 [[buffer(7)]],
    const device float* CUR1 [[buffer(8)]], const device float* Q10 [[buffer(9)]], const device float* Q11 [[buffer(10)]],
    const device float* Q12 [[buffer(11)]], const device float* Q13 [[buffer(12)]], device float* MIX1 [[buffer(13)]],
    constant i2_aot_mix_args* args [[buffer(14)]], device float* INV_OUT [[buffer(15)]],
    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  threadgroup float inv[5];
  if (tg.y == 0u) i2_aot_mix_row_inv(BLK0, CUR0, Q00, Q01, Q02, Q03, args[0], MIX0, INV_OUT, 0u, tg.x, t, sg, lane, part, w, inv);
  else i2_aot_mix_row_inv(BLK1, CUR1, Q10, Q11, Q12, Q13, args[1], MIX1, INV_OUT, 3u, tg.x, t, sg, lane, part, w, inv);
}

[[kernel]] void i2_aot_mix_inv(const device float* BLK [[buffer(0)]], const device float* CUR [[buffer(1)]],
    const device float* Q0 [[buffer(2)]], constant i2_aot_mix_args& a [[buffer(3)]],
    device float* MIX [[buffer(4)]], device float* INV_OUT [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  threadgroup float inv[5];
  i2_aot_mix_row_inv(BLK, CUR, Q0, Q0, Q0, Q0, a, MIX, INV_OUT, 7u, row, t, sg, lane, part, w, inv);
}

[[kernel]] void i2_aot_ctl_down3(const device float* MIX [[buffer(0)]], const device float* MSG [[buffer(1)]],
    const device float* INV [[buffer(2)]], const device float* W0 [[buffer(3)]],
    const device float* W1 [[buffer(4)]], const device float* W2 [[buffer(5)]],
    constant i2_aot_controls_args& a [[buffer(6)]], device float* OUT [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 2u + tid / 32u, lane = tid % 32u;
  if (col >= a.rank) return;
  const uint j = group.z;
  const uint xb = a.xb[j];
  const device float* cur = MIX + size_t(xb) * a.h;
  const device float* msg = MSG + size_t(j) * a.h;
  const float inv_c = INV[xb];
  const float inv_m = INV[3u + j];
  const uint width = 2u * a.h;
  const device float* w = (j == 0u ? W0 : j == 1u ? W1 : W2) + size_t(col) * width;
  float acc = 0.0f;
  for (uint base = 0u; base + 768u <= a.h; base += 768u) {
    float4 xv[6], wv[6];
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      const uint k = base + 128u * i + lane * 4u;
      xv[i] = float4(*reinterpret_cast<const device packed_float4*>(cur + k)) * inv_c;
      wv[i] = float4(*reinterpret_cast<const device packed_float4*>(w + k));
    }
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      acc = fma(xv[i].x, wv[i].x, acc); acc = fma(xv[i].y, wv[i].y, acc);
      acc = fma(xv[i].z, wv[i].z, acc); acc = fma(xv[i].w, wv[i].w, acc);
    }
  }
  for (uint base = a.h; base + 768u <= width; base += 768u) {
    float4 xv[6], wv[6];
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      const uint k = base + 128u * i + lane * 4u;
      xv[i] = float4(*reinterpret_cast<const device packed_float4*>(msg + (k - a.h))) * inv_m;
      wv[i] = float4(*reinterpret_cast<const device packed_float4*>(w + k));
    }
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      acc = fma(xv[i].x, wv[i].x, acc); acc = fma(xv[i].y, wv[i].y, acc);
      acc = fma(xv[i].z, wv[i].z, acc); acc = fma(xv[i].w, wv[i].w, acc);
    }
  }
  acc = simd_sum(acc);
  if (lane == 0u) OUT[size_t(j) * a.rank + col] = acc;
}

[[kernel]] void i2_aot_ctl_down1(const device float* MIX [[buffer(0)]], const device float* MSG [[buffer(1)]],
    const device float* INV [[buffer(2)]], const device float* W [[buffer(3)]],
    constant i2_aot_dense_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group * 2u + tid / 32u, lane = tid % 32u;
  if (col >= a.n) return;
  const uint h = a.k / 2u;
  const device float* cur = MIX;
  const device float* msg = MSG + size_t(3u) * h;
  const float inv_c = INV[7u];
  const float inv_m = INV[6u];
  const device float* weight = W + size_t(col) * a.k;
  float acc = 0.0f;
  for (uint base = 0u; base + 768u <= h; base += 768u) {
    float4 xv[6], wv[6];
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      const uint k = base + 128u * i + lane * 4u;
      xv[i] = float4(*reinterpret_cast<const device packed_float4*>(cur + k)) * inv_c;
      wv[i] = float4(*reinterpret_cast<const device packed_float4*>(weight + k));
    }
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      acc = fma(xv[i].x, wv[i].x, acc); acc = fma(xv[i].y, wv[i].y, acc);
      acc = fma(xv[i].z, wv[i].z, acc); acc = fma(xv[i].w, wv[i].w, acc);
    }
  }
  for (uint base = h; base + 768u <= a.k; base += 768u) {
    float4 xv[6], wv[6];
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      const uint k = base + 128u * i + lane * 4u;
      xv[i] = float4(*reinterpret_cast<const device packed_float4*>(msg + (k - h))) * inv_m;
      wv[i] = float4(*reinterpret_cast<const device packed_float4*>(weight + k));
    }
#pragma unroll
    for (uint i = 0; i < 6u; ++i) {
      acc = fma(xv[i].x, wv[i].x, acc); acc = fma(xv[i].y, wv[i].y, acc);
      acc = fma(xv[i].z, wv[i].z, acc); acc = fma(xv[i].w, wv[i].w, acc);
    }
  }
  acc = simd_sum(acc);
  if (lane == 0u) OUT[col] = acc;
}

[[kernel]] void i2_aot_ctl_apply3_gf2(const device float* X [[buffer(0)]], const device float2* GF0 [[buffer(1)]],
    const device float2* GF1 [[buffer(2)]], const device float2* GF2 [[buffer(3)]], const device float* C0 [[buffer(4)]],
    const device float* C1 [[buffer(5)]], const device float* C2 [[buffer(6)]], const device float* MSG [[buffer(7)]],
    device float* O0 [[buffer(8)]], device float* O1 [[buffer(9)]], device float* O2 [[buffer(10)]],
    constant i2_aot_controls_args& a [[buffer(11)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint lane = tid % 32u, j = group.z;
  if (col >= a.h) return;
  const device float* x = X + size_t(j) * 64u;
  const device float2* gf = (j == 0u ? GF0 : j == 1u ? GF1 : GF2) + size_t(col) * 64u;
  const device float* cur = (j == 0u ? C0 : j == 1u ? C1 : C2) + size_t(a.xb[j]) * a.h;
  const device float* msg = MSG + size_t(j) * a.h;
  device float* out = j == 0u ? O0 : j == 1u ? O1 : O2;
  const float x0 = x[lane], x1 = x[lane + 32u];
  const float2 gf0 = gf[lane], gf1 = gf[lane + 32u];
  const float g = simd_sum(fma(x1, gf1.x, fma(x0, gf0.x, 0.0f)));
  const float f = simd_sum(fma(x1, gf1.y, fma(x0, gf0.y, 0.0f)));
  if (lane == 0u) {
    const float cv = cur[col];
    out[col] = cv - precise::tanh(f) * cv + precise::tanh(g) * msg[col];
  }
}

struct i2_aot_batch_ctl_args { uint k, m0, m1, h, cxs, cxb, mxs, mxb, ys, yb; };

[[kernel]] void i2_aot_ctl_apply1_gf2(const device float* X [[buffer(0)]], const device float2* GF [[buffer(1)]],
    const device float* CUR [[buffer(2)]], const device float* MSG [[buffer(3)]], device float* OUT [[buffer(4)]],
    constant i2_aot_batch_ctl_args& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group * 8u + tid / 32u;
  const uint lane = tid % 32u;
  if (col >= a.m0) return;
  const device float2* gf = GF + size_t(col) * 64u;
  const device float* cur = CUR + size_t(a.cxb) * a.h;
  const device float* msg = MSG + size_t(a.mxb) * a.h;
  device float* out = OUT + size_t(a.yb) * a.h;
  const float x0 = X[lane], x1 = X[lane + 32u];
  const float2 gf0 = gf[lane], gf1 = gf[lane + 32u];
  const float g = simd_sum(fma(x1, gf1.x, fma(x0, gf0.x, 0.0f)));
  const float f = simd_sum(fma(x1, gf1.y, fma(x0, gf0.y, 0.0f)));
  if (lane == 0u) {
    const float cv = cur[col];
    out[col] = cv - precise::tanh(f) * cv + precise::tanh(g) * msg[col];
  }
}

[[kernel]] void i2_aot_gu_nv_packed(const device float* X [[buffer(0)]], const device ulong2* W [[buffer(1)]],
    const device uchar2* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_expert_args& a [[buffer(5)]],
    device float* ACT [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lut[16];
  threadgroup float scales[256];
  if (tid < 16u) lut[tid] = E2M1[tid];
  scales[tid] = E4M3[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 8u + sg;
  const uint slot = tg.y;
  if (c >= a.n || slot >= a.slots) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = X + (size_t)(slot / a.topk) * a.k;
  const size_t groups = a.k / 16u;
  const size_t base = ((size_t)e * a.n + c) * groups;
  const device ulong2* wgu = W + base;
  const device uchar2* sgu = S + base;
  float dg = 0.0f, du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * 16u;
    const ulong2 agu = wgu[g];
    const uchar2 scu = sgu[g];
    const float4 xv0 = *reinterpret_cast<const device float4*>(x + k0);
    const float4 xv1 = *reinterpret_cast<const device float4*>(x + k0 + 4u);
    const float4 xv2 = *reinterpret_cast<const device float4*>(x + k0 + 8u);
    const float4 xv3 = *reinterpret_cast<const device float4*>(x + k0 + 12u);
    const float xv[16] = { xv0.x, xv0.y, xv0.z, xv0.w, xv1.x, xv1.y, xv1.z, xv1.w,
                           xv2.x, xv2.y, xv2.z, xv2.w, xv3.x, xv3.y, xv3.z, xv3.w };
    const ulong ag = agu.x, au = agu.y;
    float pg = 0.0f, pu = 0.0f;
#pragma unroll
    for (uint j = 0; j < 8u; ++j) {
      const uint sh = 8u * j;
      const uint ab = uint(ag >> sh) & 255u;
      const uint ub = uint(au >> sh) & 255u;
      const float x0 = xv[2u * j], x1 = xv[2u * j + 1u];
      pg = fma(lut[ab & 15u], x0, pg);
      pg = fma(lut[ab >> 4u], x1, pg);
      pu = fma(lut[ub & 15u], x0, pu);
      pu = fma(lut[ub >> 4u], x1, pu);
    }
    dg = fma(scales[scu.x], pg, dg);
    du = fma(scales[scu.y], pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
    const float global = gscale[e];
    ACT[(size_t)slot * a.n + c] = i2_aot_situ(dg * global, du * global, a.gate_cap, a.up_cap);
  }
}

struct i2_aot_argmax_args { uint vocab, step_idx, stop_token, write_loop; };
struct i2_aot_loop_ctrl { uint cancelled, stopped, produced, pad; };

[[kernel]] void i2_aot_argmax_p1(const device float* logits [[buffer(0)]],
    constant i2_aot_argmax_args& a [[buffer(1)]], device float* out_v [[buffer(2)]],
    device uint* out_i [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[8];
  threadgroup uint si[8];
  float bv = -INFINITY;
  uint bi = 0xffffffffu;
  for (uint i = tg * 256u + tid; i < a.vocab; i += 128u * 256u) {
    const float v = logits[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(bv, off);
    const uint oi = simd_shuffle_down(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0u) { sv[sg] = bv; si[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fv = sv[0];
    uint fi = si[0];
    for (uint s = 1u; s < 8u; ++s) {
      if (sv[s] > fv || (sv[s] == fv && si[s] < fi)) { fv = sv[s]; fi = si[s]; }
    }
    out_v[tg] = fv;
    out_i[tg] = fi;
  }
}

[[kernel]] void i2_aot_argmax_p2(const device float* in_v [[buffer(0)]],
    const device uint* in_i [[buffer(1)]], constant i2_aot_argmax_args& a [[buffer(2)]],
    device uint* ids [[buffer(3)]], device uint* ring [[buffer(4)]],
    device i2_aot_loop_ctrl* ctrl [[buffer(5)]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sv[4];
  threadgroup uint si[4];
  float bv = in_v[tid];
  uint bi = in_i[tid];
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_down(bv, off);
    const uint oi = simd_shuffle_down(bi, off);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0u) { sv[sg] = bv; si[sg] = bi; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    float fv = sv[0];
    uint fi = si[0];
    for (uint s = 1u; s < 4u; ++s) {
      if (sv[s] > fv || (sv[s] == fv && si[s] < fi)) { fv = sv[s]; fi = si[s]; }
    }
    if (a.write_loop == 0u) {
      ring[0] = fi;
    } else if (ctrl->stopped == 0u) {
      if (ctrl->cancelled != 0u) {
        ctrl->stopped = 1u;
      } else {
        ids[0] = fi;
        ring[a.step_idx] = fi;
        ctrl->produced = a.step_idx + 1u;
        if (fi == a.stop_token) ctrl->stopped = 1u;
      }
    }
  }
}

inline float i2_aot_rows_sum(thread float (&p)[32], uint lane) {
  float a1[16], a2[8], a3[4], a4[2];
  const bool b0 = (lane & 1u) != 0u, b1 = (lane & 2u) != 0u, b2 = (lane & 4u) != 0u, b3 = (lane & 8u) != 0u;
  const bool b4 = (lane & 16u) != 0u;
#pragma unroll
  for (uint m = 0; m < 16u; ++m) {
    const float lo = p[2u * m], hi = p[2u * m + 1u];
    a1[m] = (b0 ? hi : lo) + simd_shuffle_xor(b0 ? lo : hi, ushort(1));
  }
#pragma unroll
  for (uint m = 0; m < 8u; ++m) {
    const float lo = a1[2u * m], hi = a1[2u * m + 1u];
    a2[m] = (b1 ? hi : lo) + simd_shuffle_xor(b1 ? lo : hi, ushort(2));
  }
#pragma unroll
  for (uint m = 0; m < 4u; ++m) {
    const float lo = a2[2u * m], hi = a2[2u * m + 1u];
    a3[m] = (b2 ? hi : lo) + simd_shuffle_xor(b2 ? lo : hi, ushort(4));
  }
#pragma unroll
  for (uint m = 0; m < 2u; ++m) {
    const float lo = a3[2u * m], hi = a3[2u * m + 1u];
    a4[m] = (b3 ? hi : lo) + simd_shuffle_xor(b3 ? lo : hi, ushort(8));
  }
  return (b4 ? a4[1] : a4[0]) + simd_shuffle_xor(b4 ? a4[0] : a4[1], ushort(16));
}

struct i2_aot_qkv_args {
  uint kv, heads, dim, rel, extent, layout, dyn, tau, kv_stride_rows, kv_row;
  float norm, q_threshold, rel_threshold, eps, floor, alpha;
};
struct i2_aot_qkv_row { device float* k_ring; device float* v_ring; device half* keys; device half* values; uint pos, pad; };

inline float i2_aot_qkv_conv(const thread float* hist, uint pos, const device bfloat* W, constant i2_aot_qkv_args& a, uint c,
    const device float* D) {
  const uint channels = a.kv * a.dim;
  float sum = hist[0];
  for (uint k = 0; k < 4u; ++k) {
    const uint lag = 3u - k;
    if (lag > pos) continue;
    sum = fma(hist[lag], float(W[a.layout ? (size_t)(3u - k) * channels + c : (size_t)c * 4u + k]), sum);
  }
  if (a.dyn) for (uint lag = 0; lag < 4u; ++lag) if (lag <= pos) sum = fma(hist[lag], D[(c / a.dim) * 4u + lag], sum);
  return sum * a.norm;
}

inline float i2_aot_qkv_total(float sum, threadgroup float* part, uint tid) {
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < 8u; ++i) total += part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

[[kernel]] void i2_aot_qkv_perm(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    const device bfloat* KW [[buffer(2)]], const device bfloat* VW [[buffer(3)]], const device float* KD [[buffer(4)]],
    const device float* VD [[buffer(5)]], device float* Q [[buffer(6)]], const device bfloat* QS [[buffer(7)]],
    const device float* R [[buffer(8)]], const device bfloat* RS [[buffer(9)]], const device bfloat* P [[buffer(10)]],
    constant i2_aot_qkv_args& a [[buffer(11)]], device float* REL [[buffer(12)]],
    constant i2_aot_qkv_row* rows [[buffer(13)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  threadgroup float rc[2][16];
  const uint row = group.y, pos = rows[row].pos;
  if (group.x >= a.kv + a.heads) {
    const uint block = group.x - a.kv - a.heads;
    const uint cols = a.extent + 1u, first = block * 256u, h0 = first / cols;
    if (tid < 64u) {
      const uint s = tid / 32u, lane = tid % 32u, h = h0 + s;
      const bool live = h < a.heads && lane < a.rel;
      const float v = live ? R[((size_t)row * a.heads + h) * a.rel + lane] * float(RS[(size_t)h * a.rel + lane]) : 0.0f;
      const float total = simd_sum(live ? fma(v, v, 0.0f) : 0.0f);
      const float clip = clamp(a.rel_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
      if (lane < 16u) rc[s][lane] = v * clip;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = first + tid;
    if (i >= cols * a.heads) return;
    const uint h = i / cols, dist = i % cols;
    const size_t out = (size_t)row * cols * a.heads + i;
    if (dist == a.extent) { REL[out] = 0.0f; return; }
    const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(pos) + 1.0f) / a.floor)) : 1.0f;
    float sum = 0.0f;
    for (uint u = 0; u < a.rel; ++u) sum = fma(float(P[u * a.extent + dist]), rc[h - h0][u], sum);
    REL[out] = sum * tau;
    return;
  }
  if (group.x >= a.kv) {
    const uint h = group.x - a.kv;
    device float* x = Q + ((size_t)row * a.heads + h) * a.dim;
    const bool live = tid < a.dim;
    const float v = live ? x[tid] * float(QS[(size_t)h * a.dim + tid]) : 0.0f;
    const float total = i2_aot_qkv_total(live ? fma(v, v, 0.0f) : 0.0f, part, tid);
    const float clip = clamp(a.q_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
    if (live) x[tid] = v * clip;
    return;
  }
  const uint count = a.kv * a.dim, c = group.x * a.dim + tid;
  const bool live = tid < a.dim;
  float kc = 0.0f, vc = 0.0f;
  if (live) {
    float kh[4], vh[4];
    const size_t at = (size_t)row * count + c;
    kh[0] = K[at];
    vh[0] = V[at];
    for (uint lag = 1; lag < 4u; ++lag) {
      const size_t r = lag <= pos ? ((pos - lag) % 4u) * count + c : 0;
      kh[lag] = lag <= pos ? rows[row].k_ring[r] : 0.0f;
      vh[lag] = lag <= pos ? rows[row].v_ring[r] : 0.0f;
    }
    kc = i2_aot_qkv_conv(kh, pos, KW, a, c, KD + (size_t)row * a.kv * 4u);
    vc = i2_aot_qkv_conv(vh, pos, VW, a, c, VD + (size_t)row * a.kv * 4u);
    rows[row].k_ring[(pos % 4u) * count + c] = kh[0];
    rows[row].v_ring[(pos % 4u) * count + c] = vh[0];
  }
  const float kt = i2_aot_qkv_total(live ? fma(kc, kc, 0.0f) : 0.0f, part, tid);
  const float clip = clamp(a.q_threshold / (sqrt(kt) + 1e-6f), 0.0f, 1.0f);
  const float vt = i2_aot_qkv_total(live ? fma(vc, vc, 0.0f) : 0.0f, part, tid);
  if (!live) return;
  const size_t dst = ((size_t)group.x * a.kv_stride_rows + a.kv_row) * a.dim + (tid & 31u) * 4u + (tid >> 5u);
  rows[row].keys[dst] = half(kc * clip);
  rows[row].values[dst] = half(vc * (1.0f / sqrt(vt / float(a.dim) + a.eps)));
}

struct i2_aot_split_perm_row { device half* keys; device half* values; i2_aot_attn_parts args; uint kv_stride_rows, pad; };

[[kernel]] void i2_aot_split_perm(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2_aot_split_perm_row* rows [[buffer(3)]],
    device float* PART [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128u, G = 3u, N = 96u, R = 32u, CELLS = (R * 32u + N - 1u) / N;
  threadgroup half4 KS4[32 * 32];
  threadgroup half4 VS4[32 * 32];
  constant i2_aot_attn_parts& a = rows[tg.y].args;
  if (tg.x >= a.kv * a.splits) return;
  const uint kv = tg.x / a.splits, s = tg.x % a.splits, h = kv * G + sg;
  const size_t head_base = (size_t)kv * rows[tg.y].kv_stride_rows * D;
  const device half* KEYS = rows[tg.y].keys + head_base;
  const device half* VALUES = rows[tg.y].values + head_base;
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  float qv[4];
  for (uint j = 0; j < 4u; ++j) qv[j] = Q[h * D + lane + 32u * j] * a.scale * tau;
  const float sk = sink[h];
  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
  const size_t base = ((size_t)h * a.splits + s) * (D + 2u);
  const uint js = lo + s * a.chunk, je = min(js + a.chunk, a.pos + 1u);
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint j0 = js; j0 < je; j0 += R) {
    const uint n = min(R, je - j0);
    const uint first = a.window ? j0 % a.window : j0;
    half4 kc[CELLS], vc[CELLS];
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      const uint r = min(e >> 5u, n - 1u), j = first + r;
      const uint rj = a.window && j >= a.window ? j - a.window : j;
      const size_t at = (size_t)rj * D + (e & 31u) * 4u;
      kc[i] = *reinterpret_cast<const device half4*>(KEYS + at);
      vc[i] = *reinterpret_cast<const device half4*>(VALUES + at);
    }
    float own = 0.0f;
    if (lane < n) own = B[h * (a.extent + 1u) + min(a.pos - (j0 + lane), a.extent)] - sk;
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      if (e < R * 32u) {
        KS4[e] = kc[i];
        VS4[e] = vc[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p[32];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float4 kk = float4(KS4[r * 32u + lane]);
      float dot = 0.0f;
      dot = fma(qv[0], kk.x, dot);
      dot = fma(qv[1], kk.y, dot);
      dot = fma(qv[2], kk.z, dot);
      dot = fma(qv[3], kk.w, dot);
      p[r] = dot;
    }
    const float score = i2_aot_rows_sum(p, lane) + own;
    float run = lane < n ? score : -INFINITY;
    for (ushort off = 1; off < 32; off <<= 1) {
      const float other = simd_shuffle_up(run, off);
      if (lane >= off) run = max(other, run);
    }
    const float peak = max(top, run);
    const float before = lane == 0u ? top : simd_shuffle_up(peak, ushort(1));
    const float rescales = precise::exp(before - peak);
    const float weights = precise::exp(score - peak);
    if (n == R) {
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        const float rescale = simd_shuffle(rescales, ushort(r));
        const float w = simd_shuffle(weights, ushort(r));
        mass = mass * rescale + w;
        const float4 vv = float4(VS4[r * 32u + lane]);
        acc[0] = acc[0] * rescale + w * vv.x;
        acc[1] = acc[1] * rescale + w * vv.y;
        acc[2] = acc[2] * rescale + w * vv.z;
        acc[3] = acc[3] * rescale + w * vv.w;
      }
    } else {
      for (uint r = 0; r < n; ++r) {
        const float rescale = simd_shuffle(rescales, ushort(r));
        const float w = simd_shuffle(weights, ushort(r));
        mass = mass * rescale + w;
        const float4 vv = float4(VS4[r * 32u + lane]);
        acc[0] = acc[0] * rescale + w * vv.x;
        acc[1] = acc[1] * rescale + w * vv.y;
        acc[2] = acc[2] * rescale + w * vv.z;
        acc[3] = acc[3] * rescale + w * vv.w;
      }
    }
    top = simd_shuffle(peak, ushort(n - 1u));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  for (uint u = 0; u < 4u; ++u) PART[base + 2u + lane + 32u * u] = acc[u];
}
