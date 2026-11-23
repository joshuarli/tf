#include <metal_stdlib>
using namespace metal;

struct i2a_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2a_expert_args { uint k, n, slots, topk; float gate_cap, up_cap; };
struct i2a_embed_args { uint width, rows; float eps; };

inline float i2a_weight(const device uint* w, const device bfloat* s, const device bfloat* b,
                        size_t row, uint k, uint width, uint bits) {
  const uint per = 32u / bits;
  const uint code = (w[row * (width / per) + k / per] >> ((k % per) * bits)) & ((1u << bits) - 1u);
  const size_t group = row * (width / 64u) + k / 64u;
  return float(code) * float(s[group]) + float(b[group]);
}

inline float i2a_dot(const device float* x, const device uint* w, const device bfloat* s,
                     const device bfloat* b, size_t row, uint width, uint bits, uint lane) {
  const uint per = 32u / bits;
  const uint mask = (1u << bits) - 1u;
  float acc = 0.0f;
  for (uint word = lane; word < width / per; word += 32u) {
    const uint packed = w[row * (width / per) + word];
    const size_t group = row * (width / 64u) + (word * per) / 64u;
    const float scale = float(s[group]), bias = float(b[group]);
    const uint4 shift = uint4(0u, bits, bits * 2u, bits * 3u);
    const float4 code0 = float4((uint4(packed) >> shift) & mask);
    const float4 input0 = float4(*reinterpret_cast<const device packed_float4*>(x + word * per));
    const float4 value0 = fma(code0, float4(scale), float4(bias));
    acc = fma(input0.x, value0.x, acc);
    acc = fma(input0.y, value0.y, acc);
    acc = fma(input0.z, value0.z, acc);
    acc = fma(input0.w, value0.w, acc);
    if (bits == 4u) {
      const float4 code1 = float4((uint4(packed >> 16u) >> shift) & mask);
      const float4 input1 = float4(*reinterpret_cast<const device packed_float4*>(x + word * per + 4u));
      const float4 value1 = fma(code1, float4(scale), float4(bias));
      acc = fma(input1.x, value1.x, acc);
      acc = fma(input1.y, value1.y, acc);
      acc = fma(input1.z, value1.z, acc);
      acc = fma(input1.w, value1.w, acc);
    }
  }
  return simd_sum(acc);
}

[[kernel]] void i2a_embed(const device uint* ids [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device bfloat* norm [[buffer(4)]], constant i2a_embed_args& a [[buffer(5)]],
    device float* y [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint id = ids[row];
  float sum = 0.0f;
  for (uint k = tid; k < a.width; k += nth) {
    const float v = i2a_weight(w, s, b, id, k, a.width, 8u);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < nth / 32u; ++i) total += part[i];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint k = tid; k < a.width; k += nth)
    y[(size_t)row * a.width + k] = i2a_weight(w, s, b, id, k, a.width, 8u) * inv * float(norm[k]);
}

[[kernel]] void i2a_float(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
    constant i2a_dense_args& a [[buffer(2)]], device float* y [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint row = group.y, lane = tid % 32u;
  if (col >= a.n || row >= a.rows) return;
  const device float* input = x + (size_t)(row * a.xs + a.xb) * a.k;
  const device float* weight = w + (size_t)col * a.k;
  float acc = 0.0f;
  uint base = 0u;
  for (; base + 128u <= a.k; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(input + k));
    const float4 wv = float4(*reinterpret_cast<const device packed_float4*>(weight + k));
    acc = fma(xv.x, wv.x, acc);
    acc = fma(xv.y, wv.y, acc);
    acc = fma(xv.z, wv.z, acc);
    acc = fma(xv.w, wv.w, acc);
  }
  for (uint k = base + lane; k < a.k; k += 32u) acc = fma(input[k], weight[k], acc);
  acc = simd_sum(acc);
  if (lane == 0u) y[(size_t)row * a.n + col] = acc;
}

[[kernel]] void i2a_dense(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2a_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint row = group.y;
  const uint lane = tid % 32u;
  if (col >= a.n || row >= a.rows) return;
  const device float* input = x + (size_t)(row * a.xs + a.xb) * a.k;
  const float acc = i2a_dot(input, w, s, b, col, a.k, a.bits, lane);
  if (lane == 0u) y[(size_t)row * a.n + col] = a.cap > 0.0f ? a.cap * precise::tanh(acc / a.cap) : acc;
}

[[kernel]] void i2a_dense_rows(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2a_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint bm = 32u, bn = 64u, bk = 32u, ap = 40u, bp = 72u;
  const uint sg = tid / 32u, m0 = group.y * bm, n0 = group.x * bn;
  threadgroup float inputs[bm * ap], weights[bk * bp], output[bm * bp];
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += bk) {
    for (uint i = tid; i < bm * bk; i += 256u) {
      const uint row = i / bk, k = i % bk;
      inputs[row * ap + k] = m0 + row < a.rows && k0 + k < a.k
        ? x[(size_t)(a.xb + (m0 + row) * a.xs) * a.k + k0 + k] : 0.0f;
    }
    for (uint i = tid; i < bk * bn; i += 256u) {
      const uint k = i / bn, col = i % bn;
      weights[k * bp + col] = n0 + col < a.n && k0 + k < a.k
        ? i2a_weight(w, s, b, n0 + col, k0 + k, a.k, a.bits) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < bk; k += 8u) {
      simdgroup_matrix<float, 8, 8> rhs;
      simdgroup_load(rhs, weights + k * bp + sg * 8u, ulong(bp));
      for (uint r = 0; r < 4u; ++r) {
        simdgroup_matrix<float, 8, 8> lhs;
        simdgroup_load(lhs, inputs + r * 8u * ap + k, ulong(ap));
        simdgroup_multiply_accumulate(acc[r], lhs, rhs, acc[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < 4u; ++r) simdgroup_store(acc[r], output + r * 8u * bp + sg * 8u, ulong(bp));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < bm * bn; i += 256u) {
    const uint row = i / bn, col = i % bn;
    if (m0 + row < a.rows && n0 + col < a.n) {
      const float value = output[row * bp + col];
      y[(size_t)(m0 + row) * a.n + n0 + col] = a.cap > 0.0f ? a.cap * precise::tanh(value / a.cap) : value;
    }
  }
}

[[kernel]] void i2a_expert_up(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device int* picks [[buffer(4)]], constant i2a_expert_args& a [[buffer(5)]],
    device float* y [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint slot = group.y, lane = tid % 32u;
  if (col >= a.n || slot >= a.slots) return;
  const size_t expert = uint(picks[slot]);
  const size_t base = expert * 2u * a.n + 2u * col;
  const device float* input = x + (size_t)(slot / a.topk) * a.k;
  const float gate = i2a_dot(input, w, s, b, base, a.k, 4u, lane);
  const float up = i2a_dot(input, w, s, b, base + 1u, a.k, 4u, lane);
  if (lane == 0u) {
    const float g = a.gate_cap * precise::tanh(gate / a.gate_cap) / (1.0f + precise::exp(-gate));
    const float u = a.up_cap > 0.0f ? a.up_cap * precise::tanh(up / a.up_cap) : up;
    y[(size_t)slot * a.n + col] = g * u;
  }
}

[[kernel]] void i2a_expert_down(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device int* picks [[buffer(4)]], constant i2a_expert_args& a [[buffer(5)]],
    device float* y [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint slot = group.y, lane = tid % 32u;
  if (col >= a.n || slot >= a.slots) return;
  const size_t row = (size_t)uint(picks[slot]) * a.n + col;
  const float acc = i2a_dot(x + (size_t)slot * a.k, w, s, b, row, a.k, 4u, lane);
  if (lane == 0u) y[(size_t)slot * a.n + col] = acc;
}

struct i2a_group_args { uint k, n, slots, topk, experts; float gate_cap, up_cap; };

[[kernel]] void i2a_group_up(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u, expert = group.y;
  const uint begin = offsets[expert] + group.z * 4u, end = offsets[expert + 1u];
  if (col >= a.n || begin >= end) return;
  uint slot[4];
  for (uint r = 0; r < 4u; ++r) slot[r] = begin + r < end ? sorted[begin + r] : 0;
  float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const size_t base = (size_t)expert * 2u * a.n + 2u * col;
  for (uint word = lane; word < a.k / 8u; word += 32u) {
    const uint gw = w[base * (a.k / 8u) + word], uw = w[(base + 1u) * (a.k / 8u) + word];
    const size_t g = base * (a.k / 64u) + word / 8u, u = g + a.k / 64u;
    const float gs = float(s[g]), gb = float(b[g]), us = float(s[u]), ub = float(b[u]);
    for (uint j = 0; j < 8u; ++j) {
      const float gv = float((gw >> (j * 4u)) & 15u) * gs + gb;
      const float uv = float((uw >> (j * 4u)) & 15u) * us + ub;
      for (uint r = 0; r < 4u; ++r) if (begin + r < end) {
        const float v = x[(size_t)(slot[r] / a.topk) * a.k + word * 8u + j];
        gate[r] = fma(v, gv, gate[r]); up[r] = fma(v, uv, up[r]);
      }
    }
  }
  for (uint r = 0; r < 4u; ++r) {
    const float g = simd_sum(gate[r]), u = simd_sum(up[r]);
    if (lane == 0u && begin + r < end) {
      const float gg = a.gate_cap * precise::tanh(g / a.gate_cap) / (1.0f + precise::exp(-g));
      const float uu = a.up_cap > 0.0f ? a.up_cap * precise::tanh(u / a.up_cap) : u;
      y[(size_t)slot[r] * a.n + col] = gg * uu;
    }
  }
}

[[kernel]] void i2a_group_down(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u, expert = group.y;
  const uint begin = offsets[expert] + group.z * 4u, end = offsets[expert + 1u];
  if (col >= a.n || begin >= end) return;
  uint slot[4];
  for (uint r = 0; r < 4u; ++r) slot[r] = begin + r < end ? sorted[begin + r] : 0;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const size_t base = (size_t)expert * a.n + col;
  for (uint word = lane; word < a.k / 8u; word += 32u) {
    const uint packed = w[base * (a.k / 8u) + word];
    const size_t g = base * (a.k / 64u) + word / 8u;
    const float scale = float(s[g]), bias = float(b[g]);
    for (uint j = 0; j < 8u; ++j) {
      const float v = float((packed >> (j * 4u)) & 15u) * scale + bias;
      for (uint r = 0; r < 4u; ++r) if (begin + r < end)
        acc[r] = fma(x[(size_t)slot[r] * a.k + word * 8u + j], v, acc[r]);
    }
  }
  for (uint r = 0; r < 4u; ++r) {
    const float value = simd_sum(acc[r]);
    if (lane == 0u && begin + r < end) y[(size_t)slot[r] * a.n + col] = value;
  }
}
