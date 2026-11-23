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
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u)
    acc = fma(x[(size_t)(row * a.xs + a.xb) * a.k + k], w[(size_t)col * a.k + k], acc);
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
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u)
    acc = fma(x[(size_t)(row * a.xs + a.xb) * a.k + k], i2a_weight(w, s, b, col, k, a.k, a.bits), acc);
  acc = simd_sum(acc);
  if (lane == 0u) y[(size_t)row * a.n + col] = a.cap > 0.0f ? a.cap * precise::tanh(acc / a.cap) : acc;
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
  float gate = 0.0f, up = 0.0f;
  for (uint k = lane; k < a.k; k += 32u) {
    const float v = x[(size_t)(slot / a.topk) * a.k + k];
    gate = fma(v, i2a_weight(w, s, b, base, k, a.k, 4u), gate);
    up = fma(v, i2a_weight(w, s, b, base + 1u, k, a.k, 4u), up);
  }
  gate = simd_sum(gate); up = simd_sum(up);
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
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u)
    acc = fma(x[(size_t)slot * a.k + k], i2a_weight(w, s, b, row, k, a.k, 4u), acc);
  acc = simd_sum(acc);
  if (lane == 0u) y[(size_t)slot * a.n + col] = acc;
}
