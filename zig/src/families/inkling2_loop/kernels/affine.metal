#include <metal_stdlib>
using namespace metal;

struct i2a_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2a_expert_args { uint k, n, slots, topk; float gate_cap, up_cap; };
struct i2a_embed_args { uint width, rows; float eps; };
struct i2a_projection_args { uint k, full; uint m[8]; };
struct i2a_pair_args { uint k, m0, m1; };

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

// Weight row `row` against `rows` inputs `stride` floats apart: each word and its scale and bias read once, each
// input's sum in i2a_dot's order.
inline void i2a_dot_rows(const device float* x, size_t stride, uint rows, const device uint* w,
                         const device bfloat* s, const device bfloat* b, size_t row, uint width, uint bits,
                         uint lane, thread float (&acc)[8]) {
  const uint per = 32u / bits;
  const uint mask = (1u << bits) - 1u;
  for (uint r = 0; r < 8u; ++r) acc[r] = 0.0f;
  for (uint word = lane; word < width / per; word += 32u) {
    const uint packed = w[row * (width / per) + word];
    const size_t group = row * (width / 64u) + (word * per) / 64u;
    const float scale = float(s[group]), bias = float(b[group]);
    const uint4 shift = uint4(0u, bits, bits * 2u, bits * 3u);
    const float4 code0 = float4((uint4(packed) >> shift) & mask);
    const float4 value0 = fma(code0, float4(scale), float4(bias));
    const float4 code1 = float4((uint4(packed >> 16u) >> shift) & mask);
    const float4 value1 = fma(code1, float4(scale), float4(bias));
    for (uint r = 0; r < 8u; ++r) {
      if (r >= rows) break;
      const device float* xr = x + r * stride + word * per;
      const float4 input0 = float4(*reinterpret_cast<const device packed_float4*>(xr));
      acc[r] = fma(input0.x, value0.x, acc[r]);
      acc[r] = fma(input0.y, value0.y, acc[r]);
      acc[r] = fma(input0.z, value0.z, acc[r]);
      acc[r] = fma(input0.w, value0.w, acc[r]);
      if (bits == 4u) {
        const float4 input1 = float4(*reinterpret_cast<const device packed_float4*>(xr + 4u));
        acc[r] = fma(input1.x, value1.x, acc[r]);
        acc[r] = fma(input1.y, value1.y, acc[r]);
        acc[r] = fma(input1.z, value1.z, acc[r]);
        acc[r] = fma(input1.w, value1.w, acc[r]);
      }
    }
  }
  for (uint r = 0; r < 8u; ++r) {
    if (r >= rows) break;
    acc[r] = simd_sum(acc[r]);
  }
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

// i2a_dense for up to eight rows sharing each weight read.
[[kernel]] void i2a_dense_shared(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2a_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group * 8u + tid / 32u;
  const uint lane = tid % 32u;
  if (col >= a.n) return;
  float acc[8];
  i2a_dot_rows(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, col, a.k, a.bits, lane, acc);
  if (lane == 0u) for (uint r = 0; r < 8u; ++r) {
    if (r >= a.rows) break;
    y[(size_t)r * a.n + col] = a.cap > 0.0f ? a.cap * precise::tanh(acc[r] / a.cap) : acc[r];
  }
}

[[kernel]] void i2a_float_control(const device float* x [[buffer(0)]], const device float* gain [[buffer(1)]],
    const device float* forget [[buffer(2)]], const device float* cur [[buffer(3)]],
    const device float* message [[buffer(4)]], device float* out [[buffer(5)]],
    constant i2a_pair_args& a [[buffer(6)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group * 8u + tid / 32u;
  const uint lane = tid % 32u;
  if (col >= a.m0) return;
  const device float* wg = gain + (size_t)col * a.k;
  const device float* wf = forget + (size_t)col * a.k;
  float g = 0.0f, f = 0.0f;
  uint base = 0u;
  for (; base + 128u <= a.k; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
    const float4 gv = float4(*reinterpret_cast<const device packed_float4*>(wg + k));
    const float4 fv = float4(*reinterpret_cast<const device packed_float4*>(wf + k));
    g = fma(xv.x, gv.x, g); g = fma(xv.y, gv.y, g); g = fma(xv.z, gv.z, g); g = fma(xv.w, gv.w, g);
    f = fma(xv.x, fv.x, f); f = fma(xv.y, fv.y, f); f = fma(xv.z, fv.z, f); f = fma(xv.w, fv.w, f);
  }
  for (uint k = base + lane; k < a.k; k += 32u) {
    g = fma(x[k], wg[k], g);
    f = fma(x[k], wf[k], f);
  }
  g = simd_sum(g);
  f = simd_sum(f);
  if (lane == 0u) out[col] = cur[col] - precise::tanh(f) * cur[col] + precise::tanh(g) * message[col];
}

struct i2a_batch_control_args { uint k, m0, m1, h, cxs, cxb, mxs, mxb, ys, yb; };

[[kernel]] void i2a_batch_float_control(const device float* x [[buffer(0)]], const device float* gain [[buffer(1)]],
    const device float* forget [[buffer(2)]], const device float* cur [[buffer(3)]],
    const device float* message [[buffer(4)]], device float* out [[buffer(5)]],
    constant i2a_batch_control_args& a [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint lane = tid % 32u;
  if (col >= a.m0) return;
  x += (size_t)group.y * a.k;
  cur += (size_t)(a.cxb + group.y * a.cxs) * a.h;
  message += (size_t)(a.mxb + group.y * a.mxs) * a.h;
  out += (size_t)(a.yb + group.y * a.ys) * a.h;
  const device float* wg = gain + (size_t)col * a.k;
  const device float* wf = forget + (size_t)col * a.k;
  float g = 0.0f, f = 0.0f;
  uint base = 0u;
  for (; base + 128u <= a.k; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
    const float4 gv = float4(*reinterpret_cast<const device packed_float4*>(wg + k));
    const float4 fv = float4(*reinterpret_cast<const device packed_float4*>(wf + k));
    g = fma(xv.x, gv.x, g); g = fma(xv.y, gv.y, g); g = fma(xv.z, gv.z, g); g = fma(xv.w, gv.w, g);
    f = fma(xv.x, fv.x, f); f = fma(xv.y, fv.y, f); f = fma(xv.z, fv.z, f); f = fma(xv.w, fv.w, f);
  }
  for (uint k = base + lane; k < a.k; k += 32u) {
    g = fma(x[k], wg[k], g);
    f = fma(x[k], wf[k], f);
  }
  g = simd_sum(g);
  f = simd_sum(f);
  if (lane == 0u) out[col] = cur[col] - precise::tanh(f) * cur[col] + precise::tanh(g) * message[col];
}

[[kernel]] void i2a_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]],
    const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]],
    const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2a_projection_args& a [[buffer(20)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint row = group * 8u + tid / 32u;
  const uint lane = tid % 32u;
  uint which = 0u;
  for (; which < 8u; ++which) {
    if (row < a.m[which]) break;
    row -= a.m[which];
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  const device float* x = key ? xk : val ? xv : xq;
  const device uint* w = a.full && key ? wk : a.full && val ? wv : wq;
  const device bfloat* s = a.full && key ? sk : a.full && val ? sv : sq;
  const device bfloat* b = a.full && key ? bk : a.full && val ? bv : bq;
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
  const float result = i2a_dot(x, w, s, b, row + offset, a.k, 8u, lane);
  if (lane == 0u) out[row] = result;
}

[[kernel]] void i2a_batch_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]],
    const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]],
    const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2a_projection_args& a [[buffer(20)]], uint2 group [[threadgroup_position_in_grid]],
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
  const device bfloat* s = a.full && key ? sk : a.full && val ? sv : sq;
  const device bfloat* b = a.full && key ? bk : a.full && val ? bv : bq;
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
  const float result = i2a_dot(x, w, s, b, row + offset, a.k, 8u, lane);
  if (lane == 0u) out[(size_t)group.y * a.m[which] + row] = result;
}

struct i2a_shared_projection_args { uint k, full, rows; uint m[8]; };

// i2a_batch_projections for up to eight rows sharing each weight read.
[[kernel]] void i2a_shared_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]],
    const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]],
    const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2a_shared_projection_args& a [[buffer(20)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint row = group * 8u + tid / 32u;
  const uint lane = tid % 32u;
  uint which = 0u;
  for (; which < 8u; ++which) {
    if (row < a.m[which]) break;
    row -= a.m[which];
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  const device float* x = key ? xk : val ? xv : xq;
  const device uint* w = a.full && key ? wk : a.full && val ? wv : wq;
  const device bfloat* s = a.full && key ? sk : a.full && val ? sv : sq;
  const device bfloat* b = a.full && key ? bk : a.full && val ? bv : bq;
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
  float acc[8];
  i2a_dot_rows(x, a.k, a.rows, w, s, b, row + offset, a.k, 8u, lane, acc);
  if (lane == 0u) for (uint r = 0; r < 8u; ++r) {
    if (r >= a.rows) break;
    out[(size_t)r * a.m[which] + row] = acc[r];
  }
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

#ifdef I2A_HALF_OPERANDS
[[kernel]] void i2a_dense_half_rows(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2a_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], device atomic_uint* overflow [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint bm = 32u, bn = 64u, bk = 32u, ap = 40u, bp = 72u;
  const uint sg = tid / 32u, m0 = group.y * bm, n0 = group.x * bn;
  threadgroup half inputs[bm * ap], weights[bk * bp];
  threadgroup float output[bm * bp];
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += bk) {
    for (uint i = tid; i < bm * bk; i += 256u) {
      const uint row = i / bk, k = i % bk;
      const float value = m0 + row < a.rows && k0 + k < a.k
        ? x[(size_t)(a.xb + (m0 + row) * a.xs) * a.k + k0 + k] : 0.0f;
      if (group.x == 0u && abs(value) > 65504.0f) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
      inputs[row * ap + k] = half(value);
    }
    for (uint i = tid; i < bk * bn / 4u; i += 256u) {
      const uint col = i / (bk / 4u), k = (i % (bk / 4u)) * 4u;
      const uint per = 32u / a.bits, mask = (1u << a.bits) - 1u;
      const bool valid = n0 + col < a.n && k0 + k < a.k;
      const size_t row = n0 + col;
      const uint packed = valid ? w[row * (a.k / per) + (k0 + k) / per] : 0u;
      const uint shift = ((k0 + k) % per) * a.bits;
      const size_t g = row * (a.k / 64u) + (k0 + k) / 64u;
      const float scale = valid ? float(s[g]) : 0.0f, bias = valid ? float(b[g]) : 0.0f;
      for (uint j = 0; j < 4u; ++j) {
        const float value = valid && k0 + k + j < a.k ? fma(float((packed >> (shift + j * a.bits)) & mask), scale, bias) : 0.0f;
        if (group.y == 0u && abs(value) > 65504.0f) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
        weights[(k + j) * bp + col] = half(value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < bk; k += 8u) {
      simdgroup_matrix<half, 8, 8> rhs;
      simdgroup_load(rhs, weights + k * bp + sg * 8u, ulong(bp));
      for (uint r = 0; r < 4u; ++r) {
        simdgroup_matrix<half, 8, 8> lhs;
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

#endif

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

inline void i2a_group_matrix(const device float* x, const device uint* w, const device bfloat* s,
    const device bfloat* b, const device uint* offsets, const device uint* sorted,
    constant i2a_group_args& a, device float* y, uint3 group, uint tid, bool up,
    threadgroup float* inputs, threadgroup float* weights, threadgroup float* output) {
  constexpr uint bm = 32u, bn = 64u, bk = 32u, ap = 40u, bp = 72u;
  const uint sg = tid / 32u, m0 = group.z * bm, n0 = group.x * bn;
  const uint start = offsets[group.y], end = offsets[group.y + 1u];
  if (start + m0 >= end) return;
  const uint cols = up ? 2u * a.n : a.n;
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += bk) {
    for (uint i = tid; i < bm * bk; i += 256u) {
      const uint row = i / bk, k = i % bk;
      const bool valid = start + m0 + row < end && k0 + k < a.k;
      const uint slot = valid ? sorted[start + m0 + row] : 0u;
      const uint xr = up ? slot / a.topk : slot;
      inputs[row * ap + k] = valid ? x[(size_t)xr * a.k + k0 + k] : 0.0f;
    }
    for (uint i = tid; i < bk * bn; i += 256u) {
      const uint k = i / bn, col = i % bn;
      const size_t row = (size_t)group.y * cols + n0 + col;
      weights[k * bp + col] = n0 + col < cols && k0 + k < a.k
        ? i2a_weight(w, s, b, row, k0 + k, a.k, 4u) : 0.0f;
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
  const uint outcols = up ? bn / 2u : bn;
  for (uint i = tid; i < bm * outcols; i += 256u) {
    const uint row = i / outcols, col = i % outcols;
    const uint target = up ? n0 / 2u + col : n0 + col;
    if (start + m0 + row < end && target < a.n) {
      const uint slot = sorted[start + m0 + row];
      float value = output[row * bp + (up ? col * 2u : col)];
      if (up) {
        const float u = output[row * bp + col * 2u + 1u];
        const float gate = a.gate_cap * precise::tanh(value / a.gate_cap) / (1.0f + precise::exp(-value));
        value = gate * (a.up_cap > 0.0f ? a.up_cap * precise::tanh(u / a.up_cap) : u);
      }
      y[(size_t)slot * a.n + target] = value;
    }
  }
}

[[kernel]] void i2a_group_up(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float inputs[32 * 40], weights[32 * 72], output[32 * 72];
  i2a_group_matrix(x, w, s, b, offsets, sorted, a, y, group, tid, true, inputs, weights, output);
}

[[kernel]] void i2a_group_down(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float inputs[32 * 40], weights[32 * 72], output[32 * 72];
  i2a_group_matrix(x, w, s, b, offsets, sorted, a, y, group, tid, false, inputs, weights, output);
}

#ifdef I2A_HALF_OPERANDS
inline void i2a_half_group_matrix(const device float* x, const device uint* w, const device bfloat* s,
    const device bfloat* b, const device uint* offsets, const device uint* sorted,
    constant i2a_group_args& a, device float* y, uint3 group, uint tid, bool up,
    threadgroup half* inputs, threadgroup half* weights, threadgroup float* output, device atomic_uint* overflow) {
  constexpr uint bm = 32u, bn = 64u, bk = 32u, ap = 40u, bp = 72u;
  const uint sg = tid / 32u, m0 = group.z * bm, n0 = group.x * bn;
  const uint start = offsets[group.y], end = offsets[group.y + 1u];
  if (start + m0 >= end) return;
  const uint cols = up ? 2u * a.n : a.n;
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += bk) {
    for (uint i = tid; i < bm * bk; i += 256u) {
      const uint row = i / bk, k = i % bk;
      const bool valid = start + m0 + row < end && k0 + k < a.k;
      const uint slot = valid ? sorted[start + m0 + row] : 0u;
      const uint xr = up ? slot / a.topk : slot;
      const float value = valid ? x[(size_t)xr * a.k + k0 + k] : 0.0f;
      if (group.x == 0u && abs(value) > 65504.0f) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
      inputs[row * ap + k] = half(value);
    }
    for (uint i = tid; i < bk * bn / 4u; i += 256u) {
      const uint col = i / (bk / 4u), k = (i % (bk / 4u)) * 4u;
      const size_t row = (size_t)group.y * cols + n0 + col;
      const bool valid = n0 + col < cols && k0 + k < a.k;
      const uint packed = valid ? w[row * (a.k / 8u) + (k0 + k) / 8u] : 0u;
      const uint shift = ((k0 + k) % 8u) * 4u;
      const size_t g = row * (a.k / 64u) + (k0 + k) / 64u;
      const float scale = valid ? float(s[g]) : 0.0f, bias = valid ? float(b[g]) : 0.0f;
      for (uint j = 0; j < 4u; ++j) {
        const float value = valid && k0 + k + j < a.k ? fma(float((packed >> (shift + j * 4u)) & 15u), scale, bias) : 0.0f;
        if (group.z == 0u && abs(value) > 65504.0f) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
        weights[(k + j) * bp + col] = half(value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < bk; k += 8u) {
      simdgroup_matrix<half, 8, 8> rhs;
      simdgroup_load(rhs, weights + k * bp + sg * 8u, ulong(bp));
      for (uint r = 0; r < 4u; ++r) {
        simdgroup_matrix<half, 8, 8> lhs;
        simdgroup_load(lhs, inputs + r * 8u * ap + k, ulong(ap));
        simdgroup_multiply_accumulate(acc[r], lhs, rhs, acc[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < 4u; ++r) simdgroup_store(acc[r], output + r * 8u * bp + sg * 8u, ulong(bp));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint outcols = up ? bn / 2u : bn;
  for (uint i = tid; i < bm * outcols; i += 256u) {
    const uint row = i / outcols, col = i % outcols;
    const uint target = up ? n0 / 2u + col : n0 + col;
    if (start + m0 + row < end && target < a.n) {
      const uint slot = sorted[start + m0 + row];
      float value = output[row * bp + (up ? col * 2u : col)];
      if (up) {
        const float u = output[row * bp + col * 2u + 1u];
        const float gate = a.gate_cap * precise::tanh(value / a.gate_cap) / (1.0f + precise::exp(-value));
        value = gate * (a.up_cap > 0.0f ? a.up_cap * precise::tanh(u / a.up_cap) : u);
      }
      y[(size_t)slot * a.n + target] = value;
    }
  }
}

[[kernel]] void i2a_half_group_up(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]], device atomic_uint* overflow [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half inputs[32 * 40], weights[32 * 72];
  threadgroup float output[32 * 72];
  i2a_half_group_matrix(x, w, s, b, offsets, sorted, a, y, group, tid, true, inputs, weights, output, overflow);
}

[[kernel]] void i2a_half_group_down(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], const device uint* sorted [[buffer(5)]],
    constant i2a_group_args& a [[buffer(6)]], device float* y [[buffer(7)]], device atomic_uint* overflow [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup half inputs[32 * 40], weights[32 * 72];
  threadgroup float output[32 * 72];
  i2a_half_group_matrix(x, w, s, b, offsets, sorted, a, y, group, tid, false, inputs, weights, output, overflow);
}

#endif
