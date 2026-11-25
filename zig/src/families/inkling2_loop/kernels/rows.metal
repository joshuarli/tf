#include <metal_stdlib>
using namespace metal;

struct i2t_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2t_projection_args { uint k, full, rows; uint m[8]; };
struct i2t_moe { uint k, n, slots, topk; float gate_cap, up_cap; };

// Apple GPUs: lane l of a simdgroup matrix holds row fm and columns fn, fn + 1, as i2t_lane computes them.
inline void i2t_lane(uint lane, thread uint& fm, thread uint& fn) {
  const uint qid = lane >> 2;
  fm = (qid & 4u) | ((lane >> 1) & 3u);
  fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
}

inline thread float2& i2t_elements(thread simdgroup_matrix<float, 8, 8>& m) {
  return reinterpret_cast<thread float2&>(m.thread_elements());
}

// Apple GPUs: a float simdgroup multiply-accumulate is one fma chain over the inner index, in index order.
inline void i2t_mma(thread float2& c, float2 a, float2 b) {
  simdgroup_matrix<float, 8, 8> A, B, C, D;
  i2t_elements(A) = a;
  i2t_elements(B) = b;
  i2t_elements(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = i2t_elements(D);
}

[[kernel]] void i2t_probe(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]],
    device float* D [[buffer(2)]], uint tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const uint at = fm * 8u + fn;
  float2 d = float2(0.0f);
  for (uint s = 0; s < 2u; ++s) {
    const device float* a = A + (tile * 2u + s) * 64u;
    const device float* b = B + (tile * 2u + s) * 64u;
    i2t_mma(d, float2(a[at], a[at + 1u]), float2(b[at], b[at + 1u]));
  }
  D[tile * 64u + at] = d.x;
  D[tile * 64u + at + 1u] = d.y;
}

template <uint N>
inline float2 i2t_tree(thread float2 (&v)[N]) {
#pragma unroll
  for (uint width = 1; width < N; width <<= 1)
#pragma unroll
    for (uint i = 0; i < N; i += 2u * width) v[i] = v[i] + v[i + width];
  return v[0];
}

template <uint SG, uint CL, uint N>
inline void i2t_partials(thread float2 (&acc)[CL][N], threadgroup float2* part, uint sg, uint lane) {
#pragma unroll
  for (uint n = 0; n < N; ++n) {
    float2 v[CL];
#pragma unroll
    for (uint l = 0; l < CL; ++l) v[l] = acc[l][n];
    part[(sg * N + n) * 32u + lane] = i2t_tree(v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint SG, uint N>
inline float2 i2t_total(threadgroup const float2* part, uint n, uint lane) {
  float2 v[SG];
#pragma unroll
  for (uint i = 0; i < SG; ++i) v[i] = part[(i * N + n) * 32u + lane];
  return i2t_tree(v);
}

template <uint SG, uint T, uint G>
inline void i2t_affine(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint row0, uint last, uint k, uint sg, uint lane, threadgroup float2* part,
    thread float2 (&out)[G][T]) {
  constexpr uint CL = 32u / SG;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const uint words = k / 4u, groups = k / 64u, chunks = words / 64u;
  const uint half_word = 32u * (fn >> 2), shift = 8u * (fn & 3u), half_k = 128u * (fm >> 2) + (fm & 3u);
  bool live[2u * G];
  const device float* xr[2u * G];
#pragma unroll
  for (uint i = 0; i < 2u * G; ++i) {
    live[i] = 8u * (i / 2u) + fn + i % 2u < rows;
    xr[i] = x + (live[i] ? 8u * (i / 2u) + fn + i % 2u : 0u) * step;
  }
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(row0 + t * 8u + fm, last);
  float2 acc[CL][G * T] = {};
  for (uint c = 0; c < chunks; ++c) {
    const uint lead = sg * CL + 64u * c;
    const uint gi = (lead + half_word) / 16u;
    float scale[T], bias[T];
#pragma unroll
    for (uint t = 0; t < T; ++t) {
      scale[t] = float(s[row[t] * groups + gi]);
      bias[t] = float(b[row[t] * groups + gi]);
    }
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint kk = 4u * (lead + l) + half_k;
      float2 bx[G];
#pragma unroll
      for (uint g = 0; g < G; ++g) bx[g] = float2(live[2u * g] ? xr[2u * g][kk] : 0.0f, live[2u * g + 1u] ? xr[2u * g + 1u][kk] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const uint packed = w[row[t] * words + lead + l + half_word];
        const float2 code = float2(float((packed >> shift) & 255u), float((packed >> (shift + 8u)) & 255u));
        const float2 weight = fma(code, float2(scale[t]), float2(bias[t]));
#pragma unroll
        for (uint g = 0; g < G; ++g) i2t_mma(acc[l][g * T + t], weight, bx[g]);
      }
    }
  }
  i2t_partials<SG, CL, G * T>(acc, part, sg, lane);
#pragma unroll
  for (uint g = 0; g < G; ++g)
#pragma unroll
    for (uint t = 0; t < T; ++t) out[g][t] = i2t_total<SG, G * T>(part, g * T + t, lane);
}

template <uint SG, uint T, uint G>
[[kernel]] void i2t_dense(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 part[SG * G * T * 32];
  const uint row0 = group * 8u * T;
  float2 out[G][T];
  i2t_affine<SG, T, G>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, row0, a.n - 1u, a.k, sg, lane, part,
      out);
  if (sg != 0u) return;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint col = row0 + t * 8u + fm;
    if (col >= a.n) break;
#pragma unroll
    for (uint g = 0; g < G; ++g) {
      float2 v = out[g][t];
      if (a.cap > 0.0f) v = float2(a.cap * precise::tanh(v.x / a.cap), a.cap * precise::tanh(v.y / a.cap));
      if (8u * g + fn < a.rows) y[(size_t)(8u * g + fn) * a.n + col] = v.x;
      if (8u * g + fn + 1u < a.rows) y[(size_t)(8u * g + fn + 1u) * a.n + col] = v.y;
    }
  }
}

#define I2T_PROJECTION_ARGS const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]], \
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]], const device bfloat* sq [[buffer(4)]], \
    const device bfloat* bq [[buffer(5)]], const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]], \
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]], const device bfloat* sv [[buffer(10)]], \
    const device bfloat* bv [[buffer(11)]], device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], \
    device float* ov [[buffer(14)]], device float* orl [[buffer(15)]], device float* og [[buffer(16)]], \
    device float* os [[buffer(17)]], device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]], \
    constant i2t_projection_args& a [[buffer(20)]], uint group [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

template <uint SG, uint T, uint G>
[[kernel]] void i2t_projections(I2T_PROJECTION_ARGS) {
  threadgroup float2 part[SG * G * T * 32];
  uint which = 0u, first = group;
  for (; which < 8u; ++which) {
    const uint span = (a.m[which] + 8u * T - 1u) / (8u * T);
    if (first < span) break;
    first -= span;
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  uint offset = 0u;
  if (which == 1u) offset = a.full ? 0u : a.m[0];
  if (which == 2u) offset = a.full ? 0u : a.m[0] + a.m[1];
  if (which == 3u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]);
  if (which == 4u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3];
  if (which == 5u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3] + a.m[4];
  if (which == 6u) offset = a.m[1];
  if (which == 7u) offset = a.m[2];
  const uint m = a.m[which], row0 = first * 8u * T;
  float2 out[G][T];
  i2t_affine<SG, T, G>(key ? xk : val ? xv : xq, a.k, a.rows, a.full && key ? wk : a.full && val ? wv : wq,
      a.full && key ? sk : a.full && val ? sv : sq, a.full && key ? bk : a.full && val ? bv : bq, offset + row0,
      offset + m - 1u, a.k, sg, lane, part, out);
  if (sg != 0u) return;
  device float* y = which == 0u ? oq : which == 1u ? ok : which == 2u ? ov : which == 3u ? orl
      : which == 4u ? og : which == 5u ? os : which == 6u ? okd : ovd;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint row = row0 + t * 8u + fm;
    if (row >= m) break;
#pragma unroll
    for (uint g = 0; g < G; ++g) {
      if (8u * g + fn < a.rows) y[(size_t)(8u * g + fn) * m + row] = out[g][t].x;
      if (8u * g + fn + 1u < a.rows) y[(size_t)(8u * g + fn + 1u) * m + row] = out[g][t].y;
    }
  }
}

#define I2T_DENSE(name, sgs, t, g) \
  template [[host_name(#name)]] [[kernel]] void i2t_dense<sgs, t, g>( \
      const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], \
      const device bfloat* b [[buffer(3)]], constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], \
      uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

I2T_DENSE(i2t_dense_8_2, 8, 2, 1)
I2T_DENSE(i2t_dense16_8_1, 8, 1, 2)

#define I2T_PROJECTIONS(name, sgs, t, g) \
  template [[host_name(#name)]] [[kernel]] void i2t_projections<sgs, t, g>(I2T_PROJECTION_ARGS);

I2T_PROJECTIONS(i2t_projections_8_2, 8, 2, 1)
I2T_PROJECTIONS(i2t_projections16_8_1, 8, 1, 2)

template <uint R, uint C, uint U>
inline void i2r_dot_ahead(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint col0, uint last, uint width, uint lane, thread float (&acc)[C][R]) {
  constexpr uint per = 4u, mask = 255u;
  const uint words = width / per;
  const uint groups = width / 64u;
  const uint4 shift = uint4(0u, 8u, 16u, 24u);
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) acc[c][r] = 0.0f;
  }
  uint word = lane;
  for (; word + 32u * (U - 1u) < words; word += 32u * U) {
    uint packed[C][U];
    float scale[C][U], bias[C][U];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
#pragma unroll
      for (uint u = 0; u < U; ++u) {
        const uint at = word + 32u * u;
        packed[c][u] = w[col * words + at];
        const size_t group = col * groups + (at * per) / 64u;
        scale[c][u] = float(s[group]);
        bias[c][u] = float(b[group]);
      }
    }
#pragma unroll
    for (uint u = 0; u < U; ++u) {
      const uint at = word + 32u * u;
      float4 v0[C];
#pragma unroll
      for (uint c = 0; c < C; ++c) v0[c] = fma(float4((uint4(packed[c][u]) >> shift) & mask), float4(scale[c][u]), float4(bias[c][u]));
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        if (r >= rows) break;
        const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(x + r * step + at * per));
#pragma unroll
        for (uint c = 0; c < C; ++c) {
          float a = acc[c][r];
          a = fma(in0.x, v0[c].x, a);
          a = fma(in0.y, v0[c].y, a);
          a = fma(in0.z, v0[c].z, a);
          a = fma(in0.w, v0[c].w, a);
          acc[c][r] = a;
        }
      }
    }
  }
  for (; word < words; word += 32u) {
    float4 v0[C];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
      const size_t group = col * groups + (word * per) / 64u;
      v0[c] = fma(float4((uint4(w[col * words + word]) >> shift) & mask), float4(float(s[group])), float4(float(b[group])));
    }
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(x + r * step + word * per));
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        float a = acc[c][r];
        a = fma(in0.x, v0[c].x, a);
        a = fma(in0.y, v0[c].y, a);
        a = fma(in0.z, v0[c].z, a);
        a = fma(in0.w, v0[c].w, a);
        acc[c][r] = a;
      }
    }
  }
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      acc[c][r] = simd_sum(acc[c][r]);
    }
  }
}

template <uint R, uint C, uint U>
[[kernel]] void i2r_dense_ahead(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col0 = (group * 8u + tid / 32u) * C;
  const uint lane = tid % 32u;
  if (col0 >= a.n) return;
  float acc[C][R];
  i2r_dot_ahead<R, C, U>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, col0, a.n - 1u, a.k, lane, acc);
  if (lane != 0u) return;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (col0 + c >= a.n) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= a.rows) break;
      const float v = acc[c][r];
      y[(size_t)r * a.n + col0 + c] = a.cap > 0.0f ? a.cap * precise::tanh(v / a.cap) : v;
    }
  }
}

template [[host_name("i2r_dense_ahead_2")]] [[kernel]] void i2r_dense_ahead<2, 4, 2>(const device float* x [[buffer(0)]],
    const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]);

struct i2r_controls_args { uint h, rank, rows, msg0; uint xs[3], xb[3]; };

[[kernel]] void i2r_controls_down(const device float* X0 [[buffer(0)]], const device float* X1 [[buffer(1)]],
    const device float* X2 [[buffer(2)]], const device float* MSG [[buffer(3)]], const device float* W0 [[buffer(4)]],
    const device float* W1 [[buffer(5)]], const device float* W2 [[buffer(6)]], constant i2r_controls_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[2][8];
  threadgroup float inv[2];
  const uint j = group.z, row = group.y;
  const device float* cur = (j == 0u ? X0 : j == 1u ? X1 : X2) + size_t(a.xb[j] + row * a.xs[j]) * a.h;
  const device float* msg = MSG + size_t(a.msg0 + j + row * 4u) * a.h;
  float sums[2] = { 0.0f, 0.0f };
  for (uint i = tid; i < a.h; i += 256u) {
    sums[0] = fma(cur[i], cur[i], sums[0]);
    sums[1] = fma(msg[i], msg[i], sums[1]);
  }
  for (uint which = 0; which < 2u; ++which) {
    const float sum = simd_sum(sums[which]);
    if (tid % 32u == 0u) part[which][tid / 32u] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 2u) {
    float total = 0.0f;
    for (uint i = 0; i < 8u; ++i) total += part[tid][i];
    inv[tid] = 1.0f / sqrt(total / float(a.h) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u;
  if (col >= a.rank) return;
  const uint width = 2u * a.h;
  const device float* w = (j == 0u ? W0 : j == 1u ? W1 : W2) + size_t(col) * width;
  float acc = 0.0f;
  uint base = 0u;
  for (; base + 128u <= width; base += 128u) {
    const uint k = base + lane * 4u;
    const bool first = k < a.h;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(first ? cur + k : msg + (k - a.h))) * (first ? inv[0] : inv[1]);
    const float4 wv = float4(*reinterpret_cast<const device packed_float4*>(w + k));
    acc = fma(xv.x, wv.x, acc); acc = fma(xv.y, wv.y, acc);
    acc = fma(xv.z, wv.z, acc); acc = fma(xv.w, wv.w, acc);
  }
  for (uint k = base + lane; k < width; k += 32u) acc = fma(k < a.h ? cur[k] * inv[0] : msg[k - a.h] * inv[1], w[k], acc);
  acc = simd_sum(acc);
  if (lane == 0u) OUT[(size_t(j) * a.rows + row) * a.rank + col] = acc;
}
