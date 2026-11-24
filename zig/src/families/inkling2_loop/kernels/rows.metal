#include <metal_stdlib>
using namespace metal;

struct i2r_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2r_projection_args { uint k, full, rows; uint m[8]; };

inline float4 i2r_value(uint packed, float scale, float bias) {
  return fma(float4((uint4(packed) >> uint4(0u, 8u, 16u, 24u)) & 255u), float4(scale), float4(bias));
}

inline float i2r_fma4(float4 x, float4 v, float acc) {
  acc = fma(x.x, v.x, acc);
  acc = fma(x.y, v.y, acc);
  acc = fma(x.z, v.z, acc);
  return fma(x.w, v.w, acc);
}

template <uint R, uint C>
inline void i2r_dot(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint col0, uint last, uint width, uint lane, thread float (&acc)[C][R]) {
  const uint words = width / 4u, groups = width / 64u;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) acc[c][r] = 0.0f;
  }
  for (uint word = lane; word < words; word += 32u) {
    float4 v[C];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
      const size_t group = col * groups + word / 16u;
      v[c] = i2r_value(w[col * words + word], float(s[group]), float(b[group]));
    }
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const float4 in = float4(*reinterpret_cast<const device packed_float4*>(x + r * step + word * 4u));
#pragma unroll
      for (uint c = 0; c < C; ++c) acc[c][r] = i2r_fma4(in, v[c], acc[c][r]);
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

template <uint BITS, uint R, uint C, uint U>
inline void i2r_dot_ahead(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint col0, uint last, uint width, uint lane, thread float (&acc)[C][R]) {
  constexpr uint per = 32u / BITS;
  constexpr uint mask = (1u << BITS) - 1u;
  const uint words = width / per;
  const uint groups = width / 64u;
  const uint4 shift = uint4(0u, BITS, BITS * 2u, BITS * 3u);
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
      float4 v0[C], v1[C];
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        v0[c] = fma(float4((uint4(packed[c][u]) >> shift) & mask), float4(scale[c][u]), float4(bias[c][u]));
        v1[c] = BITS == 4u ? fma(float4((uint4(packed[c][u] >> 16u) >> shift) & mask), float4(scale[c][u]), float4(bias[c][u])) : float4(0.0f);
      }
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        if (r >= rows) break;
        const device float* xr = x + r * step + at * per;
        const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(xr));
        const float4 in1 = BITS == 4u ? float4(*reinterpret_cast<const device packed_float4*>(xr + 4u)) : float4(0.0f);
#pragma unroll
        for (uint c = 0; c < C; ++c) {
          float a = acc[c][r];
          a = fma(in0.x, v0[c].x, a);
          a = fma(in0.y, v0[c].y, a);
          a = fma(in0.z, v0[c].z, a);
          a = fma(in0.w, v0[c].w, a);
          if (BITS == 4u) {
            a = fma(in1.x, v1[c].x, a);
            a = fma(in1.y, v1[c].y, a);
            a = fma(in1.z, v1[c].z, a);
            a = fma(in1.w, v1[c].w, a);
          }
          acc[c][r] = a;
        }
      }
    }
  }
  for (; word < words; word += 32u) {
    float4 v0[C], v1[C];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
      const uint packed = w[col * words + word];
      const size_t group = col * groups + (word * per) / 64u;
      const float scale = float(s[group]), bias = float(b[group]);
      v0[c] = fma(float4((uint4(packed) >> shift) & mask), float4(scale), float4(bias));
      v1[c] = BITS == 4u ? fma(float4((uint4(packed >> 16u) >> shift) & mask), float4(scale), float4(bias)) : float4(0.0f);
    }
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const device float* xr = x + r * step + word * per;
      const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(xr));
      const float4 in1 = BITS == 4u ? float4(*reinterpret_cast<const device packed_float4*>(xr + 4u)) : float4(0.0f);
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        float a = acc[c][r];
        a = fma(in0.x, v0[c].x, a);
        a = fma(in0.y, v0[c].y, a);
        a = fma(in0.z, v0[c].z, a);
        a = fma(in0.w, v0[c].w, a);
        if (BITS == 4u) {
          a = fma(in1.x, v1[c].x, a);
          a = fma(in1.y, v1[c].y, a);
          a = fma(in1.z, v1[c].z, a);
          a = fma(in1.w, v1[c].w, a);
        }
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
    constant i2r_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col0 = (group * 8u + tid / 32u) * C;
  const uint lane = tid % 32u;
  if (col0 >= a.n) return;
  float acc[C][R];
  i2r_dot_ahead<8u, R, C, U>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, col0, a.n - 1u, a.k, lane, acc);
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

template <uint R, uint C>
inline void i2r_store(constant i2r_dense_args& a, device float* y, uint col0, uint rows, thread const float (&acc)[C][R]) {
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (col0 + c >= a.n) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const float v = acc[c][r];
      y[(size_t)r * a.n + col0 + c] = a.cap > 0.0f ? a.cap * precise::tanh(v / a.cap) : v;
    }
  }
}

template <uint R, uint C, uint U>
[[kernel]] void i2r_dense_staged(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2r_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  constexpr uint span = 32u * U;
  threadgroup float4 xs[R * span];
  const uint col0 = (group * 8u + tid / 32u) * C;
  const uint lane = tid % 32u;
  const bool live = col0 < a.n;
  const uint words = a.k / 4u, groups = a.k / 64u, rows = min(a.rows, R);
  float acc[C][R];
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) acc[c][r] = 0.0f;
  }
  for (uint base = 0; base < words; base += span) {
    for (uint i = tid; i < R * span; i += 256u) {
      const uint r = i / span, at = base + i % span;
      xs[i] = r < rows && at < words ? float4(*reinterpret_cast<const device packed_float4*>(x + (size_t)(a.xb + r * a.xs) * a.k + at * 4u)) : float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (live) {
#pragma unroll
      for (uint u = 0; u < U; ++u) {
        const uint at = base + lane + 32u * u;
        if (at >= words) break;
        float4 v[C];
#pragma unroll
        for (uint c = 0; c < C; ++c) {
          const size_t col = min(col0 + c, a.n - 1u);
          const size_t g = col * groups + at / 16u;
          v[c] = i2r_value(w[col * words + at], float(s[g]), float(b[g]));
        }
#pragma unroll
        for (uint r = 0; r < R; ++r) {
          if (r >= rows) break;
          const float4 in = xs[r * span + lane + 32u * u];
#pragma unroll
          for (uint c = 0; c < C; ++c) acc[c][r] = i2r_fma4(in, v[c], acc[c][r]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!live) return;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      acc[c][r] = simd_sum(acc[c][r]);
    }
  }
  if (lane == 0u) i2r_store<R, C>(a, y, col0, rows, acc);
}

template <uint R, uint C>
[[kernel]] void i2r_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]],
    const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]],
    const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2r_projection_args& a [[buffer(20)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint row = (group.x * 8u + tid / 32u) * C;
  const uint lane = tid % 32u;
  const uint first = group.y * R;
  if (first >= a.rows) return;
  const uint rows = min(R, a.rows - first);
  uint which = 0u;
  for (; which < 8u; ++which) {
    if (row < a.m[which]) break;
    row -= a.m[which];
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  const device float* x = (key ? xk : val ? xv : xq) + (size_t)first * a.k;
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
  float acc[C][R];
  i2r_dot<R, C>(x, a.k, rows, w, s, b, row + offset, offset + a.m[which] - 1u, a.k, lane, acc);
  if (lane != 0u) return;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (row + c >= a.m[which]) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      out[(size_t)(first + r) * a.m[which] + row + c] = acc[c][r];
    }
  }
}

#define I2R_AHEAD(r) \
  template [[host_name("i2r_dense_ahead_" #r)]] [[kernel]] void i2r_dense_ahead<r, 4, 2>( \
      const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], \
      const device bfloat* b [[buffer(3)]], constant i2r_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], \
      uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]);

I2R_AHEAD(2)
I2R_AHEAD(4)
I2R_AHEAD(8)

#define I2R_STAGED(r) \
  template [[host_name("i2r_dense_staged_" #r)]] [[kernel]] void i2r_dense_staged<r, 2, 2>( \
      const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], \
      const device bfloat* b [[buffer(3)]], constant i2r_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], \
      uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]);

I2R_STAGED(2)
I2R_STAGED(4)
I2R_STAGED(8)

template [[host_name("i2r_projections_8")]] [[kernel]] void i2r_projections<4, 4>(
    const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]], const device float* xv [[buffer(2)]],
    const device uint* wq [[buffer(3)]], const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]], const device bfloat* bk [[buffer(8)]],
    const device uint* wv [[buffer(9)]], const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]], constant i2r_projection_args& a [[buffer(20)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]);
