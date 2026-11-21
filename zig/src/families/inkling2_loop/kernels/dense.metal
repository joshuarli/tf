#include <metal_stdlib>
using namespace metal;

struct i2_dm_args { uint m, k; float cap; };
struct i2_dmg_args { uint k, m0, m1, m2, m3; };
struct i2_dmgx_args { uint k0, k1, k2; uint m0, m1, m2, m3, m4, m5, m6, m7; uint x0, x1, x2, x3, x4, x5, x6, x7; };

inline float i2_dot(const device bfloat* w, threadgroup float* Xt, uint k, uint lane) {
  float acc = 0.0f;
  uint base = 0u;
  const device uint2* w4 = (const device uint2*)w;
  threadgroup float4* x4 = (threadgroup float4*)Xt;
  for (; base + 128u <= k; base += 128u) {
    const uint2 c = w4[base / 4u + lane];
    const float4 x = x4[base / 4u + lane];
    acc = fma(x.x, as_type<float>((c.x & 0xffffu) << 16), acc);
    acc = fma(x.y, as_type<float>(c.x & 0xffff0000u), acc);
    acc = fma(x.z, as_type<float>((c.y & 0xffffu) << 16), acc);
    acc = fma(x.w, as_type<float>(c.y & 0xffff0000u), acc);
  }
  for (uint i = base + lane; i < k; i += 32u) acc = fma(Xt[i], float(w[i]), acc);
  return simd_sum(acc);
}

[[kernel]] void i2_dmv_fused(const device float* X [[buffer(0)]],
    const device bfloat* W0 [[buffer(1)]], device float* O0 [[buffer(2)]],
    const device bfloat* W1 [[buffer(3)]], device float* O1 [[buffer(4)]],
    const device bfloat* W2 [[buffer(5)]], device float* O2 [[buffer(6)]],
    const device bfloat* W3 [[buffer(7)]], device float* O3 [[buffer(8)]],
    constant i2_dmg_args& a [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float Xt[4608];
  for (uint i = t; i < a.k; i += 256u) Xt[i] = X[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint ms[4] = { a.m0, a.m1, a.m2, a.m3 };
  uint g = tg * 8u + sg;
  uint pair = 4u;
  for (uint p = 0u; p < 4u; ++p) {
    if (g < ms[p]) {
      pair = p;
      break;
    }
    g -= ms[p];
  }
  if (pair == 4u) return;
  const device bfloat* w = pair == 0u ? W0 + (size_t)g * a.k : pair == 1u ? W1 + (size_t)g * a.k
      : pair == 2u ? W2 + (size_t)g * a.k : W3 + (size_t)g * a.k;
  device float* o = pair == 0u ? O0 + g : pair == 1u ? O1 + g : pair == 2u ? O2 + g : O3 + g;
  const float acc = i2_dot(w, Xt, a.k, lane);
  if (lane == 0u) o[0] = acc;
}

inline float i2_dotx(const device bfloat* w, const device float* X, uint k, uint lane) {
  float acc = 0.0f;
  uint base = 0u;
  const device uint2* w4 = (const device uint2*)w;
  const device float4* x4 = (const device float4*)X;
  for (; base + 128u <= k; base += 128u) {
    const uint2 c = w4[base / 4u + lane];
    const float4 x = x4[base / 4u + lane];
    acc = fma(x.x, as_type<float>((c.x & 0xffffu) << 16), acc);
    acc = fma(x.y, as_type<float>(c.x & 0xffff0000u), acc);
    acc = fma(x.z, as_type<float>((c.y & 0xffffu) << 16), acc);
    acc = fma(x.w, as_type<float>(c.y & 0xffff0000u), acc);
  }
  for (uint i = base + lane; i < k; i += 32u) acc = fma(X[i], float(w[i]), acc);
  return simd_sum(acc);
}

[[kernel]] void i2_dmv(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2_dm_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 8u + sg;
  if (row >= a.m) return;
  const float acc = i2_dotx(W + (size_t)row * a.k, X, a.k, lane);
  if (lane == 0u) O[row] = acc;
}

[[kernel]] void i2_dmv_cap(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2_dm_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 32u + sg;
  if (row >= a.m) return;
  const float acc = i2_dotx(W + (size_t)row * a.k, X, a.k, lane);
  if (lane == 0u) O[row] = a.cap * precise::tanh(acc / a.cap);
}

struct i2_ctl_args { uint k0, k1, m0, m1; };

[[kernel]] void i2_dmv_ctl(const device float* H2 [[buffer(0)]], const device bfloat* D [[buffer(1)]],
    const device bfloat* G [[buffer(2)]], const device bfloat* F [[buffer(3)]],
    device float* T0 [[buffer(4)]], device float* T1 [[buffer(5)]], device float* N [[buffer(6)]],
    constant i2_ctl_args& a [[buffer(7)]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint t [[thread_position_in_threadgroup]]) {
  threadgroup float H[4608];
  threadgroup float normed[64];
  threadgroup float4* h4 = (threadgroup float4*)H;
  const device float4* s4 = (const device float4*)H2;
  for (uint i = t; i < a.k0 / 4u; i += 1024u) h4[i] = s4[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = 0u; r < 2u; ++r) {
    const uint row = sg * 2u + r;
    if (row >= a.m0) continue;
    const float acc = i2_dot(D + (size_t)row * a.k0, H, a.k0, lane);
    if (lane == 0u) normed[row] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.m0) N[t] = normed[t];
  for (uint c = 0u; c < 9u; ++c) {
    float acc0[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint r = 0u; r < 8u; ++r) {
      const uint row = sg * 72u + c * 8u + r;
      const bool live = row < a.m1;
      const device bfloat* gw = G + (size_t)(live ? row : 0u) * a.k1;
      const device bfloat* fw = F + (size_t)(live ? row : 0u) * a.k1;
      float s0 = 0.0f, s1 = 0.0f;
      for (uint i = lane; i < a.k1; i += 32u) {
        s0 = fma(normed[i], float(gw[i]), s0);
        s1 = fma(normed[i], float(fw[i]), s1);
      }
      s0 = simd_sum(s0);
      s1 = simd_sum(s1);
      if (lane == 0u) {
        acc0[r] = s0;
        acc1[r] = s1;
      }
    }
    if (lane == 0u) {
      for (uint r = 0u; r < 8u; ++r) {
        const uint row = sg * 72u + c * 8u + r;
        if (row >= a.m1) continue;
        T0[row] = acc0[r];
        T1[row] = acc1[r];
      }
    }
  }
}

[[kernel]] void i2_dmv_fusedx(const device float* X0 [[buffer(0)]], const device float* X1 [[buffer(1)]],
    const device float* X2 [[buffer(2)]], const device bfloat* W0 [[buffer(3)]], device float* O0 [[buffer(4)]],
    const device bfloat* W1 [[buffer(5)]], device float* O1 [[buffer(6)]],
    const device bfloat* W2 [[buffer(7)]], device float* O2 [[buffer(8)]],
    const device bfloat* W3 [[buffer(9)]], device float* O3 [[buffer(10)]],
    const device bfloat* W4 [[buffer(11)]], device float* O4 [[buffer(12)]],
    const device bfloat* W5 [[buffer(13)]], device float* O5 [[buffer(14)]],
    const device bfloat* W6 [[buffer(15)]], device float* O6 [[buffer(16)]],
    const device bfloat* W7 [[buffer(17)]], device float* O7 [[buffer(18)]],
    constant i2_dmgx_args& a [[buffer(19)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  uint g = tg * 8u + sg;
  const uint ms[8] = { a.m0, a.m1, a.m2, a.m3, a.m4, a.m5, a.m6, a.m7 };
  const uint xs[8] = { a.x0, a.x1, a.x2, a.x3, a.x4, a.x5, a.x6, a.x7 };
  uint pair = 8u;
  for (uint p = 0u; p < 8u; ++p) {
    if (g < ms[p]) {
      pair = p;
      break;
    }
    g -= ms[p];
  }
  if (pair == 8u) return;
  const uint xsel = xs[pair];
  const device float* X = xsel == 0u ? X0 : xsel == 1u ? X1 : X2;
  const uint k = xsel == 0u ? a.k0 : xsel == 1u ? a.k1 : a.k2;
  const device bfloat* w = pair == 0u ? W0 + (size_t)g * k : pair == 1u ? W1 + (size_t)g * k
      : pair == 2u ? W2 + (size_t)g * k : pair == 3u ? W3 + (size_t)g * k
      : pair == 4u ? W4 + (size_t)g * k : pair == 5u ? W5 + (size_t)g * k
      : pair == 6u ? W6 + (size_t)g * k : W7 + (size_t)g * k;
  device float* o = pair == 0u ? O0 + g : pair == 1u ? O1 + g : pair == 2u ? O2 + g
      : pair == 3u ? O3 + g : pair == 4u ? O4 + g : pair == 5u ? O5 + g
      : pair == 6u ? O6 + g : O7 + g;
  const float acc = i2_dotx(w, X, k, lane);
  if (lane == 0u) o[0] = acc;
}
