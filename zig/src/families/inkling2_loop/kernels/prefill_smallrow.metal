#include <metal_stdlib>
using namespace metal;

struct i2p_small_args { uint n, k, rows, xs, xb; float cap; };

[[kernel]] void i2p_small(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_small_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint col = tg * 8u + sg;
  if (col >= a.n) return;
  const device bfloat* w = W + size_t(col) * a.k;
  float acc[16] = {};
  uint base = 0;
  const device uint2* w4 = (const device uint2*)w;
  for (; base + 128u <= a.k; base += 128u) {
    const uint2 c = w4[base / 4u + lane];
    const float4 v = float4(as_type<float>((c.x & 0xffffu) << 16), as_type<float>(c.x & 0xffff0000u),
                            as_type<float>((c.y & 0xffffu) << 16), as_type<float>(c.y & 0xffff0000u));
    for (uint r = 0; r < a.rows; ++r) {
      const device float4* x = (const device float4*)(X + size_t(a.xb + r * a.xs) * a.k);
      const float4 q = x[base / 4u + lane];
      acc[r] = fma(q.x, v.x, acc[r]);
      acc[r] = fma(q.y, v.y, acc[r]);
      acc[r] = fma(q.z, v.z, acc[r]);
      acc[r] = fma(q.w, v.w, acc[r]);
    }
  }
  for (uint i = base + lane; i < a.k; i += 32u) {
    const float v = float(w[i]);
    for (uint r = 0; r < a.rows; ++r) acc[r] = fma(X[size_t(a.xb + r * a.xs) * a.k + i], v, acc[r]);
  }
  for (uint r = 0; r < a.rows; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) O[size_t(r) * a.n + col] = a.cap > 0 ? a.cap * precise::tanh(v / a.cap) : v;
  }
}
