#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct i2as_args { uint n, k, rows, xs, xb, bits; float cap; };

kernel void i2as_prepare(device const uint* w [[buffer(0)]], device const bfloat* s [[buffer(1)]],
    device const bfloat* b [[buffer(2)]], constant i2as_args& a [[buffer(3)]], device half* out_w [[buffer(4)]],
    device atomic_uint* overflow [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n * a.k / 4u) {
    const uint row = i / (a.k / 4u), k = (i % (a.k / 4u)) * 4u, per = 32u / a.bits;
    const uint packed = w[(size_t)row * (a.k / per) + k / per];
    const uint shift = (k % per) * a.bits, mask = (1u << a.bits) - 1u;
    const size_t group = (size_t)row * (a.k / 64u) + k / 64u;
    const uint4 offsets = uint4(0u, a.bits, 2u * a.bits, 3u * a.bits);
    const float4 code = float4((uint4(packed >> shift) >> offsets) & mask);
    const float4 value = fma(code, float4(float(s[group])), float4(float(b[group])));
    if (any(abs(value) > 65504.0f)) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
    *reinterpret_cast<device half4*>(out_w + (size_t)row * a.k + k) = half4(value);
  }
}

kernel void i2as_gemm(device float* x [[buffer(0)]], device half* w [[buffer(1)]], constant i2as_args& a [[buffer(2)]],
    device float* y [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
  tensor<device float, dextents<int, 2>, tensor_inline> left(x + size_t(a.xb) * a.k, dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.xs * a.k)});
  tensor<device half, dextents<int, 2>, tensor_inline> right(w, dextents<int, 2>(a.k, a.n));
  tensor<device float, dextents<int, 2>, tensor_inline> dest(y, dextents<int, 2>(a.n, a.rows));
  constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto aa = left.slice(0, gid.y * 16);
  auto bb = right.slice(0, gid.x * 32);
  auto cc = dest.slice(gid.x * 32, gid.y * 16);
  auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  op.run(aa, bb, acc);
  if (a.cap > 0.0f) for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = a.cap * precise::tanh(acc[i] / a.cap);
  acc.store(cc);
}

// Strided f32 rows as hi = half(x) and lo = half(x - hi), about 22 bits of x; values past half range are counted.
kernel void i2as_split(device const float* x [[buffer(0)]], constant i2as_args& a [[buffer(1)]],
    device half* y [[buffer(2)]], device atomic_uint* overflow [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.rows * a.k / 4u) return;
  const uint row = i / (a.k / 4u), k = (i % (a.k / 4u)) * 4u;
  const float4 v = *(device const float4*)(x + size_t(a.xb + row * a.xs) * a.k + k);
  if (any(abs(v) > 65504.0f)) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
  const half4 hi = half4(clamp(v, -65504.0f, 65504.0f));
  *(device half4*)(y + size_t(row) * a.k + k) = hi;
  *(device half4*)(y + size_t(a.rows + row) * a.k + k) = half4(v - float4(hi));
}

// i2as_gemm with split rows: the hi and lo halves times the staged half weights, both into one f32 sum.
kernel void i2as_gemm_split(device half* x [[buffer(0)]], device half* w [[buffer(1)]], constant i2as_args& a [[buffer(2)]],
    device float* y [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
  tensor<device half, dextents<int, 2>, tensor_inline> hi(x, dextents<int, 2>(a.k, a.rows));
  tensor<device half, dextents<int, 2>, tensor_inline> lo(x + size_t(a.rows) * a.k, dextents<int, 2>(a.k, a.rows));
  tensor<device half, dextents<int, 2>, tensor_inline> right(w, dextents<int, 2>(a.k, a.n));
  tensor<device float, dextents<int, 2>, tensor_inline> dest(y, dextents<int, 2>(a.n, a.rows));
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto ah = hi.slice(0, gid.y * 64);
  auto al = lo.slice(0, gid.y * 64);
  auto bb = right.slice(0, gid.x * 64);
  auto cc = dest.slice(gid.x * 64, gid.y * 64);
  auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ah)>, metal::remove_addrspace_t<decltype(bb)>, float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  op.run(ah, bb, acc);
  op.run(al, bb, acc);
  if (a.cap > 0.0f) for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = a.cap * precise::tanh(acc[i] / a.cap);
  acc.store(cc);
}
