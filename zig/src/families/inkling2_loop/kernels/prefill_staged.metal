#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct i2ps_part { uint tile, row, n, pad; };
struct i2ps_args { uint k, rows, count, pad; i2ps_part part[6]; };
struct i2ps_gate_args { uint count; };

template <bool LO>
inline void i2ps_gemm(device half* x, device half* w, constant i2ps_args& a, device float* y0, device float* y1,
    device float* y2, device float* y3, device float* y4, device float* y5, uint2 gid) {
  uint p = 0;
  for (uint i = 1; i < a.count; ++i) if (gid.x >= a.part[i].tile) p = i;
  const i2ps_part part = a.part[p];
  device float* y = p == 0 ? y0 : p == 1 ? y1 : p == 2 ? y2 : p == 3 ? y3 : p == 4 ? y4 : y5;
  const int col = int(gid.x - part.tile) * 64;
  tensor<device half, dextents<int, 2>, tensor_inline> hi(x, dextents<int, 2>(a.k, a.rows));
  tensor<device half, dextents<int, 2>, tensor_inline> lo(x + size_t(a.rows) * a.k, dextents<int, 2>(a.k, a.rows));
  tensor<device half, dextents<int, 2>, tensor_inline> right(w + size_t(part.row) * a.k, dextents<int, 2>(a.k, part.n));
  tensor<device float, dextents<int, 2>, tensor_inline> dest(y, dextents<int, 2>(part.n, a.rows));
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto ah = hi.slice(0, gid.y * 64);
  auto al = lo.slice(0, gid.y * 64);
  auto bb = right.slice(0, col);
  auto cc = dest.slice(col, gid.y * 64);
  auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ah)>, metal::remove_addrspace_t<decltype(bb)>, float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  op.run(ah, bb, acc);
  if (LO) op.run(al, bb, acc);
  acc.store(cc);
}

#define I2PS_GEMM(NAME, LO) \
kernel void NAME(device half* x [[buffer(0)]], device half* w [[buffer(1)]], constant i2ps_args& a [[buffer(2)]], \
    device float* y0 [[buffer(3)]], device float* y1 [[buffer(4)]], device float* y2 [[buffer(5)]], \
    device float* y3 [[buffer(6)]], device float* y4 [[buffer(7)]], device float* y5 [[buffer(8)]], \
    uint2 gid [[threadgroup_position_in_grid]]) { \
  i2ps_gemm<LO>(x, w, a, y0, y1, y2, y3, y4, y5, gid); \
}
I2PS_GEMM(i2ps_gemm_split, true)
I2PS_GEMM(i2ps_gemm_hi, false)

kernel void i2ps_split_gated(device const float* x [[buffer(0)]], device const float* g [[buffer(1)]],
    constant i2ps_gate_args& a [[buffer(2)]], device half* y [[buffer(3)]], device atomic_uint* overflow [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= a.count / 4u) return;
  const float4 xv = *(device const float4*)(x + 4u * i);
  const float4 gv = *(device const float4*)(g + 4u * i);
  float4 v;
  for (uint e = 0; e < 4u; ++e) v[e] = xv[e] / (1.0f + precise::exp(-gv[e]));
  if (any(abs(v) > 65504.0f)) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
  const half4 hi = half4(clamp(v, -65504.0f, 65504.0f));
  *(device half4*)(y + 4u * i) = hi;
  *(device half4*)(y + a.count + 4u * i) = half4(v - float4(hi));
}
