#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct i2af_args { uint n, k, rows, xs, xb, bits; float cap; };
kernel void i2af_gemm(device float* x [[buffer(0)]], device float* w [[buffer(1)]],
    constant i2af_args& a [[buffer(2)]], device float* y [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
  tensor<device float, dextents<int, 2>, tensor_inline> left(x + size_t(a.xb) * a.k,
      dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.xs * a.k)});
  tensor<device float, dextents<int, 2>, tensor_inline> right(w, dextents<int, 2>(a.k, a.n));
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
struct i2af_ctl_args { uint h, rows, k, cxs, cxb, mxs, mxb, ys, yb; };
kernel void i2af_gate_ctl(device float* x [[buffer(0)]], device float* wg [[buffer(1)]], device float* wf [[buffer(2)]],
    constant i2af_ctl_args& a [[buffer(3)]], const device float* CUR [[buffer(4)]], const device float* MSG [[buffer(5)]],
    device float* O [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  threadgroup float gt[16 * 32], ft[16 * 32];
  tensor<device float, dextents<int, 2>, tensor_inline> left(x, dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.k)});
  tensor<device float, dextents<int, 2>, tensor_inline> rg(wg, dextents<int, 2>(a.k, a.h));
  tensor<device float, dextents<int, 2>, tensor_inline> rf(wf, dextents<int, 2>(a.k, a.h));
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> dg(gt, dextents<int, 2>(32, 16));
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> df(ft, dextents<int, 2>(32, 16));
  constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto aa = left.slice(0, gid.y * 16);
  auto bg = rg.slice(0, gid.x * 32);
  auto bf = rf.slice(0, gid.x * 32);
  auto ag = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bg)>, float>();
  for (uint i = 0; i < ag.get_capacity(); ++i) ag[i] = 0;
  op.run(aa, bg, ag);
  ag.store(dg);
  auto af = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bf)>, float>();
  for (uint i = 0; i < af.get_capacity(); ++i) af[i] = 0;
  op.run(aa, bf, af);
  af.store(df);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = ti; e < 16u * 32u; e += 128u) {
    const uint r = e / 32u, col = gid.x * 32u + e % 32u, row = gid.y * 16u + r;
    if (row >= a.rows || col >= a.h) continue;
    const float c = CUR[(size_t)(a.cxb + row * a.cxs) * a.h + col];
    const float m = MSG[(size_t)(a.mxb + row * a.mxs) * a.h + col];
    const float g = precise::tanh(gt[e]);
    const float f = precise::tanh(ft[e]);
    O[(size_t)(a.yb + row * a.ys) * a.h + col] = c - f * c + g * m;
  }
}
