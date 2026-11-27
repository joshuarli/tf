#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct i2af_args { uint n, k, rows, xs, xb, bits; float cap; };
template <int BM, int BN, typename W>
inline void i2af_gemm_t(device float* x, device W* w, constant i2af_args& a, device float* y, uint2 gid) {
  tensor<device float, dextents<int, 2>, tensor_inline> left(x + size_t(a.xb) * a.k,
      dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.xs * a.k)});
  tensor<device W, dextents<int, 2>, tensor_inline> right(w, dextents<int, 2>(a.k, a.n));
  tensor<device float, dextents<int, 2>, tensor_inline> dest(y, dextents<int, 2>(a.n, a.rows));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto aa = left.slice(0, gid.y * BM);
  auto bb = right.slice(0, gid.x * BN);
  auto cc = dest.slice(gid.x * BN, gid.y * BM);
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
  op.run(aa, bb, acc);
  if (a.cap > 0.0f) for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = a.cap * precise::tanh(acc[i] / a.cap);
  acc.store(cc);
}

kernel void i2af_gemm(device float* x [[buffer(0)]], device float* w [[buffer(1)]],
    constant i2af_args& a [[buffer(2)]], device float* y [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
  i2af_gemm_t<16, 32>(x, w, a, y, gid);
}

kernel void i2af_gemm_bf16(device float* x [[buffer(0)]], device bfloat* w [[buffer(1)]],
    constant i2af_args& a [[buffer(2)]], device float* y [[buffer(3)]], uint2 gid [[threadgroup_position_in_grid]]) {
  i2af_gemm_t<16, 32>(x, w, a, y, gid);
}

struct i2af_ctl_args { uint h, rows, k, cxs, cxb, mxs, mxb, ys, yb; };

template <int BM, int BN, typename W>
inline void i2af_gate_ctl_t(device float* x, device W* wg, device W* wf, constant i2af_ctl_args& a,
    const device float* CUR, const device float* MSG, device float* O, uint2 gid, uint ti,
    threadgroup float* gt, threadgroup float* ft) {
  tensor<device float, dextents<int, 2>, tensor_inline> left(x, dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.k)});
  tensor<device W, dextents<int, 2>, tensor_inline> rg(wg, dextents<int, 2>(a.k, a.h));
  tensor<device W, dextents<int, 2>, tensor_inline> rf(wf, dextents<int, 2>(a.k, a.h));
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> dg(gt, dextents<int, 2>(BN, BM));
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> df(ft, dextents<int, 2>(BN, BM));
  constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto aa = left.slice(0, gid.y * BM);
  auto bg = rg.slice(0, gid.x * BN);
  auto bf = rf.slice(0, gid.x * BN);
  auto ag = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bg)>, float>();
  for (uint i = 0; i < ag.get_capacity(); ++i) ag[i] = 0;
  op.run(aa, bg, ag);
  ag.store(dg);
  auto af = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bf)>, float>();
  for (uint i = 0; i < af.get_capacity(); ++i) af[i] = 0;
  op.run(aa, bf, af);
  af.store(df);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint e = ti; e < uint(BM * BN); e += 128u) {
    const uint r = e / uint(BN), col = gid.x * uint(BN) + e % uint(BN), row = gid.y * uint(BM) + r;
    if (row >= a.rows || col >= a.h) continue;
    const float c = CUR[(size_t)(a.cxb + row * a.cxs) * a.h + col];
    const float m = MSG[(size_t)(a.mxb + row * a.mxs) * a.h + col];
    const float g = precise::tanh(gt[e]);
    const float f = precise::tanh(ft[e]);
    O[(size_t)(a.yb + row * a.ys) * a.h + col] = c - f * c + g * m;
  }
}

kernel void i2af_gate_ctl(device float* x [[buffer(0)]], device float* wg [[buffer(1)]], device float* wf [[buffer(2)]],
    constant i2af_ctl_args& a [[buffer(3)]], const device float* CUR [[buffer(4)]], const device float* MSG [[buffer(5)]],
    device float* O [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  threadgroup float gt[16 * 32], ft[16 * 32];
  i2af_gate_ctl_t<16, 32>(x, wg, wf, a, CUR, MSG, O, gid, ti, gt, ft);
}

kernel void i2af_gate_ctl_bf16(device float* x [[buffer(0)]], device bfloat* wg [[buffer(1)]], device bfloat* wf [[buffer(2)]],
    constant i2af_ctl_args& a [[buffer(3)]], const device float* CUR [[buffer(4)]], const device float* MSG [[buffer(5)]],
    device float* O [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  threadgroup float gt[32 * 64], ft[32 * 64];
  i2af_gate_ctl_t<32, 64>(x, wg, wf, a, CUR, MSG, O, gid, ti, gt, ft);
}
