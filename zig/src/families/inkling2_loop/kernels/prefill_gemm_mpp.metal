#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct i2p_gemm_args { uint n, k, rows, xs, xb; float cap; };
kernel void i2p_gemm_mpp(device float *x [[buffer(0)]], device bfloat *w [[buffer(1)]],
    constant i2p_gemm_args &a [[buffer(2)]], device float *y [[buffer(3)]],
    uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device float, dextents<int, 2>, tensor_inline> left(
        x + size_t(a.xb) * a.k, dextents<int, 2>(a.k, a.rows), array<int, 2>{1, int(a.xs * a.k)});
    tensor<device bfloat, dextents<int, 2>, tensor_inline> right(w, dextents<int, 2>(a.k, a.n));
    tensor<device float, dextents<int, 2>, tensor_inline> dest(y, dextents<int, 2>(a.n, a.rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = left.slice(0, gid.y * 16);
    auto bb = right.slice(0, gid.x * 32);
    auto cc = dest.slice(gid.x * 32, gid.y * 16);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    op.run(aa, bb, acc);
    if (a.cap > 0.0f) {
        for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = a.cap * precise::tanh(acc[i] / a.cap);
    }
    acc.store(cc);
}
