#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using scales_t = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
using weights_t = tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline, scales_t>;
kernel void mxfp8_tensorops(device half *x [[buffer(0)]], device uchar *w [[buffer(1)]], device uchar *s [[buffer(2)]], device float *y [[buffer(3)]], constant uint &k [[buffer(4)]], constant uint &n [[buffer(5)]], constant uint &rows [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    scales_t sp(s);
    weights_t b(w, dextents<int, 2>(k, n), sp);
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16);
    auto bb = b.slice(0, gid.x * 32);
    auto cc = c.slice(gid.x * 32, gid.y * 16);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    #pragma unroll
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    op.run(aa, bb, acc);
    acc.store(cc);
}
kernel void mxfp8_situ(device const float *raw [[buffer(0)]], device float *out [[buffer(1)]], constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float g = raw[2 * i], u = raw[2 * i + 1];
    out[i] = 4.0f * precise::tanh(g / 4.0f) / (1.0f + precise::exp(-g)) * 25.0f * precise::tanh(u / 25.0f);
}
