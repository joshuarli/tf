#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void fast_half_convert(device const float *x [[buffer(0)]], device half *h [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) h[i] = half(x[i]);
}

kernel void fast_half_half(device half *x [[buffer(0)]], device half *w [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &k [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &rows [[buffer(5)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    tensor<device half, dextents<int, 2>, tensor_inline> b(w, dextents<int, 2>(k, n));
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16); auto bb = b.slice(0, gid.x * 32); auto cc = c.slice(gid.x * 32, gid.y * 16);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    op.run(aa, bb, acc); acc.store(cc);
}

kernel void fast_half_mxfp8(device half *x [[buffer(0)]], device uchar *w [[buffer(1)]],
    device uchar *scales [[buffer(2)]], device float *y [[buffer(3)]], constant uint &k [[buffer(4)]],
    constant uint &n [[buffer(5)]], constant uint &rows [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline> b(w, dextents<int, 2>(k, n));
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16); auto bb = b.slice(0, gid.x * 32);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    auto tmp = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint block = 0; block < k / 32; ++block) {
        auto ap = a.slice(block * 32, gid.y * 16); auto bp = b.slice(block * 32, gid.x * 32);
        op.run(ap, bp, tmp);
        for (uint i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) continue;
            auto idx = acc.get_multidimensional_index(i);
            uint col = gid.x * 32 + uint(idx[0]);
            if (col < n) acc[i] += tmp[i] * exp2(float(int(scales[size_t(col) * (k / 32) + block]) - 127));
        }
    }
    auto cc = c.slice(gid.x * 32, gid.y * 16); acc.store(cc);
}
