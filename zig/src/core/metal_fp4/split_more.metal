#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

inline uchar encode_e4(float v) {
    float a = min(abs(v), 448.0f);
    uint b = as_type<uint>(a);
    b += 0x7ffffu + ((b >> 20) & 1u);
    uint code = a < 0.015625f ? uint(rint(a * 512.0f)) : (((b >> 23) - 120u) << 3) | ((b >> 20) & 7u);
    return uchar(min(code, 126u) | (signbit(v) ? 128u : 0u));
}

kernel void fast_fp8_quant(device const float *x [[buffer(0)]], device uchar *out [[buffer(1)]],
    device float *scales [[buffer(2)]], constant uint &k [[buffer(3)]], constant uint &rows [[buffer(4)]],
    uint block [[thread_position_in_grid]]) {
    if (block >= rows * (k / 32)) return;
    float largest = 0;
    for (uint i = 0; i < 32; ++i) largest = max(largest, abs(x[size_t(block) * 32 + i]));
    float s = largest > 0 ? largest / 448.0f : 1.0f;
    scales[block] = s;
    for (uint i = 0; i < 32; ++i) out[size_t(block) * 32 + i] = encode_e4(x[size_t(block) * 32 + i] / s);
}

kernel void fast_fp8_mxfp8(device uchar *x [[buffer(0)]], device uchar *w [[buffer(1)]],
    device uchar *ws [[buffer(2)]], device float *xs [[buffer(3)]], device float *y [[buffer(4)]],
    constant uint &k [[buffer(5)]], constant uint &n [[buffer(6)]], constant uint &rows [[buffer(7)]],
    uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device metal_fp8_e4m3_format, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
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
            uint col = gid.x * 32 + uint(idx[0]), row = gid.y * 16 + uint(idx[1]);
            if (col < n && row < rows) acc[i] += tmp[i] * xs[size_t(row) * (k / 32) + block] * exp2(float(int(ws[size_t(col) * (k / 32) + block]) - 127));
        }
    }
    auto cc = c.slice(gid.x * 32, gid.y * 16); acc.store(cc);
}

kernel void fast_half_mask(device const float *x [[buffer(0)]], device half *out [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    half v = half(x[i]);
    out[i] = (i % 32 < 16) ? v : half(0);
    out[count + i] = (i % 32 >= 16) ? v : half(0);
}

inline float scale_e4(uchar c) {
    uint e = (c >> 3) & 15u, m = c & 7u;
    float v = e ? (1.0f + float(m) / 8.0f) * exp2(float(int(e) - 7)) : float(m) / 512.0f;
    return c & 128u ? -v : v;
}

kernel void fast_half_nvfp4(device half *x [[buffer(0)]], device uchar *w [[buffer(1)]],
    device uchar *scales [[buffer(2)]], device float *y [[buffer(3)]], constant uint &k [[buffer(4)]],
    constant uint &n [[buffer(5)]], constant uint &rows [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device half, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    tensor<device metal_fp4_e2m1_format, dextents<int, 2>, tensor_inline> b(w, dextents<int, 2>(k, n), array<int, 2>{1, int((k + 255) & ~255u)});
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16); auto bb = b.slice(0, gid.x * 32);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    auto tmp = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint block = 0; block < k / 32; ++block) {
        auto bp = b.slice(block * 32, gid.x * 32);
        for (uint part = 0; part < 2; ++part) {
            tensor<device half, dextents<int, 2>, tensor_inline> xp(x + size_t(part) * k * rows, dextents<int, 2>(k, rows));
            auto ap = xp.slice(block * 32, gid.y * 16);
            op.run(ap, bp, tmp);
            for (uint i = 0; i < acc.get_capacity(); ++i) {
                if (!acc.is_valid_element(i)) continue;
                auto idx = acc.get_multidimensional_index(i);
                uint col = gid.x * 32 + uint(idx[0]);
                if (col < n) acc[i] += tmp[i] * scale_e4(scales[size_t(col) * (k / 16) + block * 2 + part]);
            }
        }
    }
    auto cc = c.slice(gid.x * 32, gid.y * 16); acc.store(cc);
}
