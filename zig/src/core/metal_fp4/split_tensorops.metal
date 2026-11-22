#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void split_convert(device const float *x [[buffer(0)]], device bfloat *parts [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &terms [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float v = x[i];
    bfloat hi = bfloat(v);
    parts[i] = hi;
    if (terms > 1) {
        float rest = v - float(hi);
        bfloat mid = bfloat(rest);
        parts[count + i] = mid;
        if (terms > 2) parts[2 * count + i] = bfloat(rest - float(mid));
    }
}

kernel void split_float(device float *x [[buffer(0)]], device bfloat *w [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &k [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &rows [[buffer(5)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device float, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> b(w, dextents<int, 2>(k, n));
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16); auto bb = b.slice(0, gid.x * 32); auto cc = c.slice(gid.x * 32, gid.y * 16);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    op.run(aa, bb, acc); acc.store(cc);
}

kernel void split_preconverted(device bfloat *x [[buffer(0)]], device bfloat *w [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &k [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &rows [[buffer(5)]], constant uint &terms [[buffer(6)]], uint2 gid [[threadgroup_position_in_grid]]) {
    tensor<device bfloat, dextents<int, 2>, tensor_inline> b(w, dextents<int, 2>(k, n));
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    tensor<device bfloat, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    constexpr auto desc = matmul2d_descriptor(16, 32, static_cast<int>(dynamic_extent), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto aa = a.slice(0, gid.y * 16); auto bb = b.slice(0, gid.x * 32); auto cc = c.slice(gid.x * 32, gid.y * 16);
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint t = 0; t < terms; ++t) {
        tensor<device bfloat, dextents<int, 2>, tensor_inline> part(x + size_t(t) * k * rows, dextents<int, 2>(k, rows));
        auto ap = part.slice(0, gid.y * 16);
        op.run(ap, bb, acc);
    }
    acc.store(cc);
}

inline float decode_e4(uchar c) {
    uint e = (c >> 3) & 15u, m = c & 7u;
    float v = e ? (1.0f + float(m) / 8.0f) * exp2(float(int(e) - 7)) : float(m) / 512.0f;
    return c & 128u ? -v : v;
}
inline float decode_e2(uchar c) {
    const float vals[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    float v = vals[c & 7u]; return c & 8u ? -v : v;
}

kernel void split_packed(device bfloat *x [[buffer(0)]], device uchar *w [[buffer(1)]],
    device uchar *s [[buffer(2)]], device float *y [[buffer(3)]], constant uint &k [[buffer(4)]],
    constant uint &n [[buffer(5)]], constant uint &rows [[buffer(6)]], constant uint &terms [[buffer(7)]],
    constant uint &is_mx [[buffer(8)]], uint2 gid [[threadgroup_position_in_grid]]) {
    constexpr auto desc = matmul2d_descriptor(16, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device bfloat, dextents<int, 2>, tensor_inline> a(x, dextents<int, 2>(k, rows));
    tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, rows));
    auto aa = a.slice(0, gid.y * 16);
    auto bb = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto acc = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(aa)>, metal::remove_addrspace_t<decltype(bb)>, float>();
    for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0;
    for (uint k0 = 0; k0 < k; k0 += 32) {
        for (uint i = 0; i < bb.get_capacity(); ++i) {
            if (!bb.is_valid_element(i)) continue;
            auto idx = bb.get_multidimensional_index(i);
            uint depth = k0 + uint(idx[0]); uint col = gid.x * 32 + uint(idx[1]);
            float value = 0;
            if (depth < k && col < n) {
                if (is_mx) value = decode_e4(w[size_t(col) * k + depth]) * exp2(float(int(s[size_t(col) * (k / 32) + depth / 32]) - 127));
                else {
                    uchar packed = w[size_t(col) * (k / 2) + depth / 2];
                    uchar code = (depth & 1) ? packed >> 4 : packed & 15;
                    value = decode_e2(code) * decode_e4(s[size_t(col) * (k / 16) + depth / 16]);
                }
            }
            bb[i] = bfloat(value);
        }
        for (uint t = 0; t < terms; ++t) {
            tensor<device bfloat, dextents<int, 2>, tensor_inline> part(x + size_t(t) * k * rows, dextents<int, 2>(k, rows));
            auto ap = part.slice(k0, gid.y * 16);
            op.run(ap, bb, acc);
        }
    }
    auto cc = c.slice(gid.x * 32, gid.y * 16); acc.store(cc);
}
