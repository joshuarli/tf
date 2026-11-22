#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

[[kernel]] void i2p_expert_mpp(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], const device float* CODES [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 32;
  const uint expert = tg.y;
  const int start = OFF[expert], end = OFF[expert + 1u];
  const uint m0 = tg.z * BM, n0 = tg.x * BN;
  if (end - start < I2P_MATRIX_THRESHOLD || m0 >= uint(end - start)) return;
  const bool gu = a.gate_cap > 0.0f;
  const uint cols = gu ? 2u * a.n : a.n;
  threadgroup float x[BM * BK];
#ifdef I2P_MXFP8
  threadgroup float w[BK * BN];
#else
  threadgroup half w[BK * BN];
#endif
  threadgroup float out[BM * BN];
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> xt(x, dextents<int, 2>(BK, BM));
#ifdef I2P_MXFP8
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> wt(w, dextents<int, 2>(BN, BK));
#else
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> wt(w, dextents<int, 2>(BN, BK));
#endif
  tensor<threadgroup float, dextents<int, 2>, tensor_inline> ot(out, dextents<int, 2>(BN, BM));
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto acc = op.get_destination_cooperative_tensor<decltype(xt), decltype(wt), float>();
  #pragma unroll
  for (uint i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  for (uint k0 = 0; k0 < a.k; k0 += BK) {
    for (uint i = ti; i < BM * BK; i += 128u) {
      const uint r = i / BK, k = i % BK;
      const bool valid = m0 + r < uint(end - start) && k0 + k < a.k;
      const uint slot = valid ? uint(ROW_OF[start + int(m0 + r)]) : 0u;
      const uint xr = gu ? slot / a.topk : slot;
      x[i] = valid ? X[size_t(xr) * a.k + k0 + k] : 0.0f;
    }
    for (uint i = ti; i < BK * BN; i += 128u) {
      const uint k = i / BN, n = n0 + i % BN, gk = k0 + k;
      float v = 0.0f;
      if (n < cols && gk < a.k) {
        const size_t row = size_t(expert) * cols + n;
#ifdef I2P_MXFP8
        const uint code = W[row * I2P_CODE_STRIDE(a.k) + gk];
#else
        const uint byte_code = W[row * I2P_CODE_STRIDE(a.k) + gk / 2u];
        const uint code = (byte_code >> ((gk & 1u) * 4u)) & 15u;
#endif
        v = CODES[code] * i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + gk / I2P_GROUP);
      }
#ifdef I2P_MXFP8
      w[i] = v;
#else
      w[i] = half(v);
#endif
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(xt, wt, acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  acc.store(ot);
  threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef I2P_MXFP8
  const float global = 1.0f;
#else
  const float global = amax[expert] / den[0];
#endif
  const uint width = gu ? BN / 2u : BN;
  for (uint i = ti; i < BM * width; i += 128u) {
    const uint r = i / width, col = i % width;
    const uint n = gu ? n0 / 2u + col : n0 + col;
    if (m0 + r < uint(end - start) && n < a.n) {
      const uint slot = uint(ROW_OF[start + int(m0 + r)]);
      const float value = gu ? i2p_situ(out[r * BN + 2u * col] * global,
          out[r * BN + 2u * col + 1u] * global, a.gate_cap, a.up_cap)
        : out[r * BN + col] * global;
      OUT[size_t(slot) * a.n + n] = value;
    }
  }
}
