[[kernel]] void i2p_expert_matrix_wide(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], const device float* CODES [[buffer(9)]],
    device atomic_uint* overflow [[buffer(10)]],
    uint3 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 64, P = 72;
  const uint expert = tg.y, m0 = tg.z * BM, n0 = tg.x * BN;
  const int start = OFF[expert], end = OFF[expert + 1u];
  const bool gu = a.gate_cap > 0.0f;
  if (end - start < (gu ? I2P_MATRIX_THRESHOLD : I2P_DOWN_MATRIX_THRESHOLD) || m0 >= uint(end - start)) return;
  const uint cols = gu ? 2u * a.n : a.n, sg = ti / 32u, lane = ti % 32u;
  threadgroup half x[BM * P], w[BN * P];
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += BK) {
    for (uint i = ti; i < BM * BK / 4u; i += 256u) {
      const uint r = i / (BK / 4u), k = (i % (BK / 4u)) * 4u;
      const bool valid = m0 + r < uint(end - start) && k0 + k < a.k;
      const uint slot = valid ? uint(ROW_OF[start + int(m0 + r)]) : 0u;
      const uint xr = gu ? slot / a.topk : slot;
      float4 value = 0.0f;
      if (valid && k0 + k + 3u < a.k) value = *reinterpret_cast<const device float4*>(X + size_t(xr) * a.k + k0 + k);
      else if (valid) for (uint j = 0; j < 4u && k0 + k + j < a.k; ++j) value[j] = X[size_t(xr) * a.k + k0 + k + j];
      if (tg.x == 0u) {
        const uint count = uint(abs(value.x) > 65504.0f) + uint(abs(value.y) > 65504.0f) + uint(abs(value.z) > 65504.0f) + uint(abs(value.w) > 65504.0f);
        if (count) atomic_fetch_add_explicit(overflow, count, memory_order_relaxed);
      }
      *reinterpret_cast<threadgroup half4*>(x + r * P + k) = half4(clamp(value, -65504.0f, 65504.0f));
    }
    const uint n = n0 + ti / 4u, gk0 = k0 + (ti % 4u) * 16u;
    const bool valid = n < cols && gk0 < a.k;
    const size_t weight_row = size_t(expert) * cols + n;
#ifdef I2P_MXFP8
    const uint exponent = valid ? uint(S[weight_row * I2P_SCALE_STRIDE(a.k) + gk0 / I2P_GROUP]) : 0u;
    const float scale = valid ? as_type<float>(exponent == 0u ? 0x00400000u : exponent << 23u) : 0.0f;
    for (uint j = 0; j < 16u; ++j) {
      float value = 0.0f;
      if (valid && gk0 + j < a.k) value = CODES[W[weight_row * I2P_CODE_STRIDE(a.k) + gk0 + j]] * scale;
      w[(ti / 4u) * P + (ti % 4u) * 16u + j] = half(clamp(value, -65504.0f, 65504.0f));
    }
#else
    const float scale = valid ? i2p_decode_scale(S, weight_row * I2P_SCALE_STRIDE(a.k) + gk0 / I2P_GROUP) : 0.0f;
    ulong packed = 0;
    if (valid && gk0 + 15u < a.k) packed = *reinterpret_cast<const device ulong*>(W + weight_row * I2P_CODE_STRIDE(a.k) + gk0 / 2u);
    for (uint j = 0; j < 16u; j += 4u) {
      uint4 code = 0u;
      if (valid && gk0 + 15u < a.k) code = uint4(packed >> (j * 4u)) >> uint4(0u, 4u, 8u, 12u) & 15u;
      else if (valid) for (uint e = 0; e < 4u && gk0 + j + e < a.k; ++e) {
        const uint at = gk0 + j + e;
        code[e] = (W[weight_row * I2P_CODE_STRIDE(a.k) + at / 2u] >> ((at & 1u) * 4u)) & 15u;
      }
      const uint4 mag = code & 7u;
      const uint4 bits = select(0x3f000000u + (mag << 22u), mag * 0x3f000000u, mag < 2u) | ((code & 8u) << 28u);
      const float4 value = as_type<float4>(bits) * scale;
      *reinterpret_cast<threadgroup half4*>(w + (ti / 4u) * P + (ti % 4u) * 16u + j) = half4(clamp(value, -65504.0f, 65504.0f));
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < BK; k += 8u) {
      simdgroup_matrix<half, 8, 8> b;
      simdgroup_load(b, w + sg * 8u * P + k, ulong(P), ulong2(0), true);
      for (uint r = 0; r < 4u; ++r) {
        simdgroup_matrix<half, 8, 8> a8;
        simdgroup_load(a8, x + r * 8u * P + k, ulong(P));
        simdgroup_multiply_accumulate(acc[r], a8, b, acc[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#ifdef I2P_MXFP8
  const float global = 1.0f;
#else
  const float global = amax[expert] / den[0];
#endif
  const uint fr = ((lane / 4u) & 4u) + (lane / 2u) % 4u;
  const uint fc = (((lane / 4u) & 2u) + lane % 2u) * 2u;
  for (uint r = 0; r < 4u; ++r) {
    if (m0 + r * 8u + fr >= uint(end - start)) continue;
    const uint col = n0 + sg * 8u + fc;
    const uint slot = uint(ROW_OF[start + int(m0 + r * 8u + fr)]);
    const float lo = acc[r].thread_elements()[0] * global;
    const float hi = acc[r].thread_elements()[1] * global;
    if (gu) {
      if (col / 2u < a.n) OUT[size_t(slot) * a.n + col / 2u] = i2p_situ(lo, hi, a.gate_cap, a.up_cap);
    } else {
      if (col < a.n) OUT[size_t(slot) * a.n + col] = lo;
      if (col + 1u < a.n) OUT[size_t(slot) * a.n + col + 1u] = hi;
    }
  }
}
