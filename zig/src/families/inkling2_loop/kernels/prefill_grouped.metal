#if defined(I2P_TENSOR_UNITS) && !defined(I2P_MXFP8)
constant constexpr uint I2PG_COLS = 8u;
constant constexpr uint I2PG_THREADS = 32u * I2PG_COLS;
constant constexpr uint I2PG_KB = 512u;
constant constexpr uint I2PG_UP_ROWS = I2P_MATRIX_THRESHOLD - 1u;
constant constexpr uint I2PG_DOWN_ROWS = I2P_DOWN_MATRIX_THRESHOLD - 1u;

inline float i2pg_code(uint c) {
  const uint mag = c & 7u;
  return as_type<float>(select(0x3f000000u + (mag << 22u), mag * 0x3f000000u, mag < 2u) | ((c & 8u) << 28u));
}

inline void i2pg_stage(threadgroup float* xs, const device float* X, const device int* ROW_OF, int start, int count,
    uint k, uint k0, uint per_row, uint ti) {
  const uint width = min(I2PG_KB, k - k0);
  for (uint i = ti; i < uint(count) * I2PG_KB; i += I2PG_THREADS) {
    const uint r = i / I2PG_KB, kk = i % I2PG_KB;
    if (kk < width) xs[r * I2PG_KB + kk] = X[size_t(uint(ROW_OF[start + int(r)]) / per_row) * k + k0 + kk];
  }
}

[[kernel]] void i2pg_gu(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* ACT [[buffer(8)]], device half* ACT_H [[buffer(9)]], device atomic_uint* overflow [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[I2PG_UP_ROWS * I2PG_KB];
  const uint e = tg.y;
  const int start = OFF[e], end = OFF[e + 1u], count = end - start;
  if (count <= 0 || count >= int(I2P_MATRIX_THRESHOLD)) return;
  const uint c = tg.x * I2PG_COLS + sg;
  const bool live = c < a.n;
  const size_t row = (size_t)e * a.n * 2u + 2u * min(c, a.n - 1u);
  const device uchar* wg_base = W + row * I2P_CODE_STRIDE(a.k);
  const device uchar* wu_base = wg_base + I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  ulong wg_cache[8];
  ulong wu_cache[8];
  float scg_cache[8];
  float scu_cache[8];
  for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
    const uint k0 = g * I2P_GROUP;
    wg_cache[idx] = *(const device ulong*)(wg_base + I2P_BYTE_OFF(k0));
    wu_cache[idx] = *(const device ulong*)(wu_base + I2P_BYTE_OFF(k0));
    scg_cache[idx] = i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g);
    scu_cache[idx] = i2p_decode_scale(S, (row + 1u) * I2P_SCALE_STRIDE(a.k) + g);
  }
  float dg[I2PG_UP_ROWS], du[I2PG_UP_ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < I2PG_UP_ROWS; ++r) { dg[r] = 0.0f; du[r] = 0.0f; }
  const uint ti = uint(sg) * 32u + uint(lane);
  for (uint idx = 0u, k0 = 0u; k0 < a.k; ++idx, k0 += I2PG_KB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i2pg_stage(xs, X, ROW_OF, start, count, a.k, k0, a.topk, ti);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint g = idx * 32u + lane;
    if (g >= groups) continue;
    const ulong ag = wg_cache[idx];
    const ulong au = wu_cache[idx];
    const threadgroup float* x = xs + lane * I2P_GROUP;
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < I2PG_UP_ROWS; ++r) {
      if (int(r) >= count) break;
      float pg = 0.0f;
      float pu = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 8u; ++j) {
        const uint ab = i2p_lane_byte(ag, j);
        const uint ub = i2p_lane_byte(au, j);
        const float x0 = x[r * I2PG_KB + 2u * j];
        const float x1 = x[r * I2PG_KB + 2u * j + 1u];
        pg = fma(i2pg_code(ab & 15u), x0, pg);
        pg = fma(i2pg_code(ab >> 4u), x1, pg);
        pu = fma(i2pg_code(ub & 15u), x0, pu);
        pu = fma(i2pg_code(ub >> 4u), x1, pu);
      }
      dg[r] = fma(scg_cache[idx], pg, dg[r]);
      du[r] = fma(scu_cache[idx], pu, du[r]);
    }
  }
  const float global = amax[e] / den[0];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < I2PG_UP_ROWS; ++r) {
    if (int(r) >= count) break;
    const float g = simd_sum(dg[r]);
    const float u = simd_sum(du[r]);
    if (lane == 0u && live) {
      const uint slot = uint(ROW_OF[start + int(r)]);
      const float value = i2p_situ(g * global, u * global, a.gate_cap, a.up_cap);
      ACT[(size_t)slot * a.n + c] = value;
      if (count >= int(I2P_DOWN_MATRIX_THRESHOLD)) {
        if (abs(value) > 65504.0f) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
        ACT_H[(size_t)slot * a.n + c] = half(clamp(value, -65504.0f, 65504.0f));
      }
    }
  }
}

[[kernel]] void i2pg_down(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float xs[I2PG_DOWN_ROWS * I2PG_KB];
  const uint e = tg.y;
  const int start = OFF[e], end = OFF[e + 1u], count = end - start;
  if (count <= 0 || count >= int(I2P_DOWN_MATRIX_THRESHOLD)) return;
  const uint c = tg.x * I2PG_COLS + sg;
  const bool live = c < a.n;
  const size_t row = (size_t)e * a.n + min(c, a.n - 1u);
  const device uchar* w_base = W + row * I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  ulong w_cache[8];
  float sc_cache[8];
  for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
    w_cache[idx] = *(const device ulong*)(w_base + I2P_BYTE_OFF(g * I2P_GROUP));
    sc_cache[idx] = i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g);
  }
  float d[I2PG_DOWN_ROWS];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < I2PG_DOWN_ROWS; ++r) d[r] = 0.0f;
  const uint ti = uint(sg) * 32u + uint(lane);
  for (uint idx = 0u, k0 = 0u; k0 < a.k; ++idx, k0 += I2PG_KB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i2pg_stage(xs, ACT, ROW_OF, start, count, a.k, k0, 1u, ti);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint g = idx * 32u + lane;
    if (g >= groups) continue;
    const ulong aw = w_cache[idx];
    const threadgroup float* x = xs + lane * I2P_GROUP;
    _Pragma("clang loop unroll(full)")
    for (uint r = 0; r < I2PG_DOWN_ROWS; ++r) {
      if (int(r) >= count) break;
      float p = 0.0f;
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 8u; ++j) {
        const uint b = i2p_lane_byte(aw, j);
        p = fma(i2pg_code(b & 15u), x[r * I2PG_KB + 2u * j], p);
        p = fma(i2pg_code(b >> 4u), x[r * I2PG_KB + 2u * j + 1u], p);
      }
      d[r] = fma(sc_cache[idx], p, d[r]);
    }
  }
  const float global = amax[e] / den[0];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0; r < I2PG_DOWN_ROWS; ++r) {
    if (int(r) >= count) break;
    const float v = simd_sum(d[r]);
    if (lane == 0u && live) OUT[(size_t)uint(ROW_OF[start + int(r)]) * a.n + c] = v * global;
  }
}
#endif
