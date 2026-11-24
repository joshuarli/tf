#if defined(I2P_TENSOR_UNITS)
[[kernel]] void i2pa_attn_units64(const device float* Q [[buffer(0)]], const device half* KEYS [[buffer(1)]],
    const device half* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2p_attn_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    const device half* KC [[buffer(7)]], const device half* VC [[buffer(8)]],
    device atomic_uint* overflow [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 128, R = 64, KB = 32, PAD = 136, THREADS = 384;
  threadgroup half keys[KB * PAD], values[KB * PAD];
  const int kv = int(tg.x), q0 = int(tg.y) * R, t = int(sg) * 32 + int(lane);
  const int h = kv * 3 + int(sg) / 4, r0 = q0 + 16 * (int(sg) % 4);
  const short2 home = i2p_home(lane);
  const int rows = int(a.rows);
  int row[2], pos[2];
  bool active[2];
  float tau[2], sk[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; i++) {
    row[i] = r0 + home.y + 8 * i;
    active[i] = row[i] < rows;
    pos[i] = int(a.base) + row[i];
    tau[i] = a.window ? 1.0f : 1.0f + a.alpha * precise::log(max(1.0f, (float(pos[i]) + 1.0f) / a.floor));
    sk[i] = active[i] ? sink[size_t(row[i]) * a.heads + h] : 0.0f;
  }
  uint over = 0u;
  i2p_h8 q[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short d = 0; d < D / 16; d++) {
    float4 v[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      v[i] = active[i] ? *(const device float4*)(Q + (size_t(row[i]) * a.heads + h) * D + 16 * d + home.x) : float4(0.0f);
      over += uint(abs(v[i].x) > 65504.0f) + uint(abs(v[i].y) > 65504.0f) + uint(abs(v[i].z) > 65504.0f) + uint(abs(v[i].w) > 65504.0f);
    }
    q[d] = i2p_h8(half4(clamp(v[0], -65504.0f, 65504.0f)), half4(clamp(v[1], -65504.0f, 65504.0f)));
  }
  i2p_f8 acc[D / 16];
  _Pragma("clang loop unroll(full)")
  for (short d = 0; d < D / 16; d++) acc[d] = i2p_f8(0.0f);
  float maximum[2] = {0.0f, 0.0f}, denominator[2] = {1.0f, 1.0f};
  const int first = int(a.base) + q0, last = int(a.base) + min(q0 + R, rows) - 1;
  const int lower = a.window && first + 1 > int(a.window) ? first + 1 - int(a.window) : 0;
  const int cols = int(a.extent) + 1;
  for (int block = lower / KB * KB; block <= last; block += KB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = t; e < KB * D / 8; e += THREADS) {
      const int k = e / (D / 8), c = (e % (D / 8)) * 8, j = block + k;
      half4 kk[2] = {half4(0.0h), half4(0.0h)}, vv[2] = {half4(0.0h), half4(0.0h)};
      if (j <= last) {
        const int rj = a.rmode && a.window ? j % int(a.window) : j;
        const size_t at = ((j < int(a.base) ? size_t(rj) : size_t(j - int(a.base))) * a.kv + kv) * D + c;
        const device half* ks = j < int(a.base) ? KEYS : KC;
        const device half* vs = j < int(a.base) ? VALUES : VC;
        _Pragma("clang loop unroll(full)")
        for (short u = 0; u < 2; u++) {
          kk[u] = *(const device half4*)(ks + at + 4 * u);
          vv[u] = *(const device half4*)(vs + at + 4 * u);
          over += uint(isinf(kk[u].x)) + uint(isinf(kk[u].y)) + uint(isinf(kk[u].z)) + uint(isinf(kk[u].w));
          over += uint(isinf(vv[u].x)) + uint(isinf(vv[u].y)) + uint(isinf(vv[u].z)) + uint(isinf(vv[u].w));
          kk[u] = clamp(kk[u], half4(-65504.0h), half4(65504.0h));
          vv[u] = clamp(vv[u], half4(-65504.0h), half4(65504.0h));
        }
      }
      *(threadgroup half4*)(keys + k * PAD + c) = kk[0];
      *(threadgroup half4*)(keys + k * PAD + c + 4) = kk[1];
      *(threadgroup half4*)(values + k * PAD + c) = vv[0];
      *(threadgroup half4*)(values + k * PAD + c + 4) = vv[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r0 >= rows) continue;
    i2p_f8 s[2] = {i2p_f8(0.0f), i2p_f8(0.0f)};
    _Pragma("clang loop unroll(full)")
    for (short d = 0; d < D / 16; d++) {
      const threadgroup half* kp = keys + home.y * PAD + 16 * d + home.x;
      const i2p_h8 k0 = i2p_h8(*(const threadgroup half4*)kp, *(const threadgroup half4*)(kp + 8 * PAD));
      const i2p_h8 k1 = i2p_h8(*(const threadgroup half4*)(kp + 16 * PAD), *(const threadgroup half4*)(kp + 24 * PAD));
      i2p_mma(s[0], s[1], q[d], k0, k1);
    }
    float bound[2] = {-INFINITY, -INFINITY};
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 2; j++) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 8; e++) {
        const short i = e >> 2;
        const int key = block + 16 * j + home.x + (e & 3);
        float score = -INFINITY;
        if (active[i] && key <= pos[i] && (!a.window || pos[i] - key < int(a.window))) {
          const int distance = min(pos[i] - key, int(a.extent));
          score = (s[j][e] * a.scale * tau[i] + B[(size_t(row[i]) * a.heads + h) * cols + distance] - sk[i]) * M_LOG2E_F;
        }
        s[j][e] = score;
        bound[i] = max(bound[i], score);
      }
    }
    float rescale[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      bound[i] = max(bound[i], simd_shuffle_xor(bound[i], ushort(1)));
      bound[i] = max(bound[i], simd_shuffle_xor(bound[i], ushort(8)));
      bound[i] = max(bound[i], maximum[i]);
      rescale[i] = precise::exp2(maximum[i] - bound[i]);
      maximum[i] = bound[i];
    }
    float total[2] = {0.0f, 0.0f};
    i2p_h8 p[2];
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 2; j++) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 8; e++) {
        const short i = e >> 2;
        const float weight = precise::exp2(s[j][e] - bound[i]);
        total[i] += weight;
        p[j][e] = half(weight);
      }
    }
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      total[i] += simd_shuffle_xor(total[i], ushort(1));
      total[i] += simd_shuffle_xor(total[i], ushort(8));
      denominator[i] = rescale[i] * denominator[i] + total[i];
    }
    _Pragma("clang loop unroll(full)")
    for (short d = 0; d < D / 16; d++) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 8; e++) acc[d][e] *= rescale[e >> 2];
    }
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 2; j++) {
      _Pragma("clang loop unroll(full)")
      for (short d = 0; d < D / 32; d++) {
        const threadgroup half* vp = values + (16 * j + home.y) * PAD + 32 * d + home.x;
        const i2p_h8 v0 = i2p_h8(*(const threadgroup half4*)vp, *(const threadgroup half4*)(vp + 8 * PAD));
        const i2p_h8 v1 = i2p_h8(*(const threadgroup half4*)(vp + 16), *(const threadgroup half4*)(vp + 8 * PAD + 16));
        i2p_mma_nn(acc[2 * d], acc[2 * d + 1], p[j], v0, v1);
      }
    }
  }
  if (over) atomic_fetch_add_explicit(overflow, over, memory_order_relaxed);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; i++) {
    if (!active[i]) continue;
    device float* out = O + (size_t(row[i]) * a.heads + h) * D + home.x;
    _Pragma("clang loop unroll(full)")
    for (short d = 0; d < D / 16; d++) {
      *(device float4*)(out + 16 * d) = float4(acc[d][4 * i], acc[d][4 * i + 1], acc[d][4 * i + 2], acc[d][4 * i + 3]) / denominator[i];
    }
  }
}
#endif
