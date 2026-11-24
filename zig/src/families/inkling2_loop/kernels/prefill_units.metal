#if defined(I2P_TENSOR_UNITS) && !defined(I2P_MXFP8)
struct i2pu_weights { uint4 c; float scale, bias; half s0, s1; };

template <bool AFFINE>
inline i2pu_weights i2pu_load(const device uchar* wp, const device uchar* sp, const device bfloat* bp, bool valid,
    int k0) {
  i2pu_weights out;
  out.c = valid ? *(const device uint4*)(wp + k0 / 2) : uint4(0u);
  if (AFFINE) {
    out.scale = valid ? float(((const device bfloat*)sp)[k0 / 64]) : 0.0f;
    out.bias = valid ? float(bp[k0 / 64]) : 0.0f;
  } else {
    out.s0 = valid ? half(i2p_e4m3(sp[k0 / 16])) : 0.0h;
    out.s1 = valid ? half(i2p_e4m3(sp[k0 / 16 + 1])) : 0.0h;
  }
  return out;
}

template <bool AFFINE>
inline void i2pu_decode(thread const i2pu_weights& wt, threadgroup half* mine, bool count, thread uint& over) {
  if (AFFINE) {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 8; i++) {
      const float4 v = fma(float4((uint4(wt.c[i / 2] >> (16 * (i % 2))) >> uint4(0u, 4u, 8u, 12u)) & 15u), wt.scale, wt.bias);
      if (count) over += uint(abs(v.x) > 65504.0f) + uint(abs(v.y) > 65504.0f) + uint(abs(v.z) > 65504.0f) + uint(abs(v.w) > 65504.0f);
      *(threadgroup half4*)(mine + 4 * i) = half4(v);
    }
  } else {
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 4; i++) {
      const half s = i < 2 ? wt.s0 : wt.s1;
      *(threadgroup half4*)(mine + 8 * i) = i2p_fp4x4(wt.c[i]) * s;
      *(threadgroup half4*)(mine + 8 * i + 4) = i2p_fp4x4(wt.c[i] >> 16) * s;
    }
  }
}

template <bool GU, bool AFFINE>
inline void i2pu_expert_units(const device half* X, const device uchar* W, const device uchar* S,
    const device uchar* G, const device float* den, const device int* OFF, const device int* ROW_OF,
    constant i2p_moeg_args& a, device half* ACT, device float* OUT, device atomic_uint* overflow,
    threadgroup half* w, uint3 tg, ushort sg, ushort lane) {
  constexpr int BK = 64, PAD = 72, TILE = 64 * PAD;
  const int least = AFFINE ? 1 : (GU ? I2P_MATRIX_THRESHOLD : I2P_DOWN_MATRIX_THRESHOLD);
  int expert = 0, row0 = 0, rows = 0;
  if (!i2p_expert_tile(OFF, int(a.experts), least, int(tg.y), lane, expert, row0, rows)) return;
  const int cols = GU ? 2 * int(a.n) : int(a.n), K = int(a.k), n0 = int(tg.x) * 64;
  const int t = int(sg) * 32 + int(lane);
  const int tm = 32 * (int(sg) / 2), tn = 32 * (int(sg) % 2);
  const short2 home = i2p_home(lane);
  const device half* xr[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    _Pragma("clang loop unroll(full)")
    for (short h = 0; h < 2; h++) {
      const int i = tm + 16 * m + home.y + 8 * h;
      const uint slot = uint(ROW_OF[row0 + (i < rows ? i : 0)]);
      xr[m][h] = X + size_t(GU ? slot / a.topk : slot) * size_t(K) + home.x;
    }
  }
  const int wr = t / 2, wk = (t % 2) * 32;
  const bool wvalid = n0 + wr < cols;
  const bool count_w = AFFINE && row0 == OFF[expert];
  const size_t wrow = size_t(expert) * size_t(cols) + size_t(n0 + wr);
  const device uchar* wp = W + wrow * size_t(K / 2) + wk / 2;
  const device uchar* sp = S + (AFFINE ? 2 * wrow * size_t(K / 64) : wrow * size_t(K / 16) + wk / 16);
  const device bfloat* bp = (const device bfloat*)G + wrow * size_t(K / 64);
  i2p_f8 acc[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    acc[m][0] = i2p_f8(0.0f);
    acc[m][1] = i2p_f8(0.0f);
  }
  uint over = 0u;
  i2pu_weights next = i2pu_load<AFFINE>(wp, sp, bp, wvalid, 0);
  i2pu_decode<AFFINE>(next, w + wr * PAD + wk, count_w, over);
  for (int k0 = 0, at = 0; k0 < K; k0 += BK, at ^= TILE) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool more = k0 + BK < K;
    if (more) next = i2pu_load<AFFINE>(wp, sp, bp, wvalid, k0 + BK);
    const threadgroup half* cur = w + at;
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < 4; q++) {
      const threadgroup half* b = cur + (tn + home.y) * PAD + 16 * q + home.x;
      const i2p_h8 b0 = i2p_h8(*(const threadgroup half4*)b, *(const threadgroup half4*)(b + 8 * PAD));
      const i2p_h8 b1 = i2p_h8(*(const threadgroup half4*)(b + 16 * PAD), *(const threadgroup half4*)(b + 24 * PAD));
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < 2; m++) {
        const i2p_h8 x = i2p_h8(*(const device half4*)(xr[m][0] + k0 + 16 * q), *(const device half4*)(xr[m][1] + k0 + 16 * q));
        i2p_mma(acc[m][0], acc[m][1], x, b0, b1);
      }
    }
    if (more) i2pu_decode<AFFINE>(next, w + (at ^ TILE) + wr * PAD + wk, count_w, over);
  }
  const float global = AFFINE ? 1.0f : ((const device float*)G)[expert] / den[0];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    _Pragma("clang loop unroll(full)")
    for (short h = 0; h < 2; h++) {
      const int r = tm + 16 * m + home.y + 8 * h;
      if (r >= rows) continue;
      const size_t slot = size_t(ROW_OF[row0 + r]);
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 2; j++) {
        const int col = n0 + tn + 16 * j + home.x;
        const i2p_f8 v = acc[m][j];
        if (GU) {
          _Pragma("clang loop unroll(full)")
          for (short e = 0; e < 2; e++) {
            if (col / 2 + e >= int(a.n)) continue;
            const float value = i2p_situ(v[4 * h + 2 * e] * global, v[4 * h + 2 * e + 1] * global, a.gate_cap, a.up_cap);
            over += uint(abs(value) > 65504.0f);
            ACT[slot * a.n + col / 2 + e] = AFFINE ? half(value) : half(clamp(value, -65504.0f, 65504.0f));
          }
        } else {
          _Pragma("clang loop unroll(full)")
          for (short e = 0; e < 4; e++) if (col + e < int(a.n)) OUT[slot * a.n + col + e] = v[4 * h + e] * global;
        }
      }
    }
  }
  if (over) atomic_fetch_add_explicit(overflow, over, memory_order_relaxed);
}

#define I2PU_EXPERT_UNITS(NAME, GU, AFFINE) \
[[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* W [[buffer(1)]], \
    const device uchar* S [[buffer(2)]], const device uchar* G [[buffer(3)]], const device float* den [[buffer(4)]], \
    const device int* OFF [[buffer(5)]], const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]], \
    device void* Y [[buffer(8)]], device atomic_uint* overflow [[buffer(9)]], \
    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
  threadgroup half w[2 * 64 * 72]; \
  i2pu_expert_units<GU, AFFINE>(X, W, S, G, den, OFF, ROW_OF, a, (device half*)Y, (device float*)Y, overflow, w, tg, sg, lane); \
}
I2PU_EXPERT_UNITS(i2pu_units_up, true, false)
I2PU_EXPERT_UNITS(i2pu_units_down, false, false)
I2PU_EXPERT_UNITS(i2pu_affine_up, true, true)
I2PU_EXPERT_UNITS(i2pu_affine_down, false, true)
#endif
