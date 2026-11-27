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

#ifdef I2P_TENSOR_UNITS
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

typedef vec<half, 8> i2p_h8;
typedef vec<float, 8> i2p_f8;

// Lane l holds rows home.y and home.y + 8, columns home.x to home.x + 3, of a 16 x 16 tensor-unit fragment.
inline short2 i2p_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

// (lo | hi) += a (b0 | b1)^T: a 16 x 32 x 16 multiply-accumulate of half operands into f32 sums.
inline void i2p_mma(thread i2p_f8& lo, thread i2p_f8& hi, thread const i2p_h8& a, thread const i2p_h8& b0,
    thread const i2p_h8& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
      metal::remove_addrspace_t<decltype(right)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// (lo | hi) += a (b0 | b1): as i2p_mma with a K x N right operand, b0 holding columns 0 to 15 and b1 16 to 31.
inline void i2p_mma_nn(thread i2p_f8& lo, thread i2p_f8& hi, thread const i2p_h8& a, thread const i2p_h8& b0,
    thread const i2p_h8& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
      metal::remove_addrspace_t<decltype(right)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

// The tile'th 128-row tile over the experts holding at least `least` sorted rows: its expert, first row and rows.
inline bool i2p_expert_tile(const device int* OFF, int experts, int least, int tile, ushort lane,
    thread int& expert, thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? OFF[e] : 0, last = e < experts ? OFF[e + 1] : 0;
    const int count = last - first >= least ? (last - first + 127) / 128 : 0;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * 128;
      rows = min(128, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

inline half i2p_fp8_half(uchar bits) {
  const ushort v = bits & 127u;
  const ushort sign = ushort(bits & 128u) << 8u;
  const ushort u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign;
  return as_type<half>(u) * 256.0h;
}

inline half4 i2p_fp4x4(uint w) {
  const ushort4 c = ushort4(w & 15u, (w >> 4u) & 15u, (w >> 8u) & 15u, (w >> 12u) & 15u);
  const half4 mag = as_type<half4>(ushort4((c & 7u) << 9u)) * 16384.0h;
  return as_type<half4>(as_type<ushort4>(mag) | ushort4((c & 8u) << 12u));
}

// f32 rows as halves, clamped when a.y is set (as the NVFP4 matrix rounds them); values past half range counted.
[[kernel]] void i2p_half_rows(const device float4* X [[buffer(0)]], device half4* Y [[buffer(1)]],
    device atomic_uint* overflow [[buffer(2)]], constant uint2& a [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.x) return;
  const float4 v = X[i];
  const uint over = uint(abs(v.x) > 65504.0f) + uint(abs(v.y) > 65504.0f) + uint(abs(v.z) > 65504.0f) + uint(abs(v.w) > 65504.0f);
  if (over) atomic_fetch_add_explicit(overflow, over, memory_order_relaxed);
  Y[i] = a.y ? half4(clamp(v, -65504.0f, 65504.0f)) : half4(v);
}

template <bool GU, bool AFFINE>
inline void i2p_expert_units(const device half* X, const device uchar* W, const device uchar* S,
    const device uchar* G, const device float* den, const device int* OFF, const device int* ROW_OF,
    constant i2p_moeg_args& a, device half* ACT, device float* OUT, device atomic_uint* overflow,
    threadgroup half* w, threadgroup half* e4m3_lut, uint3 tg, ushort sg, ushort lane) {
  constexpr int BK = 64, PAD = 72;
  const int least = AFFINE ? 1 : (GU ? I2P_MATRIX_THRESHOLD : I2P_DOWN_MATRIX_THRESHOLD);
  int expert = 0, row0 = 0, rows = 0;
  if (!i2p_expert_tile(OFF, int(a.experts), least, int(tg.y), lane, expert, row0, rows)) return;
  const int cols = GU ? 2 * int(a.n) : int(a.n), K = int(a.k), n0 = int(tg.x) * 64;
  const int t = int(sg) * 32 + int(lane);
  if (!AFFINE) e4m3_lut[t] = i2p_fp8_half(uchar(t));
  const int tm = 32 * (int(sg) / 2), tn = 32 * (int(sg) % 2);
  const bool sg_live = tm < rows;
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
  const int wr = t / 4, wk = (t % 4) * 16;
  const bool wvalid = n0 + wr < cols;
  const bool count_w = AFFINE && row0 == OFF[expert];
  const size_t wrow = size_t(expert) * size_t(cols) + size_t(n0 + wr);
  const device uchar* wp = W + wrow * size_t(K / 2) + wk / 2;
  const device uchar* sp = S + (AFFINE ? 2 * wrow * size_t(K / 64) : wrow * size_t(K / 16) + wk / 16);
  const device bfloat* bp = (const device bfloat*)G + wrow * size_t(K / 64);
  constexpr int STRIDE_W = 64 * PAD;
  threadgroup half* mine0 = w + wr * PAD + wk;
  i2p_f8 acc[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    acc[m][0] = i2p_f8(0.0f);
    acc[m][1] = i2p_f8(0.0f);
  }
  uint over = 0u;
  if (!AFFINE) threadgroup_barrier(mem_flags::mem_threadgroup);
  auto load_tile = [&](int k0, int buf) {
    threadgroup half* mine = mine0 + buf * STRIDE_W;
    const uint2 c = wvalid ? *(const device uint2*)(wp + k0 / 2) : uint2(0u);
    if (AFFINE) {
      const float scale = wvalid ? float(((const device bfloat*)sp)[k0 / 64]) : 0.0f;
      const float bias = wvalid ? float(bp[k0 / 64]) : 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 4; i++) {
        const float4 v = fma(float4((uint4(c[i / 2] >> (16 * (i % 2))) >> uint4(0u, 4u, 8u, 12u)) & 15u), scale, bias);
        if (count_w) over += uint(abs(v.x) > 65504.0f) + uint(abs(v.y) > 65504.0f) + uint(abs(v.z) > 65504.0f) + uint(abs(v.w) > 65504.0f);
        *(threadgroup half4*)(mine + 4 * i) = half4(v);
      }
    } else {
      const half s0 = wvalid ? e4m3_lut[sp[k0 / 16]] : 0.0h;
      *(threadgroup half4*)(mine + 0) = i2p_fp4x4(c.x) * s0;
      *(threadgroup half4*)(mine + 4) = i2p_fp4x4(c.x >> 16) * s0;
      *(threadgroup half4*)(mine + 8) = i2p_fp4x4(c.y) * s0;
      *(threadgroup half4*)(mine + 12) = i2p_fp4x4(c.y >> 16) * s0;
    }
  };
  int buf = 0;
  if (K > 0) load_tile(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k0 = 0; k0 < K; k0 += BK) {
    if (k0 + BK < K) load_tile(k0 + BK, buf ^ 1);
    if (sg_live) {
      const threadgroup half* wb = w + buf * STRIDE_W + (tn + home.y) * PAD + home.x;
      _Pragma("clang loop unroll(full)")
      for (short q = 0; q < 4; q++) {
        const threadgroup half* b = wb + 16 * q;
        const i2p_h8 b0 = i2p_h8(*(const threadgroup half4*)b, *(const threadgroup half4*)(b + 8 * PAD));
        const i2p_h8 b1 = i2p_h8(*(const threadgroup half4*)(b + 16 * PAD), *(const threadgroup half4*)(b + 24 * PAD));
        _Pragma("clang loop unroll(full)")
        for (short m = 0; m < 2; m++) {
          const i2p_h8 x = i2p_h8(*(const device half4*)(xr[m][0] + k0 + 16 * q), *(const device half4*)(xr[m][1] + k0 + 16 * q));
          i2p_mma(acc[m][0], acc[m][1], x, b0, b1);
        }
      }
    }
    if (k0 + BK < K) threadgroup_barrier(mem_flags::mem_threadgroup);
    buf ^= 1;
  }
  if (!sg_live) {
    if (over) atomic_fetch_add_explicit(overflow, over, memory_order_relaxed);
    return;
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

[[kernel]] void i2p_attn_units(const device float* Q [[buffer(0)]], const device half* KEYS [[buffer(1)]],
    const device half* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2p_attn_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    const device half* KC [[buffer(7)]], const device half* VC [[buffer(8)]],
    device atomic_uint* overflow [[buffer(9)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int D = 128, R = 32, KB = 32, PAD = 136, THREADS = 192;
  threadgroup half keys[KB * PAD], values[KB * PAD];
  const int kv = int(tg.x), q0 = int(tg.y) * R, t = int(sg) * 32 + int(lane);
  const int h = kv * 3 + int(sg) / 2, r0 = q0 + 16 * (int(sg) % 2);
  const short2 home = i2p_home(lane);
  const int rows = int(a.rows);
  const int cols = int(a.extent) + 1;
  int row[2], pos[2];
  bool active[2];
  float scale_log2[2], sk_log2[2], far_shift[2];
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; i++) {
    row[i] = r0 + home.y + 8 * i;
    active[i] = row[i] < rows;
    pos[i] = int(a.base) + row[i];
    const float tau = a.window ? 1.0f : 1.0f + a.alpha * precise::log(max(1.0f, (float(pos[i]) + 1.0f) / a.floor));
    const float sk = active[i] ? sink[size_t(row[i]) * a.heads + h] : 0.0f;
    const float far_bias = active[i] ? B[(size_t(row[i]) * a.heads + h) * cols + a.extent] : 0.0f;
    scale_log2[i] = a.scale * tau * M_LOG2E_F;
    sk_log2[i] = sk * M_LOG2E_F;
    far_shift[i] = (far_bias - sk) * M_LOG2E_F;
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
  for (int block = lower / KB * KB; block <= last; block += KB) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool full_cache = !a.window && block + KB <= int(a.base);
    for (int e = t; e < KB * D / 8; e += THREADS) {
      const int k = e / (D / 8), c = (e % (D / 8)) * 8, j = block + k;
      if (full_cache) {
        const size_t at = (size_t(j) * a.kv + kv) * D + c;
        *(threadgroup ulong2*)(keys + k * PAD + c) = *(const device ulong2*)(KEYS + at);
        *(threadgroup ulong2*)(values + k * PAD + c) = *(const device ulong2*)(VALUES + at);
      } else if (j <= last) {
        const int rj = a.rmode && a.window ? j % int(a.window) : j;
        const size_t at = ((j < int(a.base) ? size_t(rj) : size_t(j - int(a.base))) * a.kv + kv) * D + c;
        const device half* ks = j < int(a.base) ? KEYS : KC;
        const device half* vs = j < int(a.base) ? VALUES : VC;
        *(threadgroup ulong2*)(keys + k * PAD + c) = *(const device ulong2*)(ks + at);
        *(threadgroup ulong2*)(values + k * PAD + c) = *(const device ulong2*)(vs + at);
      } else {
        *(threadgroup ulong2*)(keys + k * PAD + c) = ulong2(0ul);
        *(threadgroup ulong2*)(values + k * PAD + c) = ulong2(0ul);
      }
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
    if (!a.window && block + KB - 1 <= first - int(a.extent)) {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 2; j++) {
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; e++) {
          const short i = e >> 2;
          const float score = active[i] ? fma(s[j][e], scale_log2[i], far_shift[i]) : -INFINITY;
          s[j][e] = score;
          bound[i] = max(bound[i], score);
        }
      }
    } else {
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 2; j++) {
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 8; e++) {
          const short i = e >> 2;
          const int key = block + 16 * j + home.x + (e & 3);
          float score = -INFINITY;
          if (active[i] && key <= pos[i] && (!a.window || pos[i] - key < int(a.window))) {
            const int distance = min(pos[i] - key, int(a.extent));
            score = fma(s[j][e], scale_log2[i], fma(B[(size_t(row[i]) * a.heads + h) * cols + distance], M_LOG2E_F, -sk_log2[i]));
          }
          s[j][e] = score;
          bound[i] = max(bound[i], score);
        }
      }
    }
    float rescale[2];
    _Pragma("clang loop unroll(full)")
    for (short i = 0; i < 2; i++) {
      bound[i] = max(bound[i], simd_shuffle_xor(bound[i], ushort(1)));
      bound[i] = max(bound[i], simd_shuffle_xor(bound[i], ushort(8)));
      bound[i] = max(bound[i], maximum[i]);
      rescale[i] = fast::exp2(maximum[i] - bound[i]);
      maximum[i] = bound[i];
    }
    float total[2] = {0.0f, 0.0f};
    i2p_h8 p[2];
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 2; j++) {
      _Pragma("clang loop unroll(full)")
      for (short e = 0; e < 8; e++) {
        const short i = e >> 2;
        const float weight = fast::exp2(s[j][e] - bound[i]);
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
      if (rescale[0] != 1.0f || rescale[1] != 1.0f) {
        _Pragma("clang loop unroll(full)")
        for (short d = 0; d < D / 16; d++) {
          _Pragma("clang loop unroll(full)")
          for (short e = 0; e < 8; e++) acc[d][e] *= rescale[e >> 2];
        }
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

#define I2P_EXPERT_UNITS(NAME, GU, AFFINE) \
[[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* W [[buffer(1)]], \
    const device uchar* S [[buffer(2)]], const device uchar* G [[buffer(3)]], const device float* den [[buffer(4)]], \
    const device int* OFF [[buffer(5)]], const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]], \
    device void* Y [[buffer(8)]], device atomic_uint* overflow [[buffer(9)]], \
    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
  threadgroup half w[2 * 64 * 72]; \
  threadgroup half e4m3_lut[AFFINE ? 1 : 256]; \
  i2p_expert_units<GU, AFFINE>(X, W, S, G, den, OFF, ROW_OF, a, (device half*)Y, (device float*)Y, overflow, w, e4m3_lut, tg, sg, lane); \
}
I2P_EXPERT_UNITS(i2p_expert_units_up, true, false)
I2P_EXPERT_UNITS(i2p_expert_units_down, false, false)
I2P_EXPERT_UNITS(i2p_affine_units_up, true, true)
I2P_EXPERT_UNITS(i2p_affine_units_down, false, true)
#endif
