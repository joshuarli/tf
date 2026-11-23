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

// The tile'th 64-row tile over the experts holding at least `least` sorted rows: its expert, first row and rows.
inline bool i2p_expert_tile(const device int* OFF, int experts, int least, int tile, ushort lane,
    thread int& expert, thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? OFF[e] : 0, last = e < experts ? OFF[e + 1] : 0;
    const int count = last - first >= least ? (last - first + 63) / 64 : 0;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * 64;
      rows = min(64, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

// The e2m1 codes in bits 0 to 15 of w as halves.
inline half4 i2p_fp4x4(uint w) {
  const uint4 c = (uint4(w) >> uint4(0u, 4u, 8u, 12u)) & 15u;
  const uint4 mag = c & 7u;
  const uint4 bits = select(0x3800u + (mag << 9), mag * 0x3800u, mag < 2u) | ((c & 8u) << 12);
  return as_type<half4>(ushort4(bits));
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

// A 64 x 64 routed expert tile on the tensor units: half rows times half weights (as the half matrices round them).
template <bool GU, bool AFFINE>
inline void i2p_expert_units(const device half* X, const device uchar* W, const device uchar* S,
    const device uchar* G, const device float* den, const device int* OFF, const device int* ROW_OF,
    constant i2p_moeg_args& a, device half* ACT, device float* OUT, device atomic_uint* overflow,
    threadgroup half* w, uint3 tg, ushort sg, ushort lane) {
  constexpr int BK = 64, PAD = 72;
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
  threadgroup half* mine = w + wr * PAD + wk;
  i2p_f8 acc[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    acc[m][0] = i2p_f8(0.0f);
    acc[m][1] = i2p_f8(0.0f);
  }
  uint over = 0u;
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint4 c = wvalid ? *(const device uint4*)(wp + k0 / 2) : uint4(0u);
    if (AFFINE) {
      const float scale = wvalid ? float(((const device bfloat*)sp)[k0 / 64]) : 0.0f;
      const float bias = wvalid ? float(bp[k0 / 64]) : 0.0f;
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 8; i++) {
        const float4 v = fma(float4((uint4(c[i / 2] >> (16 * (i % 2))) >> uint4(0u, 4u, 8u, 12u)) & 15u), scale, bias);
        if (count_w) over += uint(abs(v.x) > 65504.0f) + uint(abs(v.y) > 65504.0f) + uint(abs(v.z) > 65504.0f) + uint(abs(v.w) > 65504.0f);
        *(threadgroup half4*)(mine + 4 * i) = half4(v);
      }
    } else {
      const half s0 = wvalid ? half(i2p_e4m3(sp[k0 / 16])) : 0.0h;
      const half s1 = wvalid ? half(i2p_e4m3(sp[k0 / 16 + 1])) : 0.0h;
      _Pragma("clang loop unroll(full)")
      for (short i = 0; i < 4; i++) {
        const half s = i < 2 ? s0 : s1;
        *(threadgroup half4*)(mine + 8 * i) = i2p_fp4x4(c[i]) * s;
        *(threadgroup half4*)(mine + 8 * i + 4) = i2p_fp4x4(c[i] >> 16) * s;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < 4; q++) {
      const threadgroup half* b = w + (tn + home.y) * PAD + 16 * q + home.x;
      const i2p_h8 b0 = i2p_h8(*(const threadgroup half4*)b, *(const threadgroup half4*)(b + 8 * PAD));
      const i2p_h8 b1 = i2p_h8(*(const threadgroup half4*)(b + 16 * PAD), *(const threadgroup half4*)(b + 24 * PAD));
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < 2; m++) {
        const i2p_h8 x = i2p_h8(*(const device half4*)(xr[m][0] + k0 + 16 * q), *(const device half4*)(xr[m][1] + k0 + 16 * q));
        i2p_mma(acc[m][0], acc[m][1], x, b0, b1);
      }
    }
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

#define I2P_EXPERT_UNITS(NAME, GU, AFFINE) \
[[kernel]] void NAME(const device half* X [[buffer(0)]], const device uchar* W [[buffer(1)]], \
    const device uchar* S [[buffer(2)]], const device uchar* G [[buffer(3)]], const device float* den [[buffer(4)]], \
    const device int* OFF [[buffer(5)]], const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]], \
    device void* Y [[buffer(8)]], device atomic_uint* overflow [[buffer(9)]], \
    uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
  threadgroup half w[64 * 72]; \
  i2p_expert_units<GU, AFFINE>(X, W, S, G, den, OFF, ROW_OF, a, (device half*)Y, (device float*)Y, overflow, w, tg, sg, lane); \
}
I2P_EXPERT_UNITS(i2p_expert_units_up, true, false)
I2P_EXPERT_UNITS(i2p_expert_units_down, false, false)
I2P_EXPERT_UNITS(i2p_affine_units_up, true, true)
I2P_EXPERT_UNITS(i2p_affine_units_down, false, true)
#endif
