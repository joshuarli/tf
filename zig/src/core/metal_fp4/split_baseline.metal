#include <metal_stdlib>
using namespace metal;
#ifdef I2P_MXFP8
#define I2P_GROUP 32u
#define I2P_MATRIX_THRESHOLD 8
#define I2P_CODE_STRIDE(k) (k)
#define I2P_SCALE_STRIDE(k) ((k) / 32u)
#define I2P_BYTE_OFF(k) (k)
#else
#define I2P_GROUP 16u
#define I2P_MATRIX_THRESHOLD 16
#define I2P_CODE_STRIDE(k) ((k) / 2u)
#define I2P_SCALE_STRIDE(k) ((k) / 16u)
#define I2P_BYTE_OFF(k) ((k) / 2u)
#endif

inline float i2p_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

inline float i2p_e2m1(uchar c) {
  const float t[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
  const float v = t[c & 7u];
  return (c & 8u) ? -v : v;
}

inline float i2p_decode_scale(const device uchar* S, size_t at) {
#ifdef I2P_MXFP8
  return exp2(float(int(S[at]) - 127));
#else
  return i2p_e4m3(S[at]);
#endif
}

inline uint i2p_lane_byte(ulong w, uint j) {
  return uint(w >> (8u * j)) & 255u;
}
inline float i2p_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

struct i2p_moeg_args { uint k, n, slots, topk, experts; float gate_cap, up_cap; };
[[kernel]] void i2p_down_grouped(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], const device float* CODES [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  const uint e = tg.y;
  if (r >= a.n || e >= a.experts) return;
  const int start = OFF[e];
  const int end = OFF[e + 1u];
  if (start >= end || end - start >= I2P_MATRIX_THRESHOLD) return;
  const size_t row = (size_t)e * a.n + r;
  const device uchar* w_base = W + row * I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  const uint per = (groups + 31u) / 32u;
  ulong w_cache[8];
#ifdef I2P_MXFP8
  ulong w1_cache[8];
  ulong w2_cache[8];
  ulong w3_cache[8];
#endif
  float sc_cache[8];
  for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
    const uint k0 = g * I2P_GROUP;
#ifdef I2P_MXFP8
    const device ulong* pw = (const device ulong*)(w_base + k0);
    w_cache[idx] = pw[0]; w1_cache[idx] = pw[1]; w2_cache[idx] = pw[2]; w3_cache[idx] = pw[3];
#else
    w_cache[idx] = *(const device ulong*)(w_base + I2P_BYTE_OFF(k0));
#endif
    sc_cache[idx] = i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g);
  }
#ifdef I2P_MXFP8
  const float has_global = 0.0f;
#else
  const float global = amax[e] / den[0];
#endif
  for (int oi = start; oi < end; ++oi) {
    const uint slot = uint(ROW_OF[oi]);
    const device float* x = ACT + (size_t)slot * a.k;
    float d = 0.0f;
    for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
      const uint k0 = g * I2P_GROUP;
      float p = 0.0f;
#ifdef I2P_MXFP8
      const ulong aw[4] = { w_cache[idx], w1_cache[idx], w2_cache[idx], w3_cache[idx] };
      for (uint q = 0u; q < 4u; ++q) {
        for (uint j = 0u; j < 8u; ++j) p = fma(CODES[i2p_lane_byte(aw[q], j)], x[k0 + q * 8u + j], p);
      }
#else
      const ulong aw = w_cache[idx];
      for (uint j = 0u; j < 8u; ++j) {
        const uint b = i2p_lane_byte(aw, j);
        p = fma(CODES[b & 15u], x[k0 + 2u * j], p);
        p = fma(CODES[b >> 4u], x[k0 + 2u * j + 1u], p);
      }
#endif
      d = fma(sc_cache[idx], p, d);
    }
    d = simd_sum(d);
    if (lane == 0u) {
#ifdef I2P_MXFP8
      OUT[(size_t)slot * a.n + r] = d;
#else
      OUT[(size_t)slot * a.n + r] = d * global;
#endif
    }
  }
}

[[kernel]] void i2p_expert_matrix(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* OUT [[buffer(8)]], const device float* CODES [[buffer(9)]],
    uint3 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  constexpr uint BM = 32, BN = 64, BK = 32, AP = 40, BP = 72;
  const uint expert = tg.y;
  const int start = OFF[expert], end = OFF[expert + 1u];
  const uint m0 = tg.z * BM, n0 = tg.x * BN;
  if (end - start < I2P_MATRIX_THRESHOLD || m0 >= uint(end - start)) return;
  const bool gu = a.gate_cap > 0.0f;
  const uint cols = gu ? 2u * a.n : a.n;
  const uint sg = ti / 32u;
  threadgroup float x[BM * AP];
  threadgroup float w[BK * BP];
  threadgroup float out[BM * BP];
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (uint k0 = 0; k0 < a.k; k0 += BK) {
    for (uint i = ti; i < BM * BK; i += 256u) {
      const uint r = i / BK, k = i % BK;
      const bool valid = m0 + r < uint(end - start) && k0 + k < a.k;
      const uint slot = valid ? uint(ROW_OF[start + int(m0 + r)]) : 0u;
      const uint xr = gu ? slot / a.topk : slot;
      x[r * AP + k] = valid ? X[size_t(xr) * a.k + k0 + k] : 0.0f;
    }
    for (uint i = ti; i < BK * BN; i += 256u) {
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
      w[k * BP + i % BN] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < BK; k += 8u) {
      simdgroup_matrix<float, 8, 8> b;
      simdgroup_load(b, w + k * BP + sg * 8u, ulong(BP));
      for (uint r = 0; r < 4u; ++r) {
        simdgroup_matrix<float, 8, 8> a8;
        simdgroup_load(a8, x + r * 8u * AP + k, ulong(AP));
        simdgroup_multiply_accumulate(acc[r], a8, b, acc[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint r = 0; r < 4u; ++r) simdgroup_store(acc[r], out + r * 8u * BP + sg * 8u, ulong(BP));
  threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef I2P_MXFP8
  const float global = 1.0f;
#else
  const float global = amax[expert] / den[0];
#endif
  const uint width = gu ? BN / 2u : BN;
  for (uint i = ti; i < BM * width; i += 256u) {
    const uint r = i / width, col = i % width;
    const uint n = gu ? n0 / 2u + col : n0 + col;
    if (m0 + r < uint(end - start) && n < a.n) {
      const uint slot = uint(ROW_OF[start + int(m0 + r)]);
      float value = gu ? i2p_situ(out[r * BP + 2u * col] * global,
          out[r * BP + 2u * col + 1u] * global, a.gate_cap, a.up_cap)
        : out[r * BP + col] * global;
      OUT[size_t(slot) * a.n + n] = value;
    }
  }
}
