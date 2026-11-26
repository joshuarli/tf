// The routed expert projections: gate/up over the checkpoint's interleaved stored rows, then down.
// One expert a threadgroup row, 32 output columns a threadgroup, 2 rows by 2 columns a thread; every member of the
// expert (up to 32 at a time) is folded through the same staged weight tile. NVFP4 is the default decoder.
#include <metal_stdlib>
using namespace metal;

struct i2_moe { uint k, n, rows; float gate_cap, up_cap; };

// An e4m3 code's value. 0x7f/0xff are NaN, as the reference's table reads them.
inline float i2_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  if (e != 0u) return as_type<float>(((uint(c) & 128u) << 24u) | ((e + 120u) << 23u) | (man << 20u));
  const float v = float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

// An e2m1 nibble's value.
inline float i2_e2m1(uchar c) {
  const float t[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
  const float v = t[c & 7u];
  return (c & 8u) ? -v : v;
}

// Expert `e`'s stored row `row` at input `k`, decoded as the checkpoint stores it.
inline float i2_weight(const device uchar* W, const device uchar* S, const device float* amax,
                       const device float* den, uint e, uint row, uint k, uint K, uint rows) {
  const size_t base = (size_t)e * rows + row;
#ifdef I2_MXFP8
  return i2_e4m3(W[base * K + k]) * exp2(float(int(S[base * (K / 32u) + k / 32u]) - 127));
#else
  const uchar b = W[base * (K / 2u) + k / 2u];
  const float code = (k & 1u) ? i2_e2m1(b >> 4) : i2_e2m1(b & 15u);
  return code * i2_e4m3(S[base * (K / 16u) + k / 16u]) * (amax[e] / den[0]);
#endif
}

// act[slot, col] = situ(gate, up) for each member of expert tg.y: one token, so every member shares the input
// row X, and ORDER names the pick slot its activation lands in.
[[kernel]] void i2_gu(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ORDER [[buffer(6)]], constant i2_moe& a [[buffer(7)]], device float* ACT [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint e = tg.y, col0 = tg.x * 32u;
  const uint from = uint(OFF[e]), to = uint(OFF[e + 1u]);
  const uint tx = tid.x % 16u, ty = tid.x / 16u;
  threadgroup float xs[32][9];
  threadgroup float wg[32][9];
  threadgroup float wu[32][9];
  for (uint base = from; base < to; base += 32u) {
    const uint cnt = min(32u, to - base);
    float ag[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    float au[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    for (uint k0 = 0; k0 < a.k; k0 += 8u) {
      for (uint i = tid.x; i < 256u; i += 256u) {
        const uint r = i / 8u, kk = i % 8u, gk = k0 + kk;
        xs[r][kk] = (r < cnt && gk < a.k) ? X[gk] : 0.0f;
        const uint col = col0 + r;
        const bool live = col < a.n && gk < a.k;
        wg[r][kk] = live ? i2_weight(W, S, amax, den, e, 2u * col, gk, a.k, a.rows) : 0.0f;
        wu[r][kk] = live ? i2_weight(W, S, amax, den, e, 2u * col + 1u, gk, a.k, a.rows) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < 8u; ++kk) {
        const float x0 = xs[ty * 2u][kk], x1 = xs[ty * 2u + 1u][kk];
        for (uint c = 0; c < 2u; ++c) {
          const uint lc = tx * 2u + c;
          ag[0][c] = fma(x0, wg[lc][kk], ag[0][c]);
          au[0][c] = fma(x0, wu[lc][kk], au[0][c]);
          ag[1][c] = fma(x1, wg[lc][kk], ag[1][c]);
          au[1][c] = fma(x1, wu[lc][kk], au[1][c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 2u; ++r) for (uint c = 0; c < 2u; ++c) {
      const uint member = ty * 2u + r;
      const uint col = col0 + tx * 2u + c;
      if (member >= cnt || col >= a.n) continue;
      const uint slot = uint(ORDER[base + member]);
      const float g = ag[r][c], u = au[r][c];
      const float gate = a.gate_cap * precise::tanh(g / a.gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
      const float up = a.up_cap > 0.0f ? a.up_cap * precise::tanh(u / a.up_cap) : u;
      ACT[(size_t)slot * a.n + col] = gate * up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// out[slot, col] = act[slot, :] . W2[e, col, :] for each member of expert tg.y.
[[kernel]] void i2_down(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ORDER [[buffer(6)]], constant i2_moe& a [[buffer(7)]], device float* OUT [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint e = tg.y, col0 = tg.x * 32u;
  const uint from = uint(OFF[e]), to = uint(OFF[e + 1u]);
  const uint tx = tid.x % 16u, ty = tid.x / 16u;
  threadgroup float xs[32][9];
  threadgroup float ws[32][9];
  for (uint base = from; base < to; base += 32u) {
    const uint cnt = min(32u, to - base);
    float acc[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    for (uint k0 = 0; k0 < a.k; k0 += 8u) {
      for (uint i = tid.x; i < 256u; i += 256u) {
        const uint r = i / 8u, kk = i % 8u, gk = k0 + kk;
        xs[r][kk] = (r < cnt && gk < a.k) ? ACT[(size_t)uint(ORDER[base + r]) * a.k + gk] : 0.0f;
        const uint col = col0 + r;
        ws[r][kk] = (col < a.n && gk < a.k) ? i2_weight(W, S, amax, den, e, col, gk, a.k, a.rows) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < 8u; ++kk) {
        const float x0 = xs[ty * 2u][kk], x1 = xs[ty * 2u + 1u][kk];
        for (uint c = 0; c < 2u; ++c) {
          const float w0 = ws[tx * 2u + c][kk];
          acc[0][c] = fma(x0, w0, acc[0][c]);
          acc[1][c] = fma(x1, w0, acc[1][c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 2u; ++r) for (uint c = 0; c < 2u; ++c) {
      const uint member = ty * 2u + r;
      const uint col = col0 + tx * 2u + c;
      if (member >= cnt || col >= a.n) continue;
      OUT[(size_t)uint(ORDER[base + member]) * a.n + col] = acc[r][c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#ifdef I2_MXFP8
#define I2_GROUP 32u
#define I2_CODE_STRIDE(k) (k)
#define I2_SCALE_STRIDE(k) ((k) / 32u)
#define I2_BYTE_OFF(k) (k)
#else
#define I2_GROUP 16u
#define I2_CODE_STRIDE(k) ((k) / 2u)
#define I2_SCALE_STRIDE(k) ((k) / 16u)
#define I2_BYTE_OFF(k) ((k) / 2u)
#endif

inline float i2_decode_scale(const device uchar* S, size_t at) {
#ifdef I2_MXFP8
  return exp2(float(int(S[at]) - 127));
#else
  return i2_e4m3(S[at]);
#endif
}

inline uint i2_lane_byte(ulong w, uint j) {
  return uint(w >> (8u * j)) & 255u;
}

inline float i2_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

inline void i2_table_pair(const device uchar* wg, const device uchar* wu, const device float* x,
    uint k0, const device float* codes, thread float& dg, thread float& du) {
#ifdef I2_MXFP8
  const ulong ag[4] = { *(const device ulong*)(wg), *(const device ulong*)(wg + 8),
                        *(const device ulong*)(wg + 16), *(const device ulong*)(wg + 24) };
  const ulong au[4] = { *(const device ulong*)(wu), *(const device ulong*)(wu + 8),
                        *(const device ulong*)(wu + 16), *(const device ulong*)(wu + 24) };
  for (uint q = 0; q < 4u; ++q) {
    for (uint j = 0; j < 8u; ++j) {
      const float xv = x[k0 + q * 8u + j];
      dg = fma(codes[i2_lane_byte(ag[q], j)], xv, dg);
      du = fma(codes[i2_lane_byte(au[q], j)], xv, du);
    }
  }
#else
  const ulong ag = *(const device ulong*)(wg);
  const ulong au = *(const device ulong*)(wu);
  for (uint j = 0; j < 8u; ++j) {
    const uint ab = i2_lane_byte(ag, j);
    const uint ub = i2_lane_byte(au, j);
    const float x0 = x[k0 + 2u * j];
    const float x1 = x[k0 + 2u * j + 1u];
    dg = fma(codes[ab & 15u], x0, dg);
    dg = fma(codes[ab >> 4u], x1, dg);
    du = fma(codes[ub & 15u], x0, du);
    du = fma(codes[ub >> 4u], x1, du);
  }
#endif
}

inline void i2_table_row(const device uchar* w, const device float* x,
    uint k0, const device float* codes, thread float& d) {
#ifdef I2_MXFP8
  const ulong a[4] = { *(const device ulong*)(w), *(const device ulong*)(w + 8),
                       *(const device ulong*)(w + 16), *(const device ulong*)(w + 24) };
  for (uint q = 0; q < 4u; ++q) {
    for (uint j = 0; j < 8u; ++j) d = fma(codes[i2_lane_byte(a[q], j)], x[k0 + q * 8u + j], d);
  }
#else
  const ulong a = *(const device ulong*)(w);
  for (uint j = 0; j < 8u; ++j) {
    const uint b = i2_lane_byte(a, j);
    d = fma(codes[b & 15u], x[k0 + 2u * j], d);
    d = fma(codes[b >> 4u], x[k0 + 2u * j + 1u], d);
  }
#endif
}

inline void i2_plain_row(const device uchar* w, const device float* x, uint k0, thread float& d) {
#ifdef I2_MXFP8
  const ulong a[4] = { *(const device ulong*)(w), *(const device ulong*)(w + 8),
                       *(const device ulong*)(w + 16), *(const device ulong*)(w + 24) };
  for (uint q = 0; q < 4u; ++q) {
    for (uint j = 0; j < 8u; ++j) d = fma(i2_e4m3(uchar(i2_lane_byte(a[q], j))), x[k0 + q * 8u + j], d);
  }
#else
  const ulong a = *(const device ulong*)(w);
  for (uint j = 0; j < 8u; ++j) {
    const uint b = i2_lane_byte(a, j);
    d = fma(i2_e2m1(uchar(b & 15u)), x[k0 + 2u * j], d);
    d = fma(i2_e2m1(uchar(b >> 4u)), x[k0 + 2u * j + 1u], d);
  }
#endif
}

[[kernel]] void i2_gu_table(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2_moe& a [[buffer(7)]], device float* ACT [[buffer(8)]],
    const device float* CODES [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x;
  const uint slot = tg.y;
  if (c >= a.n) return;
  const uint e = uint(PICKS[slot]);
  const size_t row = (size_t)e * a.rows + 2u * c;
  const device uchar* wg = W + row * I2_CODE_STRIDE(a.k);
  const device uchar* wu = wg + I2_CODE_STRIDE(a.k);
  const uint groups = a.k / I2_GROUP;
  float dg = 0.0f;
  float du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2_GROUP;
    float pg = 0.0f;
    float pu = 0.0f;
    i2_table_pair(wg + I2_BYTE_OFF(k0), wu + I2_BYTE_OFF(k0), X, k0, CODES, pg, pu);
    dg = fma(i2_decode_scale(S, row * I2_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(i2_decode_scale(S, (row + 1u) * I2_SCALE_STRIDE(a.k) + g), pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
#ifdef I2_MXFP8
    ACT[(size_t)slot * a.n + c] = i2_situ(dg, du, a.gate_cap, a.up_cap);
#else
    const float global = amax[e] / den[0];
    ACT[(size_t)slot * a.n + c] = i2_situ(dg * global, du * global, a.gate_cap, a.up_cap);
#endif
  }
}

[[kernel]] void i2_down_table(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2_moe& a [[buffer(7)]], device float* OUT [[buffer(8)]],
    const device float* CODES [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  const uint slot = tg.y;
  if (r >= a.n) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = ACT + (size_t)slot * a.k;
  const size_t row = (size_t)e * a.rows + r;
  const device uchar* w = W + row * I2_CODE_STRIDE(a.k);
  const uint groups = a.k / I2_GROUP;
  float d = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2_GROUP;
    float p = 0.0f;
    i2_table_row(w + I2_BYTE_OFF(k0), x, k0, CODES, p);
    d = fma(i2_decode_scale(S, row * I2_SCALE_STRIDE(a.k) + g), p, d);
  }
  d = simd_sum(d);
  if (lane == 0u) {
#ifdef I2_MXFP8
    OUT[(size_t)slot * a.n + r] = d;
#else
    OUT[(size_t)slot * a.n + r] = d * (amax[e] / den[0]);
#endif
  }
}

[[kernel]] void i2_down_thread(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2_moe& a [[buffer(7)]], device float* OUT [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint r = tg.x * 64u + tid.x;
  const uint slot = tg.y;
  if (r >= a.n) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = ACT + (size_t)slot * a.k;
  const size_t row = (size_t)e * a.rows + r;
  const device uchar* w = W + row * I2_CODE_STRIDE(a.k);
  const uint groups = a.k / I2_GROUP;
  float d = 0.0f;
  for (uint g = 0; g < groups; ++g) {
    const uint k0 = g * I2_GROUP;
    float p = 0.0f;
    i2_plain_row(w + I2_BYTE_OFF(k0), x, k0, p);
    d = fma(i2_decode_scale(S, row * I2_SCALE_STRIDE(a.k) + g), p, d);
  }
#ifdef I2_MXFP8
  OUT[(size_t)slot * a.n + r] = d;
#else
  OUT[(size_t)slot * a.n + r] = d * (amax[e] / den[0]);
#endif
}

#ifdef I2T_ROUTED
#define I2T_EXPERT_ARGS const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]], \
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]], const device float* den [[buffer(4)]], \
    const device int* PICKS [[buffer(5)]], constant i2t_moe& a [[buffer(6)]], device float* Y [[buffer(7)]], \
    const device float* CODES [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

template <uint Q>
inline uint i2e_run(const device int* PICKS, uint slots, uint slot, uint lane, thread uint (&member)[8]) {
  const int e = PICKS[slot];
  uint mask[Q], rank = 0u;
#pragma unroll
  for (uint q = 0; q < Q; ++q) {
    const uint at = lane + 32u * q, base = 32u * q;
    mask[q] = uint(ulong(simd_vote::vote_t(simd_ballot(at < slots && PICKS[at] == e))));
    const uint below = slot >= base + 32u ? 0xffffffffu : (slot <= base ? 0u : (1u << (slot - base)) - 1u);
    rank += popcount(mask[q] & below);
    mask[q] &= ~below;
  }
  if (rank % 8u != 0u) return 0u;
  uint count = 0u;
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    uint s = 32u * Q;
#pragma unroll
    for (uint q = 0; q < Q; ++q) {
      if (s == 32u * Q && mask[q] != 0u) {
        s = 32u * q + ctz(mask[q]);
        mask[q] &= mask[q] - 1u;
      }
    }
    member[i] = s;
    count += s < 32u * Q ? 1u : 0u;
  }
  return count;
}

template <bool GU, uint R, uint PER>
inline void i2e_scalar(const device uchar* We, const device uchar* Se, uint first, uint stride, uint groups,
    const device float* const (&x)[R], const thread uint (&slot)[R], threadgroup const float* lut,
    threadgroup const float* scales, uint lane, float global, constant i2t_moe& a, device float* Y) {
  constexpr uint U = GU ? 2u : 1u;
  float d[PER][U][R] = {};
  size_t ra[PER];
#pragma unroll
  for (uint j = 0; j < PER; ++j) ra[j] = U * min(first + j, a.n - 1u);
  for (uint g = lane; g < groups; g += 32u) {
    ulong word[PER][U];
    float p[PER][U][R] = {};
#pragma unroll
    for (uint j = 0; j < PER; ++j) {
#pragma unroll
      for (uint u = 0; u < U; ++u) word[j][u] = *(const device ulong*)(We + (ra[j] + u) * stride + g * 8u);
    }
#pragma unroll
    for (uint q = 0; q < 4u; ++q) {
      float4 v[R];
#pragma unroll
      for (uint r = 0; r < R; ++r) v[r] = *reinterpret_cast<const device float4*>(x[r] + g * 16u + 4u * q);
#pragma unroll
      for (uint j = 0; j < PER; ++j) {
#pragma unroll
        for (uint u = 0; u < U; ++u) {
          const uint b = uint(word[j][u] >> (16u * q)) & 0xffffu;
          const float c0 = lut[b & 15u], c1 = lut[(b >> 4u) & 15u], c2 = lut[(b >> 8u) & 15u], c3 = lut[b >> 12u];
#pragma unroll
          for (uint r = 0; r < R; ++r) p[j][u][r] = fma(c3, v[r].w, fma(c2, v[r].z, fma(c1, v[r].y, fma(c0, v[r].x, p[j][u][r]))));
        }
      }
    }
#pragma unroll
    for (uint j = 0; j < PER; ++j) {
#pragma unroll
      for (uint u = 0; u < U; ++u) {
        const float scale = scales[Se[(ra[j] + u) * groups + g]];
#pragma unroll
        for (uint r = 0; r < R; ++r) d[j][u][r] = fma(scale, p[j][u][r], d[j][u][r]);
      }
    }
  }
#pragma unroll
  for (uint j = 0; j < PER; ++j) {
    if (first + j >= a.n) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float s0 = simd_sum(d[j][0][r]);
      const float s1 = GU ? simd_sum(d[j][U - 1u][r]) : 0.0f;
      if (lane == 0u) Y[(size_t)slot[r] * a.n + first + j] = GU ? i2_situ(s0 * global, s1 * global, a.gate_cap, a.up_cap) : s0 * global;
    }
  }
}

template <uint R>
inline void i2e_pair(const device uchar* We, size_t ra, uint stride, uint groups, uint g,
    const device float* const (&x)[R], threadgroup const float* lut, threadgroup const float* scales,
    const device uchar* Se, thread float (&dg)[R], thread float (&du)[R]) {
  const uint k0 = g * 16u;
  const ulong ag = *(const device ulong*)(We + ra * stride + g * 8u);
  const ulong au = *(const device ulong*)(We + (ra + 1u) * stride + g * 8u);
  float pg[R] = {}, pu[R] = {};
  for (uint j = 0; j < 8u; ++j) {
    const uint ab = uint(ag >> (8u * j)) & 255u, ub = uint(au >> (8u * j)) & 255u;
    const float cg0 = lut[ab & 15u], cg1 = lut[ab >> 4u], cu0 = lut[ub & 15u], cu1 = lut[ub >> 4u];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float x0 = x[r][k0 + 2u * j], x1 = x[r][k0 + 2u * j + 1u];
      pg[r] = fma(cg1, x1, fma(cg0, x0, pg[r]));
      pu[r] = fma(cu1, x1, fma(cu0, x0, pu[r]));
    }
  }
  const float sg = scales[Se[ra * groups + g]], su = scales[Se[(ra + 1u) * groups + g]];
#pragma unroll
  for (uint r = 0; r < R; ++r) dg[r] = fma(sg, pg[r], dg[r]), du[r] = fma(su, pu[r], du[r]);
}

template <bool GU, uint R, uint PER>
inline void i2e_held(const device uchar* We, const device uchar* Se, uint first, uint stride, uint groups,
    const device float* const (&x)[R], const thread uint (&slot)[R], threadgroup const float* lut,
    threadgroup const float* scales, uint lane, float global, constant i2t_moe& a, device float* Y) {
  if (GU) {
    for (uint j = 0; j < PER; ++j) {
      const uint unit = first + j;
      if (unit >= a.n) break;
      float dg[R] = {}, du[R] = {};
      for (uint g = lane; g < groups; g += 32u) i2e_pair<R>(We, 2u * unit, stride, groups, g, x, lut, scales, Se, dg, du);
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        const float gsum = simd_sum(dg[r]), usum = simd_sum(du[r]);
        if (lane == 0u) Y[(size_t)slot[r] * a.n + unit] = i2_situ(gsum * global, usum * global, a.gate_cap, a.up_cap);
      }
    }
    return;
  }
  size_t ra[PER];
  float d[PER][R] = {};
#pragma unroll
  for (uint j = 0; j < PER; ++j) ra[j] = min(first + j, a.n - 1u);
  for (uint g = lane; g < groups; g += 32u) {
    float xv[R][16];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
#pragma unroll
      for (uint i = 0; i < 4u; ++i) {
        const float4 q = *reinterpret_cast<const device float4*>(x[r] + g * 16u + 4u * i);
        xv[r][4u * i] = q.x, xv[r][4u * i + 1u] = q.y, xv[r][4u * i + 2u] = q.z, xv[r][4u * i + 3u] = q.w;
      }
    }
#pragma unroll
    for (uint j = 0; j < PER; ++j) {
      const ulong word = *(const device ulong*)(We + ra[j] * stride + g * 8u);
      float p[R] = {};
#pragma unroll
      for (uint i = 0; i < 8u; ++i) {
        const uint byte = uint(word >> (8u * i)) & 255u;
#pragma unroll
        for (uint r = 0; r < R; ++r) p[r] = fma(lut[byte >> 4u], xv[r][2u * i + 1u], fma(lut[byte & 15u], xv[r][2u * i], p[r]));
      }
      const float scale = scales[Se[ra[j] * groups + g]];
#pragma unroll
      for (uint r = 0; r < R; ++r) d[j][r] = fma(scale, p[r], d[j][r]);
    }
  }
#pragma unroll
  for (uint j = 0; j < PER; ++j) {
    if (first + j >= a.n) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float v = simd_sum(d[j][r]);
      if (lane == 0u) Y[(size_t)slot[r] * a.n + first + j] = v * global;
    }
  }
}

template <bool GU, uint SG, uint T, uint R, bool HELD>
inline void i2e_few(const device float* X, const device uchar* We, const device uchar* Se, const thread uint (&member)[8],
    uint r0, uint sg, threadgroup const float* lut, threadgroup const float* scales, uint lane, float global,
    constant i2t_moe& a, device float* Y) {
  constexpr uint PER = (GU ? 4u : 8u) * T / SG;
  const device float* x[R];
  uint slot[R];
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    slot[r] = member[r];
    x[r] = X + (size_t)(GU ? member[r] / a.topk : member[r]) * a.k;
  }
  const uint first = (GU ? r0 / 2u : r0) + sg * PER;
  if (HELD) i2e_held<GU, R, PER>(We, Se, first, a.k / 2u, a.k / 16u, x, slot, lut, scales, lane, global, a, Y);
  else i2e_scalar<GU, R, PER>(We, Se, first, a.k / 2u, a.k / 16u, x, slot, lut, scales, lane, global, a, Y);
}

template <bool GU, uint SG, uint T>
inline void i2e_tiles(const device float* X, const device uchar* We, const device uchar* Se, threadgroup const float* lut,
    threadgroup const float* scales, threadgroup float2* part, const thread uint (&member)[8], uint count, uint r0,
    uint wrows, uint groups, uint stride, float global, constant i2t_moe& a, device float* Y, uint sg, uint lane) {
  constexpr uint CL = 32u / SG;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const bool live0 = fn < count, live1 = fn + 1u < count;
  const device float* x0 = X + (size_t)(live0 ? (GU ? member[fn] / a.topk : member[fn]) : 0u) * a.k;
  const device float* x1 = X + (size_t)(live1 ? (GU ? member[fn + 1u] / a.topk : member[fn + 1u]) : 0u) * a.k;
  float2 acc[CL][T] = {};
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(r0 + t * 8u + fm, wrows - 1u);
  const uint nb = fn >> 1;
  for (uint base = sg * CL; base < groups; base += 32u) {
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint gi = base + l;
      if (gi >= groups) break;
      const uint k0 = gi * 16u + fm;
      const float2 b0 = float2(live0 ? x0[k0] : 0.0f, live1 ? x1[k0] : 0.0f);
      const float2 b1 = float2(live0 ? x0[k0 + 8u] : 0.0f, live1 ? x1[k0 + 8u] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const ulong word = *(const device ulong*)(We + row[t] * stride + gi * 8u);
        const uint c0 = uint(word >> (8u * nb)) & 255u, c1 = uint(word >> (8u * (4u + nb))) & 255u;
        const float2 a0 = float2(lut[c0 & 15u], lut[c0 >> 4u]), a1 = float2(lut[c1 & 15u], lut[c1 >> 4u]);
        float2 p = float2(0.0f);
        i2t_mma(p, a0, b0);
        i2t_mma(p, a1, b1);
        acc[l][t] = fma(float2(scales[Se[row[t] * groups + gi]]), p, acc[l][t]);
      }
    }
  }
  i2t_partials<SG, CL, T>(acc, part, sg, lane);
  if (sg != 0u) return;
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint r = r0 + t * 8u + fm;
    const float2 sum = i2t_total<SG, T>(part, t, lane);
    const size_t y0 = (size_t)member[fn] * a.n, y1 = (size_t)member[fn + 1u] * a.n;
    if (GU) {
      const float2 up = float2(simd_shuffle_xor(sum.x, 2u), simd_shuffle_xor(sum.y, 2u));
      if ((fm & 1u) == 0u && r < wrows) {
        if (live0) Y[y0 + r / 2u] = i2_situ(sum.x * global, up.x * global, a.gate_cap, a.up_cap);
        if (live1) Y[y1 + r / 2u] = i2_situ(sum.y * global, up.y * global, a.gate_cap, a.up_cap);
      }
    } else if (r < wrows) {
      if (live0) Y[y0 + r] = sum.x * global;
      if (live1) Y[y1 + r] = sum.y * global;
    }
  }
}

template <bool GU, uint SG, uint T, uint Q, uint FEW, bool ONE_HELD, bool FEW_HELD>
[[kernel]] void i2e_routed(I2T_EXPERT_ARGS) {
  threadgroup float lut[16];
  threadgroup float scales[256];
  threadgroup float2 part[SG * T * 32];
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  uint member[8];
  const uint count = i2e_run<Q>(PICKS, a.slots, slot, lane, member);
  if (count == 0u) return;
  if (tid < 16u) lut[tid] = CODES[tid];
  for (uint i = tid; i < 256u; i += 32u * SG) scales[i] = i2_e4m3(uchar(i));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = PICKS[slot];
  const uint wrows = GU ? 2u * a.n : a.n;
  const uint r0 = tg.x * 8u * T;
  const float global = amax[e] / den[0];
  const device uchar* We = W + (size_t)uint(e) * wrows * (a.k / 2u);
  const device uchar* Se = S + (size_t)uint(e) * wrows * (a.k / 16u);
  if (count == 1u) return i2e_few<GU, SG, T, 1, ONE_HELD>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  if (FEW >= 2u && count == 2u) return i2e_few<GU, SG, T, 2, FEW_HELD>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  if (FEW >= 3u && count == 3u) return i2e_few<GU, SG, T, 3, FEW_HELD>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  i2e_tiles<GU, SG, T>(X, We, Se, lut, scales, part, member, count, r0, wrows, a.k / 16u, a.k / 2u, global, a, Y, sg, lane);
}

#define I2E_ROUTED(name, gu, sgs, t, q, few, one, held) \
  template [[host_name(#name)]] [[kernel]] void i2e_routed<gu, sgs, t, q, few, one, held>(I2T_EXPERT_ARGS);

I2E_ROUTED(i2e_gate_up, true, 4, 2, 2, 3, true, true)
I2E_ROUTED(i2e_gate_up16, true, 4, 2, 4, 3, false, false)
I2E_ROUTED(i2e_down1, false, 8, 4, 2, 1, true, true)
I2E_ROUTED(i2e_down, false, 8, 4, 2, 3, false, false)
I2E_ROUTED(i2e_down16, false, 8, 4, 4, 3, false, false)

inline float i2e_aff4_situ(float gate, float up, float gate_cap, float up_cap) {
  const float g = gate_cap * precise::tanh(gate / gate_cap) / (1.0f + precise::exp(-gate));
  const float u = up_cap > 0.0f ? up_cap * precise::tanh(up / up_cap) : up;
  return g * u;
}

#define I2E_AFF4_ARGS const device float* X [[buffer(0)]], const device uint* W [[buffer(1)]], \
    const device bfloat* S [[buffer(2)]], const device bfloat* B [[buffer(3)]], \
    const device int* PICKS [[buffer(4)]], constant i2t_moe& a [[buffer(5)]], device float* Y [[buffer(6)]], \
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]]

template <bool GU, uint SG, uint T, uint Q>
[[kernel]] void i2e_aff4_routed(I2E_AFF4_ARGS) {
  constexpr uint CL = 32u / SG;
  threadgroup float2 part[SG * T * 32];
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  uint member[8];
  const uint count = i2e_run<Q>(PICKS, a.slots, slot, lane, member);
  if (count == 0u) return;
  const uint e = uint(PICKS[slot]);
  const uint wrows = GU ? 2u * a.n : a.n;
  const uint r0 = tg.x * 8u * T;
  const uint words = a.k / 8u, groups = a.k / 64u;
  const device uint* We = W + (size_t)e * wrows * words;
  const device bfloat* Se = S + (size_t)e * wrows * groups;
  const device bfloat* Be = B + (size_t)e * wrows * groups;
  if (count <= 2u) {
    constexpr uint PER = (GU ? 4u : 8u) * T / SG;
    const uint first = (GU ? r0 / 2u : r0) + sg * PER;
    const device float* x0 = X + (size_t)(GU ? member[0] / a.topk : member[0]) * a.k;
    const device float* x1 = count > 1u ? X + (size_t)(GU ? member[1] / a.topk : member[1]) * a.k : x0;
    const uint4 shift = uint4(0u, 4u, 8u, 12u);
    for (uint j = 0; j < PER; ++j) {
      const uint col = first + j;
      if (col >= a.n) break;
      if (GU) {
        float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
        const size_t ra0 = 2u * col, ra1 = 2u * col + 1u;
        for (uint w = lane; w < words; w += 32u) {
          const uint g = w / 8u;
          const uint pg = We[ra0 * words + w], pu = We[ra1 * words + w];
          const float sg0 = float(Se[ra0 * groups + g]), bg0 = float(Be[ra0 * groups + g]);
          const float su0 = float(Se[ra1 * groups + g]), bu0 = float(Be[ra1 * groups + g]);
          const float4 vg0 = fma(float4((uint4(pg) >> shift) & 15u), float4(sg0), float4(bg0));
          const float4 vg1 = fma(float4((uint4(pg >> 16u) >> shift) & 15u), float4(sg0), float4(bg0));
          const float4 vu0 = fma(float4((uint4(pu) >> shift) & 15u), float4(su0), float4(bu0));
          const float4 vu1 = fma(float4((uint4(pu >> 16u) >> shift) & 15u), float4(su0), float4(bu0));
          const float4 i00 = float4(*reinterpret_cast<const device packed_float4*>(x0 + w * 8u));
          const float4 i01 = float4(*reinterpret_cast<const device packed_float4*>(x0 + w * 8u + 4u));
          ag0 = fma(i00.w, vg0.w, fma(i00.z, vg0.z, fma(i00.y, vg0.y, fma(i00.x, vg0.x, ag0))));
          ag0 = fma(i01.w, vg1.w, fma(i01.z, vg1.z, fma(i01.y, vg1.y, fma(i01.x, vg1.x, ag0))));
          au0 = fma(i00.w, vu0.w, fma(i00.z, vu0.z, fma(i00.y, vu0.y, fma(i00.x, vu0.x, au0))));
          au0 = fma(i01.w, vu1.w, fma(i01.z, vu1.z, fma(i01.y, vu1.y, fma(i01.x, vu1.x, au0))));
          if (count > 1u) {
            const float4 i10 = float4(*reinterpret_cast<const device packed_float4*>(x1 + w * 8u));
            const float4 i11 = float4(*reinterpret_cast<const device packed_float4*>(x1 + w * 8u + 4u));
            ag1 = fma(i10.w, vg0.w, fma(i10.z, vg0.z, fma(i10.y, vg0.y, fma(i10.x, vg0.x, ag1))));
            ag1 = fma(i11.w, vg1.w, fma(i11.z, vg1.z, fma(i11.y, vg1.y, fma(i11.x, vg1.x, ag1))));
            au1 = fma(i10.w, vu0.w, fma(i10.z, vu0.z, fma(i10.y, vu0.y, fma(i10.x, vu0.x, au1))));
            au1 = fma(i11.w, vu1.w, fma(i11.z, vu1.z, fma(i11.y, vu1.y, fma(i11.x, vu1.x, au1))));
          }
        }
        ag0 = simd_sum(ag0);
        au0 = simd_sum(au0);
        if (lane == 0u) Y[(size_t)member[0] * a.n + col] = i2e_aff4_situ(ag0, au0, a.gate_cap, a.up_cap);
        if (count > 1u) {
          ag1 = simd_sum(ag1);
          au1 = simd_sum(au1);
          if (lane == 0u) Y[(size_t)member[1] * a.n + col] = i2e_aff4_situ(ag1, au1, a.gate_cap, a.up_cap);
        }
      } else {
        float a0 = 0.0f, a1 = 0.0f;
        for (uint w = lane; w < words; w += 32u) {
          const uint g = w / 8u;
          const uint p = We[(size_t)col * words + w];
          const float sc = float(Se[(size_t)col * groups + g]), bi = float(Be[(size_t)col * groups + g]);
          const float4 v0 = fma(float4((uint4(p) >> shift) & 15u), float4(sc), float4(bi));
          const float4 v1 = fma(float4((uint4(p >> 16u) >> shift) & 15u), float4(sc), float4(bi));
          const float4 i00 = float4(*reinterpret_cast<const device packed_float4*>(x0 + w * 8u));
          const float4 i01 = float4(*reinterpret_cast<const device packed_float4*>(x0 + w * 8u + 4u));
          a0 = fma(i00.w, v0.w, fma(i00.z, v0.z, fma(i00.y, v0.y, fma(i00.x, v0.x, a0))));
          a0 = fma(i01.w, v1.w, fma(i01.z, v1.z, fma(i01.y, v1.y, fma(i01.x, v1.x, a0))));
          if (count > 1u) {
            const float4 i10 = float4(*reinterpret_cast<const device packed_float4*>(x1 + w * 8u));
            const float4 i11 = float4(*reinterpret_cast<const device packed_float4*>(x1 + w * 8u + 4u));
            a1 = fma(i10.w, v0.w, fma(i10.z, v0.z, fma(i10.y, v0.y, fma(i10.x, v0.x, a1))));
            a1 = fma(i11.w, v1.w, fma(i11.z, v1.z, fma(i11.y, v1.y, fma(i11.x, v1.x, a1))));
          }
        }
        a0 = simd_sum(a0);
        if (lane == 0u) Y[(size_t)member[0] * a.n + col] = a0;
        if (count > 1u) {
          a1 = simd_sum(a1);
          if (lane == 0u) Y[(size_t)member[1] * a.n + col] = a1;
        }
      }
    }
    return;
  }
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const bool live0 = fn < count, live1 = fn + 1u < count;
  const device float* x0 = X + (size_t)(live0 ? (GU ? member[fn] / a.topk : member[fn]) : 0u) * a.k;
  const device float* x1 = X + (size_t)(live1 ? (GU ? member[fn + 1u] / a.topk : member[fn + 1u]) : 0u) * a.k;
  float2 acc[CL][T] = {};
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(r0 + t * 8u + fm, wrows - 1u);
  const uint shift = 4u * fn;
  for (uint base = sg * CL; base < words; base += 32u) {
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint w = base + l;
      if (w >= words) break;
      const uint g = w / 8u;
      const uint kk = w * 8u + fm;
      const float2 bx = float2(live0 ? x0[kk] : 0.0f, live1 ? x1[kk] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const uint packed = We[row[t] * words + w];
        const float sc = float(Se[row[t] * groups + g]), bi = float(Be[row[t] * groups + g]);
        const float2 code = float2(float((packed >> shift) & 15u), float((packed >> (shift + 4u)) & 15u));
        i2t_mma(acc[l][t], fma(code, float2(sc), float2(bi)), bx);
      }
    }
  }
  i2t_partials<SG, CL, T>(acc, part, sg, lane);
  if (sg != 0u) return;
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint r = r0 + t * 8u + fm;
    const float2 sum = i2t_total<SG, T>(part, t, lane);
    const size_t y0 = (size_t)member[fn] * a.n, y1 = (size_t)member[fn + 1u] * a.n;
    if (GU) {
      const float2 up = float2(simd_shuffle_xor(sum.x, 2u), simd_shuffle_xor(sum.y, 2u));
      if ((fm & 1u) == 0u && r < wrows) {
        if (live0) Y[y0 + r / 2u] = i2e_aff4_situ(sum.x, up.x, a.gate_cap, a.up_cap);
        if (live1) Y[y1 + r / 2u] = i2e_aff4_situ(sum.y, up.y, a.gate_cap, a.up_cap);
      }
    } else if (r < wrows) {
      if (live0) Y[y0 + r] = sum.x;
      if (live1) Y[y1 + r] = sum.y;
    }
  }
}

#define I2E_AFF4(name, gu, sgs, t, q) \
  template [[host_name(#name)]] [[kernel]] void i2e_aff4_routed<gu, sgs, t, q>(I2E_AFF4_ARGS);

I2E_AFF4(i2e_aff4_up, true, 4, 2, 2)
I2E_AFF4(i2e_aff4_up16, true, 4, 2, 4)
I2E_AFF4(i2e_aff4_down, false, 8, 4, 2)
I2E_AFF4(i2e_aff4_down16, false, 8, 4, 4)
#endif
