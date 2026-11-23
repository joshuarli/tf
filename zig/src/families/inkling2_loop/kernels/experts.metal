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
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
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

#ifndef I2_MXFP8
struct i2_rows_moe { uint k, n, slots, topk; float gate_cap, up_cap; };

// The slots routed to expert `e`, lowest first, at most eight: one a batch row, since a row picks an expert once.
inline uint i2_members(const device int* PICKS, uint slots, int e, uint lane, thread uint (&member)[8]) {
  const bool lo = lane < slots && PICKS[lane] == e;
  const bool hi = lane + 32u < slots && PICKS[lane + 32u] == e;
  uint ml = uint(ulong(simd_vote::vote_t(simd_ballot(lo))));
  uint mh = uint(ulong(simd_vote::vote_t(simd_ballot(hi))));
  uint count = 0u;
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    uint s = 64u;
    if (ml != 0u) {
      s = ctz(ml);
      ml &= ml - 1u;
    } else if (mh != 0u) {
      s = 32u + ctz(mh);
      mh &= mh - 1u;
    }
    member[i] = s;
    count += s < 64u ? 1u : 0u;
  }
  return count;
}

// i2p_gu for a batch: the lowest slot routed to an expert decodes each gate/up group once for every slot routed to
// it, each slot's sums in i2p_gu's order; the expert's other slots return at once.
[[kernel]] void i2_gu_rows(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2_rows_moe& a [[buffer(6)]], device float* ACT [[buffer(7)]],
    const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x;
  const uint slot = tg.y;
  if (c >= a.n || slot >= a.slots) return;
  const int e = PICKS[slot];
  uint member[8];
  const uint count = i2_members(PICKS, a.slots, e, lane, member);
  if (member[0] != slot) return;
  const size_t row = (size_t)uint(e) * a.n * 2u + 2u * c;
  const device uchar* wg = W + row * I2_CODE_STRIDE(a.k);
  const device uchar* wu = wg + I2_CODE_STRIDE(a.k);
  const uint groups = a.k / I2_GROUP;
  float dg[8], du[8];
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    dg[i] = 0.0f;
    du[i] = 0.0f;
  }
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2_GROUP;
    const ulong ag = *(const device ulong*)(wg + I2_BYTE_OFF(k0));
    const ulong au = *(const device ulong*)(wu + I2_BYTE_OFF(k0));
    const float sg = i2_decode_scale(S, row * I2_SCALE_STRIDE(a.k) + g);
    const float su = i2_decode_scale(S, (row + 1u) * I2_SCALE_STRIDE(a.k) + g);
    float cg[16], cu[16];
#pragma unroll
    for (uint j = 0; j < 8u; ++j) {
      const uint ab = i2_lane_byte(ag, j);
      const uint ub = i2_lane_byte(au, j);
      cg[2u * j] = CODES[ab & 15u];
      cg[2u * j + 1u] = CODES[ab >> 4u];
      cu[2u * j] = CODES[ub & 15u];
      cu[2u * j + 1u] = CODES[ub >> 4u];
    }
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
      if (i >= count) break;
      const device float* x = X + (size_t)(member[i] / a.topk) * a.k;
      float pg = 0.0f;
      float pu = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8u; ++j) {
        const float x0 = x[k0 + 2u * j];
        const float x1 = x[k0 + 2u * j + 1u];
        pg = fma(cg[2u * j], x0, pg);
        pg = fma(cg[2u * j + 1u], x1, pg);
        pu = fma(cu[2u * j], x0, pu);
        pu = fma(cu[2u * j + 1u], x1, pu);
      }
      dg[i] = fma(sg, pg, dg[i]);
      du[i] = fma(su, pu, du[i]);
    }
  }
  const float global = amax[e] / den[0];
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    if (i >= count) break;
    const float g = simd_sum(dg[i]);
    const float u = simd_sum(du[i]);
    if (lane == 0u) ACT[(size_t)member[i] * a.n + c] = i2_situ(g * global, u * global, a.gate_cap, a.up_cap);
  }
}

// i2p_down for a batch, shared the same way: each slot keeps its own activation row and i2p_down's sums.
[[kernel]] void i2_down_rows(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2_rows_moe& a [[buffer(6)]], device float* OUT [[buffer(7)]],
    const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  const uint slot = tg.y;
  if (r >= a.n || slot >= a.slots) return;
  const int e = PICKS[slot];
  uint member[8];
  const uint count = i2_members(PICKS, a.slots, e, lane, member);
  if (member[0] != slot) return;
  const size_t row = (size_t)uint(e) * a.n + r;
  const device uchar* w = W + row * I2_CODE_STRIDE(a.k);
  const uint groups = a.k / I2_GROUP;
  float d[8];
#pragma unroll
  for (uint i = 0; i < 8u; ++i) d[i] = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2_GROUP;
    const ulong aw = *(const device ulong*)(w + I2_BYTE_OFF(k0));
    const float sw = i2_decode_scale(S, row * I2_SCALE_STRIDE(a.k) + g);
    float cw[16];
#pragma unroll
    for (uint j = 0; j < 8u; ++j) {
      const uint b = i2_lane_byte(aw, j);
      cw[2u * j] = CODES[b & 15u];
      cw[2u * j + 1u] = CODES[b >> 4u];
    }
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
      if (i >= count) break;
      const device float* x = ACT + (size_t)member[i] * a.k;
      float p = 0.0f;
#pragma unroll
      for (uint j = 0; j < 8u; ++j) {
        p = fma(cw[2u * j], x[k0 + 2u * j], p);
        p = fma(cw[2u * j + 1u], x[k0 + 2u * j + 1u], p);
      }
      d[i] = fma(sw, p, d[i]);
    }
  }
  const float global = amax[e] / den[0];
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    if (i >= count) break;
    const float v = simd_sum(d[i]);
    if (lane == 0u) OUT[(size_t)member[i] * a.n + r] = v * global;
  }
}
#endif
