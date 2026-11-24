#include <metal_stdlib>
using namespace metal;

struct i2g_moe { uint k, n, slots, topk; float gate_cap, up_cap; };

inline float i2g_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

inline float i2g_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

inline uint i2g_members(const device int* PICKS, uint slots, int e, uint lane, thread uint (&member)[8]) {
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

inline void i2g_load16(const device float* x, thread float (&v)[16]) {
#pragma unroll
  for (uint i = 0; i < 4u; ++i) {
    const float4 q = float4(*reinterpret_cast<const device packed_float4*>(x + 4u * i));
    v[4u * i] = q.x;
    v[4u * i + 1u] = q.y;
    v[4u * i + 2u] = q.z;
    v[4u * i + 3u] = q.w;
  }
}

inline void i2g_decode(ulong w, threadgroup const float* lut, thread float (&c)[16]) {
#pragma unroll
  for (uint j = 0; j < 8u; ++j) {
    const uint b = uint(w >> (8u * j)) & 255u;
    c[2u * j] = lut[b & 15u];
    c[2u * j + 1u] = lut[b >> 4u];
  }
}

inline float i2g_group(thread const float (&c)[16], thread const float (&x)[16]) {
  float p = 0.0f;
#pragma unroll
  for (uint j = 0; j < 8u; ++j) {
    p = fma(c[2u * j], x[2u * j], p);
    p = fma(c[2u * j + 1u], x[2u * j + 1u], p);
  }
  return p;
}

template <uint C>
[[kernel]] void i2g_down_all(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2g_moe& a [[buffer(6)]], device float* OUT [[buffer(7)]], const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lut[16];
  if (tid < 16u) lut[tid] = CODES[tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  const int e = PICKS[slot];
  uint member[8];
  const uint count = i2g_members(PICKS, a.slots, e, lane, member);
  const uint r0 = (tg.x * 8u + sg) * C;
  if (member[0] != slot || r0 >= a.n) return;
  const uint groups = a.k / 16u, stride = a.k / 2u, sstride = a.k / 16u;
  float d[C][8];
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint i = 0; i < 8u; ++i) d[c][i] = 0.0f;
  }
  for (uint g = lane; g < groups; g += 32u) {
    float code[C][16], sw[C];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t row = (size_t)uint(e) * a.n + min(r0 + c, a.n - 1u);
      i2g_decode(*(const device ulong*)(W + row * stride + g * 8u), lut, code[c]);
      sw[c] = i2g_e4m3(S[row * sstride + g]);
    }
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
      if (i >= count) break;
      float xv[16];
      i2g_load16(ACT + (size_t)member[i] * a.k + g * 16u, xv);
#pragma unroll
      for (uint c = 0; c < C; ++c) d[c][i] = fma(sw[c], i2g_group(code[c], xv), d[c][i]);
    }
  }
  const float global = amax[e] / den[0];
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (r0 + c >= a.n) break;
#pragma unroll
    for (uint i = 0; i < 8u; ++i) {
      if (i >= count) break;
      const float v = simd_sum(d[c][i]);
      if (lane == 0u) OUT[(size_t)member[i] * a.n + r0 + c] = v * global;
    }
  }
}


inline void i2g_codes(ulong w, threadgroup const float* lut, thread float (&c)[16]) {
  const uint lo = uint(w), hi = uint(w >> 32);
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    c[i] = lut[extract_bits(lo, 4u * i, 4u)];
    c[8u + i] = lut[extract_bits(hi, 4u * i, 4u)];
  }
}

inline void i2g_tables(const device float* CODES, uint tid, threadgroup float* lut, threadgroup float* scales) {
  if (tid < 16u) lut[tid] = CODES[tid];
  scales[tid] = i2g_e4m3(uchar(tid));
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint C>
[[kernel]] void i2g_gate_up(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2g_moe& a [[buffer(6)]], device float* ACT [[buffer(7)]], const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float lut[16];
  threadgroup float scales[256];
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  const int e = PICKS[slot];
  uint member[8];
  const uint count = i2g_members(PICKS, a.slots, e, lane, member);
  if (member[0] != slot) return;
  i2g_tables(CODES, tid, lut, scales);
  const uint c0 = (tg.x * 8u + sg) * C;
  if (c0 >= a.n) return;
  const uint groups = a.k / 16u, stride = a.k / 2u, sstride = a.k / 16u;
  const float global = amax[e] / den[0];
  for (uint p = 0; p < count; p += 2u) {
    const bool pair = p + 1u < count;
    const device float* x0 = X + (size_t)(member[p] / a.topk) * a.k;
    const device float* x1 = X + (size_t)(member[pair ? p + 1u : p] / a.topk) * a.k;
    float dg[C][2], du[C][2];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      dg[c][0] = 0.0f;
      dg[c][1] = 0.0f;
      du[c][0] = 0.0f;
      du[c][1] = 0.0f;
    }
    for (uint g = lane; g < groups; g += 32u) {
      float xa[16], xb[16], code[16];
      i2g_load16(x0 + g * 16u, xa);
      i2g_load16(x1 + g * 16u, xb);
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        const size_t row = (size_t)uint(e) * a.n * 2u + 2u * min(c0 + c, a.n - 1u);
        const ulong ag = *(const device ulong*)(W + row * stride + g * 8u);
        const ulong au = *(const device ulong*)(W + (row + 1u) * stride + g * 8u);
        const float sgate = scales[S[row * sstride + g]];
        const float sup = scales[S[(row + 1u) * sstride + g]];
        i2g_codes(ag, lut, code);
        dg[c][0] = fma(sgate, i2g_group(code, xa), dg[c][0]);
        if (pair) dg[c][1] = fma(sgate, i2g_group(code, xb), dg[c][1]);
        i2g_codes(au, lut, code);
        du[c][0] = fma(sup, i2g_group(code, xa), du[c][0]);
        if (pair) du[c][1] = fma(sup, i2g_group(code, xb), du[c][1]);
      }
    }
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      if (c0 + c >= a.n) break;
#pragma unroll
      for (uint m = 0; m < 2u; ++m) {
        if (m == 1u && !pair) break;
        const float g = simd_sum(dg[c][m]);
        const float u = simd_sum(du[c][m]);
        if (lane == 0u) ACT[(size_t)member[p + m] * a.n + c0 + c] = i2g_situ(g * global, u * global, a.gate_cap, a.up_cap);
      }
    }
  }
}

#define I2G_KERNEL(name, c) \
  template [[host_name(#name "_" #c)]] [[kernel]] void name<c>(const device float* X [[buffer(0)]], \
      const device uchar* W [[buffer(1)]], const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]], \
      const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]], constant i2g_moe& a [[buffer(6)]], \
      device float* Y [[buffer(7)]], const device float* CODES [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

I2G_KERNEL(i2g_gate_up, 1)
I2G_KERNEL(i2g_gate_up, 2)
I2G_KERNEL(i2g_down_all, 2)
