#include <metal_stdlib>
using namespace metal;

struct i2t_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2t_projection_args { uint k, full, rows; uint m[8]; };
struct i2t_moe { uint k, n, slots, topk; float gate_cap, up_cap; };

// Apple GPUs: lane l of a simdgroup matrix holds row fm and columns fn, fn + 1, as i2t_lane computes them.
inline void i2t_lane(uint lane, thread uint& fm, thread uint& fn) {
  const uint qid = lane >> 2;
  fm = (qid & 4u) | ((lane >> 1) & 3u);
  fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
}

inline thread float2& i2t_elements(thread simdgroup_matrix<float, 8, 8>& m) {
  return reinterpret_cast<thread float2&>(m.thread_elements());
}

// Apple GPUs: a float simdgroup multiply-accumulate is one fma chain over the inner index, in index order.
inline void i2t_mma(thread float2& c, float2 a, float2 b) {
  simdgroup_matrix<float, 8, 8> A, B, C, D;
  i2t_elements(A) = a;
  i2t_elements(B) = b;
  i2t_elements(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = i2t_elements(D);
}

template <uint N>
inline float2 i2t_tree(thread float2 (&v)[N]) {
#pragma unroll
  for (uint width = 1; width < N; width <<= 1)
#pragma unroll
    for (uint i = 0; i < N; i += 2u * width) v[i] = v[i] + v[i + width];
  return v[0];
}

template <uint SG, uint T>
inline void i2t_affine(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint row0, uint last, uint k, uint sg, uint lane, threadgroup float2 (&part)[SG][T][32],
    thread float2 (&out)[T]) {
  constexpr uint CL = 32u / SG;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const uint words = k / 4u, groups = k / 64u, chunks = words / 64u;
  const uint half_word = 32u * (fn >> 2), shift = 8u * (fn & 3u), half_k = 128u * (fm >> 2) + (fm & 3u);
  const bool live0 = fn < rows, live1 = fn + 1u < rows;
  const device float* x0 = x + (live0 ? fn : 0u) * step;
  const device float* x1 = x + (live1 ? fn + 1u : 0u) * step;
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(row0 + t * 8u + fm, last);
  float2 acc[CL][T];
#pragma unroll
  for (uint l = 0; l < CL; ++l)
#pragma unroll
    for (uint t = 0; t < T; ++t) acc[l][t] = float2(0.0f);
  for (uint c = 0; c < chunks; ++c) {
    const uint lead = sg * CL + 64u * c;
    const uint g = (lead + half_word) / 16u;
    float scale[T], bias[T];
#pragma unroll
    for (uint t = 0; t < T; ++t) {
      scale[t] = float(s[row[t] * groups + g]);
      bias[t] = float(b[row[t] * groups + g]);
    }
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint kk = 4u * (lead + l) + half_k;
      const float2 bx = float2(live0 ? x0[kk] : 0.0f, live1 ? x1[kk] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const uint packed = w[row[t] * words + lead + l + half_word];
        const float2 code = float2(float((packed >> shift) & 255u), float((packed >> (shift + 8u)) & 255u));
        i2t_mma(acc[l][t], fma(code, float2(scale[t]), float2(bias[t])), bx);
      }
    }
  }
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    float2 v[CL];
#pragma unroll
    for (uint l = 0; l < CL; ++l) v[l] = acc[l][t];
    part[sg][t][lane] = i2t_tree(v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    float2 v[SG];
#pragma unroll
    for (uint i = 0; i < SG; ++i) v[i] = part[i][t][lane];
    out[t] = i2t_tree(v);
  }
}

template <uint SG, uint T>
[[kernel]] void i2t_dense(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 part[SG][T][32];
  const uint row0 = group * 8u * T;
  float2 out[T];
  i2t_affine<SG, T>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, row0, a.n - 1u, a.k, sg, lane, part,
      out);
  if (sg != 0u) return;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint col = row0 + t * 8u + fm;
    if (col >= a.n) break;
    float2 v = out[t];
    if (a.cap > 0.0f) v = float2(a.cap * precise::tanh(v.x / a.cap), a.cap * precise::tanh(v.y / a.cap));
    if (fn < a.rows) y[(size_t)fn * a.n + col] = v.x;
    if (fn + 1u < a.rows) y[(size_t)(fn + 1u) * a.n + col] = v.y;
  }
}

template <uint SG, uint T>
[[kernel]] void i2t_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]],
    const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]],
    const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2t_projection_args& a [[buffer(20)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 part[SG][T][32];
  uint which = 0u, first = group;
  for (; which < 8u; ++which) {
    const uint span = (a.m[which] + 8u * T - 1u) / (8u * T);
    if (first < span) break;
    first -= span;
  }
  if (which == 8u) return;
  const bool key = which == 1u || which == 6u;
  const bool val = which == 2u || which == 7u;
  uint offset = 0u;
  if (which == 1u) offset = a.full ? 0u : a.m[0];
  if (which == 2u) offset = a.full ? 0u : a.m[0] + a.m[1];
  if (which == 3u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]);
  if (which == 4u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3];
  if (which == 5u) offset = a.m[0] + (a.full ? 0u : a.m[1] + a.m[2]) + a.m[3] + a.m[4];
  if (which == 6u) offset = a.m[1];
  if (which == 7u) offset = a.m[2];
  const uint m = a.m[which], row0 = first * 8u * T;
  float2 out[T];
  i2t_affine<SG, T>(key ? xk : val ? xv : xq, a.k, a.rows, a.full && key ? wk : a.full && val ? wv : wq,
      a.full && key ? sk : a.full && val ? sv : sq, a.full && key ? bk : a.full && val ? bv : bq, offset + row0,
      offset + m - 1u, a.k, sg, lane, part, out);
  if (sg != 0u) return;
  device float* y = which == 0u ? oq : which == 1u ? ok : which == 2u ? ov : which == 3u ? orl
      : which == 4u ? og : which == 5u ? os : which == 6u ? okd : ovd;
  uint fm, fn;
  i2t_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint row = row0 + t * 8u + fm;
    if (row >= m) break;
    if (fn < a.rows) y[(size_t)fn * m + row] = out[t].x;
    if (fn + 1u < a.rows) y[(size_t)(fn + 1u) * m + row] = out[t].y;
  }
}

#define I2T_DENSE(sgs, t) \
  template [[host_name("i2t_dense_" #sgs "_" #t)]] [[kernel]] void i2t_dense<sgs, t>( \
      const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], \
      const device bfloat* b [[buffer(3)]], constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], \
      uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

I2T_DENSE(8, 2)

template [[host_name("i2t_projections_8_2")]] [[kernel]] void i2t_projections<8, 2>(
    const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]], const device float* xv [[buffer(2)]],
    const device uint* wq [[buffer(3)]], const device bfloat* sq [[buffer(4)]], const device bfloat* bq [[buffer(5)]],
    const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]], const device bfloat* bk [[buffer(8)]],
    const device uint* wv [[buffer(9)]], const device bfloat* sv [[buffer(10)]], const device bfloat* bv [[buffer(11)]],
    device float* oq [[buffer(12)]], device float* ok [[buffer(13)]], device float* ov [[buffer(14)]],
    device float* orl [[buffer(15)]], device float* og [[buffer(16)]], device float* os [[buffer(17)]],
    device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]], constant i2t_projection_args& a [[buffer(20)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);

inline float i2t_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

inline float i2t_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

inline uint i2t_members(const device int* PICKS, uint slots, int e, uint lane, thread uint (&member)[8]) {
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

template <uint R>
inline void i2t_pair(const device uchar* We, size_t ra, uint stride, uint groups, uint g,
    const device float* const (&x)[R], threadgroup const float* lut, threadgroup const float* scales,
    const device uchar* Se, thread float (&dg)[R], thread float (&du)[R]) {
  const uint k0 = g * 16u;
  const ulong ag = *(const device ulong*)(We + ra * stride + g * 8u);
  const ulong au = *(const device ulong*)(We + (ra + 1u) * stride + g * 8u);
  float pg[R], pu[R];
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    pg[r] = 0.0f;
    pu[r] = 0.0f;
  }
  for (uint j = 0; j < 8u; ++j) {
    const uint ab = uint(ag >> (8u * j)) & 255u, ub = uint(au >> (8u * j)) & 255u;
    const float cg0 = lut[ab & 15u], cg1 = lut[ab >> 4u], cu0 = lut[ub & 15u], cu1 = lut[ub >> 4u];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float x0 = x[r][k0 + 2u * j], x1 = x[r][k0 + 2u * j + 1u];
      pg[r] = fma(cg0, x0, pg[r]);
      pg[r] = fma(cg1, x1, pg[r]);
      pu[r] = fma(cu0, x0, pu[r]);
      pu[r] = fma(cu1, x1, pu[r]);
    }
  }
  const float sg = scales[Se[ra * groups + g]], su = scales[Se[(ra + 1u) * groups + g]];
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    dg[r] = fma(sg, pg[r], dg[r]);
    du[r] = fma(su, pu[r], du[r]);
  }
}

template <uint R, uint PER>
inline void i2t_columns(const device uchar* We, uint first, uint n, uint stride, uint groups,
    const device float* const (&x)[R], threadgroup const float* lut, threadgroup const float* scales,
    const device uchar* Se, uint lane, float global, constant i2t_moe& a, device float* Y, const thread uint (&slot)[R]) {
  for (uint j = 0; j < PER; ++j) {
    const uint unit = first + j;
    if (unit >= n) break;
    float dg[R], du[R];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      dg[r] = 0.0f;
      du[r] = 0.0f;
    }
    for (uint g = lane; g < groups; g += 32u) i2t_pair<R>(We, 2u * unit, stride, groups, g, x, lut, scales, Se, dg, du);
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float gsum = simd_sum(dg[r]), usum = simd_sum(du[r]);
      if (lane == 0u) Y[(size_t)slot[r] * a.n + unit] = i2t_situ(gsum * global, usum * global, a.gate_cap, a.up_cap);
    }
  }
}

template <uint PER>
inline void i2t_rows(const device uchar* We, uint first, uint n, uint stride, uint groups, const device float* x,
    threadgroup const float* lut, threadgroup const float* scales, const device uchar* Se, uint lane,
    thread float (&d)[PER]) {
  size_t ra[PER];
#pragma unroll
  for (uint j = 0; j < PER; ++j) {
    ra[j] = min(first + j, n - 1u);
    d[j] = 0.0f;
  }
  for (uint g = lane; g < groups; g += 32u) {
    float xv[16];
#pragma unroll
    for (uint i = 0; i < 4u; ++i) {
      const float4 q = *reinterpret_cast<const device float4*>(x + g * 16u + 4u * i);
      xv[4u * i] = q.x;
      xv[4u * i + 1u] = q.y;
      xv[4u * i + 2u] = q.z;
      xv[4u * i + 3u] = q.w;
    }
#pragma unroll
    for (uint j = 0; j < PER; ++j) {
      const ulong word = *(const device ulong*)(We + ra[j] * stride + g * 8u);
      float p = 0.0f;
#pragma unroll
      for (uint i = 0; i < 8u; ++i) {
        const uint byte = uint(word >> (8u * i)) & 255u;
        p = fma(lut[byte & 15u], xv[2u * i], p);
        p = fma(lut[byte >> 4u], xv[2u * i + 1u], p);
      }
      d[j] = fma(scales[Se[ra[j] * groups + g]], p, d[j]);
    }
  }
}

template <bool GU, uint SG, uint T>
[[kernel]] void i2t_experts(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2t_moe& a [[buffer(6)]], device float* Y [[buffer(7)]], const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint CL = 32u / SG;
  threadgroup float lut[16];
  threadgroup float scales[256];
  threadgroup float2 part[SG][T][32];
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  const int e = PICKS[slot];
  uint member[8];
  const uint count = i2t_members(PICKS, a.slots, e, lane, member);
  if (member[0] != slot) return;
  if (tid < 16u) lut[tid] = CODES[tid];
  for (uint i = tid; i < 256u; i += 32u * SG) scales[i] = i2t_e4m3(uchar(i));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint wrows = GU ? 2u * a.n : a.n;
  const uint groups = a.k / 16u, stride = a.k / 2u;
  const uint r0 = tg.x * 8u * T;
  const float global = amax[e] / den[0];
  const device uchar* We = W + (size_t)uint(e) * wrows * stride;
  const device uchar* Se = S + (size_t)uint(e) * wrows * groups;
  if (GU && count == 2u) {
    constexpr uint per = 4u * T / SG;
    const device float* const x[2] = { X + (size_t)(member[0] / a.topk) * a.k, X + (size_t)(member[1] / a.topk) * a.k };
    const uint slots[2] = { member[0], member[1] };
    i2t_columns<2, per>(We, r0 / 2u + sg * per, a.n, stride, groups, x, lut, scales, Se, lane, global, a, Y, slots);
    return;
  }
  if (count == 1u) {
    const device float* x = X + (size_t)(GU ? slot / a.topk : slot) * a.k;
    if (GU) {
      constexpr uint per = 4u * T / SG;
      const device float* const xs[1] = { x };
      const uint slots[1] = { slot };
      i2t_columns<1, per>(We, r0 / 2u + sg * per, a.n, stride, groups, xs, lut, scales, Se, lane, global, a, Y, slots);
    } else {
      constexpr uint per = 8u * T / SG;
      const uint first = r0 + sg * per;
      float d[per];
      i2t_rows<per>(We, first, a.n, stride, groups, x, lut, scales, Se, lane, d);
#pragma unroll
      for (uint j = 0; j < per; ++j) {
        const float v = simd_sum(d[j]);
        if (lane == 0u && first + j < a.n) Y[(size_t)slot * a.n + first + j] = v * global;
      }
    }
    return;
  }
  uint fm, fn;
  i2t_lane(lane, fm, fn);
  const bool live0 = fn < count, live1 = fn + 1u < count;
  const device float* x0 = X + (size_t)(live0 ? (GU ? member[fn] / a.topk : member[fn]) : 0u) * a.k;
  const device float* x1 = X + (size_t)(live1 ? (GU ? member[fn + 1u] / a.topk : member[fn + 1u]) : 0u) * a.k;
  float2 acc[CL][T];
#pragma unroll
  for (uint l = 0; l < CL; ++l)
#pragma unroll
    for (uint t = 0; t < T; ++t) acc[l][t] = float2(0.0f);
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(r0 + t * 8u + fm, wrows - 1u);
  const uint nb = fn >> 1;
  for (uint base = sg * CL; base < groups; base += 32u) {
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint g = base + l;
      if (g >= groups) break;
      const uint k0 = g * 16u + fm;
      const float2 b0 = float2(live0 ? x0[k0] : 0.0f, live1 ? x1[k0] : 0.0f);
      const float2 b1 = float2(live0 ? x0[k0 + 8u] : 0.0f, live1 ? x1[k0 + 8u] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const ulong word = *(const device ulong*)(We + row[t] * stride + g * 8u);
        const uint c0 = uint(word >> (8u * nb)) & 255u, c1 = uint(word >> (8u * (4u + nb))) & 255u;
        float2 p = float2(0.0f);
        i2t_mma(p, float2(lut[c0 & 15u], lut[c0 >> 4u]), b0);
        i2t_mma(p, float2(lut[c1 & 15u], lut[c1 >> 4u]), b1);
        acc[l][t] = fma(float2(scales[Se[row[t] * groups + g]]), p, acc[l][t]);
      }
    }
  }
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    float2 v[CL];
#pragma unroll
    for (uint l = 0; l < CL; ++l) v[l] = acc[l][t];
    part[sg][t][lane] = i2t_tree(v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0u) return;
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    float2 v[SG];
#pragma unroll
    for (uint i = 0; i < SG; ++i) v[i] = part[i][t][lane];
    const float2 sum = i2t_tree(v);
    const uint r = r0 + t * 8u + fm;
    if (GU) {
      const float2 up = float2(simd_shuffle_xor(sum.x, 2u), simd_shuffle_xor(sum.y, 2u));
      if ((fm & 1u) != 0u || r >= wrows) continue;
      if (live0) Y[(size_t)member[fn] * a.n + r / 2u] = i2t_situ(sum.x * global, up.x * global, a.gate_cap, a.up_cap);
      if (live1) Y[(size_t)member[fn + 1u] * a.n + r / 2u] = i2t_situ(sum.y * global, up.y * global, a.gate_cap, a.up_cap);
    } else if (r < wrows) {
      if (live0) Y[(size_t)member[fn] * a.n + r] = sum.x * global;
      if (live1) Y[(size_t)member[fn + 1u] * a.n + r] = sum.y * global;
    }
  }
}

#define I2T_EXPERTS(name, gu, sgs, t) \
  template [[host_name(#name)]] [[kernel]] void i2t_experts<gu, sgs, t>(const device float* X [[buffer(0)]], \
      const device uchar* W [[buffer(1)]], const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]], \
      const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]], constant i2t_moe& a [[buffer(6)]], \
      device float* Y [[buffer(7)]], const device float* CODES [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]], \
      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

I2T_EXPERTS(i2t_gate_up, true, 4, 2)
I2T_EXPERTS(i2t_down, false, 8, 4)

template <uint BITS, uint R, uint C, uint U>
inline void i2r_dot_ahead(const device float* x, size_t step, uint rows, const device uint* w, const device bfloat* s,
    const device bfloat* b, uint col0, uint last, uint width, uint lane, thread float (&acc)[C][R]) {
  constexpr uint per = 32u / BITS;
  constexpr uint mask = (1u << BITS) - 1u;
  const uint words = width / per;
  const uint groups = width / 64u;
  const uint4 shift = uint4(0u, BITS, BITS * 2u, BITS * 3u);
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) acc[c][r] = 0.0f;
  }
  uint word = lane;
  for (; word + 32u * (U - 1u) < words; word += 32u * U) {
    uint packed[C][U];
    float scale[C][U], bias[C][U];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
#pragma unroll
      for (uint u = 0; u < U; ++u) {
        const uint at = word + 32u * u;
        packed[c][u] = w[col * words + at];
        const size_t group = col * groups + (at * per) / 64u;
        scale[c][u] = float(s[group]);
        bias[c][u] = float(b[group]);
      }
    }
#pragma unroll
    for (uint u = 0; u < U; ++u) {
      const uint at = word + 32u * u;
      float4 v0[C], v1[C];
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        v0[c] = fma(float4((uint4(packed[c][u]) >> shift) & mask), float4(scale[c][u]), float4(bias[c][u]));
        v1[c] = BITS == 4u ? fma(float4((uint4(packed[c][u] >> 16u) >> shift) & mask), float4(scale[c][u]), float4(bias[c][u])) : float4(0.0f);
      }
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        if (r >= rows) break;
        const device float* xr = x + r * step + at * per;
        const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(xr));
        const float4 in1 = BITS == 4u ? float4(*reinterpret_cast<const device packed_float4*>(xr + 4u)) : float4(0.0f);
#pragma unroll
        for (uint c = 0; c < C; ++c) {
          float a = acc[c][r];
          a = fma(in0.x, v0[c].x, a);
          a = fma(in0.y, v0[c].y, a);
          a = fma(in0.z, v0[c].z, a);
          a = fma(in0.w, v0[c].w, a);
          if (BITS == 4u) {
            a = fma(in1.x, v1[c].x, a);
            a = fma(in1.y, v1[c].y, a);
            a = fma(in1.z, v1[c].z, a);
            a = fma(in1.w, v1[c].w, a);
          }
          acc[c][r] = a;
        }
      }
    }
  }
  for (; word < words; word += 32u) {
    float4 v0[C], v1[C];
#pragma unroll
    for (uint c = 0; c < C; ++c) {
      const size_t col = min(col0 + c, last);
      const uint packed = w[col * words + word];
      const size_t group = col * groups + (word * per) / 64u;
      const float scale = float(s[group]), bias = float(b[group]);
      v0[c] = fma(float4((uint4(packed) >> shift) & mask), float4(scale), float4(bias));
      v1[c] = BITS == 4u ? fma(float4((uint4(packed >> 16u) >> shift) & mask), float4(scale), float4(bias)) : float4(0.0f);
    }
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      const device float* xr = x + r * step + word * per;
      const float4 in0 = float4(*reinterpret_cast<const device packed_float4*>(xr));
      const float4 in1 = BITS == 4u ? float4(*reinterpret_cast<const device packed_float4*>(xr + 4u)) : float4(0.0f);
#pragma unroll
      for (uint c = 0; c < C; ++c) {
        float a = acc[c][r];
        a = fma(in0.x, v0[c].x, a);
        a = fma(in0.y, v0[c].y, a);
        a = fma(in0.z, v0[c].z, a);
        a = fma(in0.w, v0[c].w, a);
        if (BITS == 4u) {
          a = fma(in1.x, v1[c].x, a);
          a = fma(in1.y, v1[c].y, a);
          a = fma(in1.z, v1[c].z, a);
          a = fma(in1.w, v1[c].w, a);
        }
        acc[c][r] = a;
      }
    }
  }
#pragma unroll
  for (uint c = 0; c < C; ++c) {
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      acc[c][r] = simd_sum(acc[c][r]);
    }
  }
}

template <uint R, uint C, uint U>
[[kernel]] void i2r_dense_ahead(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col0 = (group * 8u + tid / 32u) * C;
  const uint lane = tid % 32u;
  if (col0 >= a.n) return;
  float acc[C][R];
  i2r_dot_ahead<8u, R, C, U>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, s, b, col0, a.n - 1u, a.k, lane, acc);
  if (lane != 0u) return;
#pragma unroll
  for (uint c = 0; c < C; ++c) {
    if (col0 + c >= a.n) break;
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      if (r >= a.rows) break;
      const float v = acc[c][r];
      y[(size_t)r * a.n + col0 + c] = a.cap > 0.0f ? a.cap * precise::tanh(v / a.cap) : v;
    }
  }
}

template [[host_name("i2r_dense_ahead_2")]] [[kernel]] void i2r_dense_ahead<2, 4, 2>(const device float* x [[buffer(0)]],
    const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2t_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]);
