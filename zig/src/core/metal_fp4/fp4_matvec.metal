// Decode-time routed-expert matvecs for Inkling 2 Mini's ModelOpt stores, in four structures:
// one simdgroup an output pair, four pairs a threadgroup over the activation in threadgroup memory,
// table decode, and a thread an output pair. Compiled once per stored format: NVFP4 is e2m1
// nibbles under an e4m3 scale every 16 inputs times a per-tensor global, MXFP8 e4m3 bytes under
// an e8m0 scale every 32 inputs.
#include <metal_stdlib>
using namespace metal;

struct fp4_args {
  uint k, n, rows, topk;
  float gate_cap, up_cap;
};

#ifdef FP4_MXFP8
#define FP4_CODE_STRIDE(k) (k)
#define FP4_SCALE_STRIDE(k) ((k) / 32u)
#define FP4_BYTE_OFFSET(k) (k)
#define FP4_GROUP 32u
#else
#define FP4_CODE_STRIDE(k) ((k) / 2u)
#define FP4_SCALE_STRIDE(k) ((k) / 16u)
#define FP4_BYTE_OFFSET(k) ((k) / 2u)
#define FP4_GROUP 16u
#endif

inline float fp4_e2m1(uint c) {
  const float t[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
  const float v = t[c & 7u];
  return (c & 8u) ? -v : v;
}

inline float fp4_e4m3(uint c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (c >> 3u) & 15u, m = c & 7u;
  const float v = e ? (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7)) : float(m) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

inline float fp4_e8m0(uint s) {
  return exp2(float(int(s & 127u)) - 127.0f);
}

inline float fp4_scale(const device uchar* s, ulong at) {
#ifdef FP4_MXFP8
  return fp4_e8m0(s[at]);
#else
  return fp4_e4m3(s[at]);
#endif
}

inline uint fp4_byte(ulong word, uint j) {
  return uint(word >> (8u * j)) & 255u;
}

#ifdef FP4_MXFP8
// A 32-code group is four 8-byte words; each word is eight e4m3 codes.
#define FP4_PAIR_BODY \
  const ulong ag[4] = { *(const device ulong*)(wg), *(const device ulong*)(wg + 8), \
                        *(const device ulong*)(wg + 16), *(const device ulong*)(wg + 24) }; \
  const ulong au[4] = { *(const device ulong*)(wu), *(const device ulong*)(wu + 8), \
                        *(const device ulong*)(wu + 16), *(const device ulong*)(wu + 24) }; \
  for (uint q = 0; q < 4u; ++q) { \
    for (uint j = 0; j < 8u; ++j) { \
      const uint at = q * 8u + j; \
      const float xv = x[k0 + at]; \
      dg = fma(fp4_e4m3(fp4_byte(ag[q], j)), xv, dg); \
      du = fma(fp4_e4m3(fp4_byte(au[q], j)), xv, du); \
    } \
  }
#define FP4_ROW_BODY \
  const ulong a[4] = { *(const device ulong*)(w), *(const device ulong*)(w + 8), \
                       *(const device ulong*)(w + 16), *(const device ulong*)(w + 24) }; \
  for (uint q = 0; q < 4u; ++q) { \
    for (uint j = 0; j < 8u; ++j) { \
      d = fma(fp4_e4m3(fp4_byte(a[q], j)), x[k0 + q * 8u + j], d); \
    } \
  }
#define FP4_TABLE_PAIR_BODY \
  const ulong ag[4] = { *(const device ulong*)(wg), *(const device ulong*)(wg + 8), \
                        *(const device ulong*)(wg + 16), *(const device ulong*)(wg + 24) }; \
  const ulong au[4] = { *(const device ulong*)(wu), *(const device ulong*)(wu + 8), \
                        *(const device ulong*)(wu + 16), *(const device ulong*)(wu + 24) }; \
  for (uint q = 0; q < 4u; ++q) { \
    for (uint j = 0; j < 8u; ++j) { \
      const float xv = x[k0 + q * 8u + j]; \
      dg = fma(codes[fp4_byte(ag[q], j)], xv, dg); \
      du = fma(codes[fp4_byte(au[q], j)], xv, du); \
    } \
  }
#define FP4_TABLE_ROW_BODY \
  const ulong a[4] = { *(const device ulong*)(w), *(const device ulong*)(w + 8), \
                       *(const device ulong*)(w + 16), *(const device ulong*)(w + 24) }; \
  for (uint q = 0; q < 4u; ++q) { \
    for (uint j = 0; j < 8u; ++j) d = fma(codes[fp4_byte(a[q], j)], x[k0 + q * 8u + j], d); \
  }
#else
// A 16-code group is one 8-byte word, low nibble first.
#define FP4_PAIR_BODY \
  const ulong ag = *(const device ulong*)(wg), au = *(const device ulong*)(wu); \
  for (uint j = 0; j < 8u; ++j) { \
    const uint ab = fp4_byte(ag, j), ub = fp4_byte(au, j); \
    const float x0 = x[k0 + 2u * j], x1 = x[k0 + 2u * j + 1u]; \
    dg = fma(fp4_e2m1(ab & 15u), x0, dg); \
    dg = fma(fp4_e2m1(ab >> 4u), x1, dg); \
    du = fma(fp4_e2m1(ub & 15u), x0, du); \
    du = fma(fp4_e2m1(ub >> 4u), x1, du); \
  }
#define FP4_ROW_BODY \
  const ulong a = *(const device ulong*)(w); \
  for (uint j = 0; j < 8u; ++j) { \
    const uint b = fp4_byte(a, j); \
    d = fma(fp4_e2m1(b & 15u), x[k0 + 2u * j], d); \
    d = fma(fp4_e2m1(b >> 4u), x[k0 + 2u * j + 1u], d); \
  }
#define FP4_TABLE_PAIR_BODY \
  const ulong ag = *(const device ulong*)(wg), au = *(const device ulong*)(wu); \
  for (uint j = 0; j < 8u; ++j) { \
    const uint ab = fp4_byte(ag, j), ub = fp4_byte(au, j); \
    const float x0 = x[k0 + 2u * j], x1 = x[k0 + 2u * j + 1u]; \
    dg = fma(codes[ab & 15u], x0, dg); \
    dg = fma(codes[ab >> 4u], x1, dg); \
    du = fma(codes[ub & 15u], x0, du); \
    du = fma(codes[ub >> 4u], x1, du); \
  }
#define FP4_TABLE_ROW_BODY \
  const ulong a = *(const device ulong*)(w); \
  for (uint j = 0; j < 8u; ++j) { \
    const uint b = fp4_byte(a, j); \
    d = fma(codes[b & 15u], x[k0 + 2u * j], d); \
    d = fma(codes[b >> 4u], x[k0 + 2u * j + 1u], d); \
  }
#endif

inline void fp4_pair_dot(const device uchar* wg, const device uchar* wu, const device float* x,
                         uint k0, thread float& dg, thread float& du) {
  FP4_PAIR_BODY
}

inline void fp4_pair_dot_tg(const device uchar* wg, const device uchar* wu, const threadgroup float* x,
                            uint k0, thread float& dg, thread float& du) {
  FP4_PAIR_BODY
}

inline void fp4_row_dot(const device uchar* w, const device float* x, uint k0, thread float& d) {
  FP4_ROW_BODY
}

inline void fp4_row_dot_tg(const device uchar* w, const threadgroup float* x, uint k0, thread float& d) {
  FP4_ROW_BODY
}

inline void fp4_table_pair_dot(const device uchar* wg, const device uchar* wu, const device float* x,
                               uint k0, const device float* codes, thread float& dg, thread float& du) {
  FP4_TABLE_PAIR_BODY
}

inline void fp4_table_row_dot(const device uchar* w, const device float* x, uint k0,
                              const device float* codes, thread float& d) {
  FP4_TABLE_ROW_BODY
}

inline float fp4_gate(float g, float cap) {
  return cap * precise::tanh(g / cap) * (1.0f / (1.0f + precise::exp(-g)));
}

inline float fp4_up(float u, float cap) {
  return cap > 0.0f ? cap * precise::tanh(u / cap) : u;
}

inline float fp4_situ(float g, float u, float cap_g, float cap_u) {
  return fp4_gate(g, cap_g) * fp4_up(u, cap_u);
}

// One simdgroup a gate/up pair: 32 lanes cover the scale groups of its two rows.
kernel void fp4_gu_rows(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                        const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                        const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                        device float* ACT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x, e = tg.y;
  if (c >= a.n) return;
  const uint t = e / a.topk;
  const device float* x = X + (ulong)t * a.k;
  const ulong row = ulong(e) * a.rows + 2u * c;
  const device uchar* wg = W + row * FP4_CODE_STRIDE(a.k);
  const device uchar* wu = wg + FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float dg = 0.0f, du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float pg = 0.0f, pu = 0.0f;
    fp4_pair_dot(wg + FP4_BYTE_OFFSET(k0), wu + FP4_BYTE_OFFSET(k0), x, k0, pg, pu);
    dg = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(fp4_scale(S, (row + 1u) * FP4_SCALE_STRIDE(a.k) + g), pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
    const float global = AMAX[e] / DEN[0];
    ACT[ulong(e) * a.n + c] = fp4_situ(dg * global, du * global, a.gate_cap, a.up_cap);
  }
}

// One simdgroup an output row.
kernel void fp4_down_rows(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                          const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                          const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                          device float* OUT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x, e = tg.y;
  if (r >= a.n) return;
  const device float* x = X + (ulong)e * a.k;
  const ulong row = ulong(e) * a.rows + r;
  const device uchar* w = W + row * FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float d = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float p = 0.0f;
    fp4_row_dot(w + FP4_BYTE_OFFSET(k0), x, k0, p);
    d = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), p, d);
  }
  d = simd_sum(d);
  if (lane == 0u) OUT[ulong(e) * a.n + r] = d * (AMAX[e] / DEN[0]);
}

// Four gate/up pairs a threadgroup over one staged activation row.
kernel void fp4_gu_shared(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                          const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                          const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                          device float* ACT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float xs[2304];
  const uint e = tg.y;
  const uint t = e / a.topk;
  const device float* x = X + (ulong)t * a.k;
  for (uint i = tid.x; i < a.k; i += 128u) xs[i] = x[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tg.x * 4u + sg;
  if (c >= a.n) return;
  const ulong row = ulong(e) * a.rows + 2u * c;
  const device uchar* wg = W + row * FP4_CODE_STRIDE(a.k);
  const device uchar* wu = wg + FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float dg = 0.0f, du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float pg = 0.0f, pu = 0.0f;
    fp4_pair_dot_tg(wg + FP4_BYTE_OFFSET(k0), wu + FP4_BYTE_OFFSET(k0), xs, k0, pg, pu);
    dg = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(fp4_scale(S, (row + 1u) * FP4_SCALE_STRIDE(a.k) + g), pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
    const float global = AMAX[e] / DEN[0];
    ACT[ulong(e) * a.n + c] = fp4_situ(dg * global, du * global, a.gate_cap, a.up_cap);
  }
}

// Four output rows a threadgroup over one staged activation row.
kernel void fp4_down_shared(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                            const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                            const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                            device float* OUT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                            uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float xs[2304];
  const uint e = tg.y;
  const device float* x = X + (ulong)e * a.k;
  for (uint i = tid.x; i < a.k; i += 128u) xs[i] = x[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint r = tg.x * 4u + sg;
  if (r >= a.n) return;
  const ulong row = ulong(e) * a.rows + r;
  const device uchar* w = W + row * FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float d = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float p = 0.0f;
    fp4_row_dot_tg(w + FP4_BYTE_OFFSET(k0), xs, k0, p);
    d = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), p, d);
  }
  d = simd_sum(d);
  if (lane == 0u) OUT[ulong(e) * a.n + r] = d * (AMAX[e] / DEN[0]);
}

// One simdgroup a gate/up pair whose codes come from a lookup table.
kernel void fp4_gu_table(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                         const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                         const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                         device float* ACT [[buffer(6)]], const device float* T16 [[buffer(7)]],
                         const device float* T256 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x, e = tg.y;
  if (c >= a.n) return;
  const uint t = e / a.topk;
  const device float* x = X + (ulong)t * a.k;
  const ulong row = ulong(e) * a.rows + 2u * c;
  const device uchar* wg = W + row * FP4_CODE_STRIDE(a.k);
  const device uchar* wu = wg + FP4_CODE_STRIDE(a.k);
#ifdef FP4_MXFP8
  const device float* codes = T256;
#else
  const device float* codes = T16;
#endif
  const uint groups = a.k / FP4_GROUP;
  float dg = 0.0f, du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float pg = 0.0f, pu = 0.0f;
    fp4_table_pair_dot(wg + FP4_BYTE_OFFSET(k0), wu + FP4_BYTE_OFFSET(k0), x, k0, codes, pg, pu);
    dg = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(fp4_scale(S, (row + 1u) * FP4_SCALE_STRIDE(a.k) + g), pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
    const float global = AMAX[e] / DEN[0];
    ACT[ulong(e) * a.n + c] = fp4_situ(dg * global, du * global, a.gate_cap, a.up_cap);
  }
}

// One simdgroup an output row whose codes come from a lookup table.
kernel void fp4_down_table(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                           const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                           const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                           device float* OUT [[buffer(6)]], const device float* T16 [[buffer(7)]],
                           const device float* T256 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x, e = tg.y;
  if (r >= a.n) return;
  const device float* x = X + (ulong)e * a.k;
  const ulong row = ulong(e) * a.rows + r;
  const device uchar* w = W + row * FP4_CODE_STRIDE(a.k);
#ifdef FP4_MXFP8
  const device float* codes = T256;
#else
  const device float* codes = T16;
#endif
  const uint groups = a.k / FP4_GROUP;
  float d = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * FP4_GROUP;
    float p = 0.0f;
    fp4_table_row_dot(w + FP4_BYTE_OFFSET(k0), x, k0, codes, p);
    d = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), p, d);
  }
  d = simd_sum(d);
  if (lane == 0u) OUT[ulong(e) * a.n + r] = d * (AMAX[e] / DEN[0]);
}

// One thread a gate/up pair: its two rows stream through the thread with no reduction.
kernel void fp4_gu_thread(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                          const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                          const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                          device float* ACT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
  const uint c = tg.x * 64u + tid.x;
  const uint e = tg.y;
  if (c >= a.n) return;
  const uint t = e / a.topk;
  const device float* x = X + (ulong)t * a.k;
  const ulong row = ulong(e) * a.rows + 2u * c;
  const device uchar* wg = W + row * FP4_CODE_STRIDE(a.k);
  const device uchar* wu = wg + FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float dg = 0.0f, du = 0.0f;
  for (uint g = 0; g < groups; ++g) {
    const uint k0 = g * FP4_GROUP;
    float pg = 0.0f, pu = 0.0f;
    fp4_pair_dot(wg + FP4_BYTE_OFFSET(k0), wu + FP4_BYTE_OFFSET(k0), x, k0, pg, pu);
    dg = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(fp4_scale(S, (row + 1u) * FP4_SCALE_STRIDE(a.k) + g), pu, du);
  }
  const float global = AMAX[e] / DEN[0];
  ACT[ulong(e) * a.n + c] = fp4_situ(dg * global, du * global, a.gate_cap, a.up_cap);
}

// One thread an output row.
kernel void fp4_down_thread(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
                            const device uchar* S [[buffer(2)]], const device float* AMAX [[buffer(3)]],
                            const device float* DEN [[buffer(4)]], constant fp4_args& a [[buffer(5)]],
                            device float* OUT [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                            uint2 tid [[thread_position_in_threadgroup]]) {
  const uint r = tg.x * 64u + tid.x;
  const uint e = tg.y;
  if (r >= a.n) return;
  const device float* x = X + (ulong)e * a.k;
  const ulong row = ulong(e) * a.rows + r;
  const device uchar* w = W + row * FP4_CODE_STRIDE(a.k);
  const uint groups = a.k / FP4_GROUP;
  float d = 0.0f;
  for (uint g = 0; g < groups; ++g) {
    const uint k0 = g * FP4_GROUP;
    float p = 0.0f;
    fp4_row_dot(w + FP4_BYTE_OFFSET(k0), x, k0, p);
    d = fma(fp4_scale(S, row * FP4_SCALE_STRIDE(a.k) + g), p, d);
  }
  OUT[ulong(e) * a.n + r] = d * (AMAX[e] / DEN[0]);
}

kernel void fp4_copy(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < count) dst[gid] = src[gid];
}

// Four 16-byte loads folded to one word: the read-only ceiling the matvecs are measured against,
// with a sixteenth of the traffic going back out so the stream is not write-limited.
kernel void fp4_read(const device uint4* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                     constant uint& count [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  uint acc = 0;
  for (uint i = 0; i < 4u; ++i) {
    const uint at = gid * 4u + i;
    if (at < count) {
      const uint4 v = src[at];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  dst[gid] = acc;
}
