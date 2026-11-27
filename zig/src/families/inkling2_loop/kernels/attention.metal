#include <metal_stdlib>
using namespace metal;

struct i2_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };

// One split's rows for the G query heads of a key/value head, one SIMD group a head, a row at a time: the order every
// decode result, and prefill's attention, follow.
inline void i2_split_serial(const device float* Q, const device half* KEYS, const device half* VALUES,
    const device float* B, const device float* sink, constant i2_attn_parts& a, device float* PART,
    uint tgp, uint tid, uint sg, uint lane, uint nth, threadgroup float4* KB4, threadgroup float4* VB4) {
  const uint G = a.heads / a.kv;
  const uint D = a.dim, per = max(1u, D / 32u);
  const uint kv = tgp / a.splits, s = tgp % a.splits, h = kv * G + sg;
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  float qv[4];
  for (uint j = 0; j < per; ++j) qv[j] = (lane + 32u * j < D) ? Q[h * D + lane + 32u * j] * a.scale * tau : 0.0f;
  const float sk = sink[h];
  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
  const size_t base = ((size_t)h * a.splits + s) * (D + 2u);
  const uint js = lo + s * a.chunk, je = min(js + a.chunk, a.pos + 1u);
  if (js >= je) {
    if (lane == 0u) {
      PART[base] = -INFINITY;
      PART[base + 1u] = 0.0f;
    }
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = 0.0f;
    return;
  }
  threadgroup float* KB = (threadgroup float*)KB4;
  threadgroup float* VB = (threadgroup float*)VB4;
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint j = js; j < je; ++j) {
    const uint col = min(a.pos - j, a.extent);
    const uint rj = a.window ? (j % a.window) : j;
    const size_t at = ((size_t)rj * a.kv + kv) * D;
    const float bias = B[h * (a.extent + 1u) + col] - sk;
    for (uint v = tid; v < D / 4u; v += nth) {
      KB4[v] = float4(*reinterpret_cast<const device packed_half4*>(KEYS + at + v * 4u));
      VB4[v] = float4(*reinterpret_cast<const device packed_half4*>(VALUES + at + v * 4u));
    }
    for (uint i = (D / 4u) * 4u + tid; i < D; i += nth) {
      KB[i] = KEYS[at + i];
      VB[i] = VALUES[at + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot = 0.0f;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) dot = fma(qv[u], KB[lane + 32u * u], dot);
    const float score = simd_sum(dot) + bias;
    const float peak = max(top, score);
    const float rescale = precise::exp(top - peak);
    const float w = precise::exp(score - peak);
    mass = mass * rescale + w;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * rescale + w * VB[lane + 32u * u];
    top = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = acc[u];
}

// Row r's sum over the 32 lanes lands in lane r, with the additions simd_sum makes: partners at xor 1, 2, 4, 8, 16.
inline float i2_rows_sum(thread float (&p)[32], uint lane) {
  float a1[16], a2[8], a3[4], a4[2];
  const bool b0 = (lane & 1u) != 0u, b1 = (lane & 2u) != 0u, b2 = (lane & 4u) != 0u, b3 = (lane & 8u) != 0u;
  const bool b4 = (lane & 16u) != 0u;
#pragma unroll
  for (uint m = 0; m < 16u; ++m) {
    const float lo = p[2u * m], hi = p[2u * m + 1u];
    a1[m] = (b0 ? hi : lo) + simd_shuffle_xor(b0 ? lo : hi, ushort(1));
  }
#pragma unroll
  for (uint m = 0; m < 8u; ++m) {
    const float lo = a1[2u * m], hi = a1[2u * m + 1u];
    a2[m] = (b1 ? hi : lo) + simd_shuffle_xor(b1 ? lo : hi, ushort(2));
  }
#pragma unroll
  for (uint m = 0; m < 4u; ++m) {
    const float lo = a2[2u * m], hi = a2[2u * m + 1u];
    a3[m] = (b2 ? hi : lo) + simd_shuffle_xor(b2 ? lo : hi, ushort(4));
  }
#pragma unroll
  for (uint m = 0; m < 2u; ++m) {
    const float lo = a3[2u * m], hi = a3[2u * m + 1u];
    a4[m] = (b3 ? hi : lo) + simd_shuffle_xor(b3 ? lo : hi, ushort(8));
  }
  return (b4 ? a4[1] : a4[0]) + simd_shuffle_xor(b4 ? a4[0] : a4[1], ushort(16));
}

// The same split for 128-wide heads, three to a key/value head, 96 threads: 32-row blocks staged by every thread, lane
// l holding dimensions l, l + 32, l + 64 and l + 96 of each row. The scores come out of one reduction for the block,
// the running maximum out of a lane prefix maximum and the exponentials one row a lane; only the mass and value
// recurrences stay serial. Every value is the serial kernel's: the same products, sums and expressions.
inline void i2_split_blocks(const device float* Q, const device half* KEYS, const device half* VALUES,
    const device float* B, const device float* sink, constant i2_attn_parts& a, device float* PART,
    uint tgp, uint tid, uint sg, uint lane, threadgroup half4* KS4, threadgroup half4* VS4) {
  constexpr uint D = 128u, G = 3u, N = 96u, R = 32u, CELLS = (R * 32u + N - 1u) / N;
  const uint kv = tgp / a.splits, s = tgp % a.splits, h = kv * G + sg;
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  const float4 qv = *reinterpret_cast<const device float4*>(Q + h * D + lane * 4u) * (a.scale * tau);
  const float sk = sink[h];
  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
  const size_t base = ((size_t)h * a.splits + s) * (D + 2u);
  const uint js = lo + s * a.chunk, je = min(js + a.chunk, a.pos + 1u);
  float top = -INFINITY, mass = 0.0f;
  float4 acc = float4(0.0f);
  for (uint j0 = js; j0 < je; j0 += R) {
    const uint n = min(R, je - j0);
    const uint first = a.window ? j0 % a.window : j0;
    half4 kc[CELLS], vc[CELLS];
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      const uint r = min(e / 32u, n - 1u), j = first + r;
      const uint rj = a.window && j >= a.window ? j - a.window : j;
      const size_t at = ((size_t)rj * a.kv + kv) * D + (e % 32u) * 4u;
      kc[i] = *reinterpret_cast<const device half4*>(KEYS + at);
      vc[i] = *reinterpret_cast<const device half4*>(VALUES + at);
    }
    float own = 0.0f;
    if (lane < n) own = B[h * (a.extent + 1u) + min(a.pos - (j0 + lane), a.extent)] - sk;
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      if (e < R * 32u) {
        KS4[e] = kc[i];
        VS4[e] = vc[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p[32];
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float4 kk = float4(KS4[r * 32u + lane]);
      p[r] = fma(qv.w, kk.w, fma(qv.z, kk.z, fma(qv.y, kk.y, qv.x * kk.x)));
    }
    const float score = i2_rows_sum(p, lane) + own;
    const float bmax = simd_max(lane < n ? score : -INFINITY);
    const float peak = max(top, bmax);
    const float rescale = top == -INFINITY ? 0.0f : precise::exp(top - peak);
    const float weights = lane < n ? precise::exp(score - peak) : 0.0f;
    mass = mass * rescale + simd_sum(weights);
    acc *= rescale;
    if (n == R) {
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        const float w = simd_shuffle(weights, ushort(r));
        acc = fma(float4(w), float4(VS4[r * 32u + lane]), acc);
      }
    } else {
      for (uint r = 0; r < n; ++r) {
        const float w = simd_shuffle(weights, ushort(r));
        acc = fma(float4(w), float4(VS4[r * 32u + lane]), acc);
      }
    }
    top = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  *reinterpret_cast<device packed_float4*>(PART + base + 2u + lane * 4u) = packed_float4(acc);
}

inline bool i2_blocked(constant i2_attn_parts& a, uint nth) {
  return a.dim == 128u && a.heads == 3u * a.kv && nth == 96u;
}

[[kernel]] void i2_attn_split(const device float* Q [[buffer(0)]], const device half* KEYS [[buffer(1)]],
    const device half* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2_attn_parts& a [[buffer(5)]], device float* PART [[buffer(6)]],
    uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup half4 KS4[32 * 32];
  threadgroup half4 VS4[32 * 32];
  if (i2_blocked(a, nth)) i2_split_blocks(Q, KEYS, VALUES, B, sink, a, PART, tgp, tid, sg, lane, KS4, VS4);
  else i2_split_serial(Q, KEYS, VALUES, B, sink, a, PART, tgp, tid, sg, lane, nth, (threadgroup float4*)KS4, (threadgroup float4*)VS4);
}

// The splits' partials merged in split order from the sink's (0, 1).
inline void i2_combine_serial(const device float* PART, constant i2_attn_parts& a, device float* O, uint h, uint lane) {
  const uint D = a.dim, S = a.splits, per = max(1u, D / 32u);
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint s = 0; s < S; ++s) {
    const size_t base = ((size_t)h * S + s) * (D + 2u);
    const float ptop = PART[base];
    const float pmass = PART[base + 1u];
    if (pmass == 0.0f) continue;
    const float peak = max(top, ptop);
    const float r0 = precise::exp(top - peak);
    const float r1 = precise::exp(ptop - peak);
    mass = mass * r0 + pmass * r1;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * r0 + r1 * PART[base + 2u + lane + 32u * u];
    top = peak;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) O[h * D + lane + 32u * u] = acc[u] / mass;
}

// The same merge for 128-wide heads and at most 16 splits, every partial loaded before the merge starts.
inline void i2_combine_loaded(const device float* PART, constant i2_attn_parts& a, device float* O, uint h, uint lane) {
  constexpr uint D = 128u;
  const uint S = a.splits;
  const size_t first = (size_t)h * S * (D + 2u);
  const size_t mine = first + (size_t)min(lane, S - 1u) * (D + 2u);
  const float tops = PART[mine], masses = PART[mine + 1u];
  float pv[16][4];
#pragma unroll
  for (uint s = 0; s < 16u; ++s) {
    const size_t base = first + (size_t)min(s, S - 1u) * (D + 2u) + 2u + lane;
    pv[s][0] = PART[base];
    pv[s][1] = PART[base + 32u];
    pv[s][2] = PART[base + 64u];
    pv[s][3] = PART[base + 96u];
  }
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
  for (uint s = 0; s < 16u; ++s) {
    if (s >= S) break;
    const float ptop = simd_shuffle(tops, ushort(s));
    const float pmass = simd_shuffle(masses, ushort(s));
    if (pmass == 0.0f) continue;
    const float peak = max(top, ptop);
    const float r0 = precise::exp(top - peak);
    const float r1 = precise::exp(ptop - peak);
    mass = mass * r0 + pmass * r1;
    for (uint u = 0; u < 4u; ++u) acc[u] = acc[u] * r0 + r1 * pv[s][u];
    top = peak;
  }
  for (uint u = 0; u < 4u; ++u) O[h * D + lane + 32u * u] = acc[u] / mass;
}

[[kernel]] void i2_attn_combine(const device float* PART [[buffer(0)]],
    constant i2_attn_parts& a [[buffer(1)]], device float* O [[buffer(2)]],
    uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (a.dim == 128u && a.splits <= 16u) i2_combine_loaded(PART, a, O, h, lane);
  else i2_combine_serial(PART, a, O, h, lane);
}

struct i2_batch_row { device half* keys; device half* values; i2_attn_parts args; };

[[kernel]] void i2_batch_split(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2_batch_row* rows [[buffer(3)]], device float* PART [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 nth2 [[threads_per_threadgroup]]) {
  threadgroup half4 KS4[32 * 32];
  threadgroup half4 VS4[32 * 32];
  constant i2_attn_parts& a = rows[tg.y].args;
  if (tg.x >= a.kv * a.splits) return;
  Q += size_t(tg.y) * a.heads * a.dim;
  B += size_t(tg.y) * a.heads * (a.extent + 1u);
  sink += size_t(tg.y) * a.heads;
  PART += size_t(tg.y) * a.heads * 16u * (a.dim + 2u);
  if (i2_blocked(a, nth2.x)) i2_split_blocks(Q, rows[tg.y].keys, rows[tg.y].values, B, sink, a, PART, tg.x, tid2.x, sg, lane, KS4, VS4);
  else i2_split_serial(Q, rows[tg.y].keys, rows[tg.y].values, B, sink, a, PART, tg.x, tid2.x, sg, lane, nth2.x, (threadgroup float4*)KS4, (threadgroup float4*)VS4);
}

[[kernel]] void i2_batch_combine(const device float* PART [[buffer(0)]], constant i2_batch_row* rows [[buffer(1)]],
    device float* O [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constant i2_attn_parts& a = rows[tg.y].args;
  PART += size_t(tg.y) * a.heads * 16u * (a.dim + 2u);
  O += size_t(tg.y) * a.heads * a.dim;
  if (a.dim == 128u && a.splits <= 16u) i2_combine_loaded(PART, a, O, tg.x, lane);
  else i2_combine_serial(PART, a, O, tg.x, lane);
}

[[kernel]] void i2_batch_combine_sig_mul(const device float* PART [[buffer(0)]], constant i2_batch_row* rows [[buffer(1)]],
    const device float* GATE [[buffer(2)]], device float* O [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constant i2_attn_parts& a = rows[tg.y].args;
  PART += size_t(tg.y) * a.heads * 16u * (a.dim + 2u);
  GATE += size_t(tg.y) * a.heads * a.dim;
  O += size_t(tg.y) * a.heads * a.dim;
  const uint h = tg.x;
  if (a.dim == 128u && a.splits <= 16u) {
    constexpr uint D = 128u;
    const uint S = a.splits;
    const size_t first = (size_t)h * S * (D + 2u);
    const size_t mine = first + (size_t)min(lane, S - 1u) * (D + 2u);
    const float tops = PART[mine], masses = PART[mine + 1u];
    float pv[16][4];
#pragma unroll
    for (uint s = 0; s < 16u; ++s) {
      const size_t base = first + (size_t)min(s, S - 1u) * (D + 2u) + 2u + lane;
      pv[s][0] = PART[base];
      pv[s][1] = PART[base + 32u];
      pv[s][2] = PART[base + 64u];
      pv[s][3] = PART[base + 96u];
    }
    float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
    for (uint s = 0; s < 16u; ++s) {
      if (s >= S) break;
      const float ptop = simd_shuffle(tops, ushort(s));
      const float pmass = simd_shuffle(masses, ushort(s));
      if (pmass == 0.0f) continue;
      const float peak = max(top, ptop);
      const float r0 = precise::exp(top - peak);
      const float r1 = precise::exp(ptop - peak);
      mass = mass * r0 + pmass * r1;
      for (uint u = 0; u < 4u; ++u) acc[u] = acc[u] * r0 + r1 * pv[s][u];
      top = peak;
    }
    for (uint u = 0; u < 4u; ++u) {
      const uint idx = h * D + lane + 32u * u;
      O[idx] = (acc[u] / mass) / (1.0f + precise::exp(-GATE[idx]));
    }
    return;
  }
  const uint D = a.dim, S = a.splits, per = max(1u, D / 32u);
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint s = 0; s < S; ++s) {
    const size_t base = ((size_t)h * S + s) * (D + 2u);
    const float ptop = PART[base];
    const float pmass = PART[base + 1u];
    if (pmass == 0.0f) continue;
    const float peak = max(top, ptop);
    const float r0 = precise::exp(top - peak);
    const float r1 = precise::exp(ptop - peak);
    mass = mass * r0 + pmass * r1;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * r0 + r1 * PART[base + 2u + lane + 32u * u];
    top = peak;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) {
    const uint idx = h * D + lane + 32u * u;
    O[idx] = (acc[u] / mass) / (1.0f + precise::exp(-GATE[idx]));
  }
}

struct i2_mix_args { uint width, depth, k, rows; float scale; };

// A token's depth mix for up to four queries over up to five states, one threadgroup of 256 a row: every query and
// state's dot is accumulated in one pass, each with the per-thread order, SIMD reduction and part order of i2p_mix.
inline void i2_mix_rows(const device float* S, const device float* q0, const device float* q1, const device float* q2,
    const device float* q3, constant i2_mix_args& a, uint kn, device float* O, uint t, uint i,
    threadgroup float (&part)[4][5][8], threadgroup float (&w)[4][5]) {
  const device float* st = S + (size_t)t * a.depth * a.width;
  device float* out = O + (size_t)t * kn * a.width;
  const device float* qs[4] = { q0, q1, q2, q3 };
  float sums[4][5];
#pragma unroll
  for (uint k = 0; k < 4u; ++k) {
#pragma unroll
    for (uint j = 0; j < 5u; ++j) sums[k][j] = 0.0f;
  }
  for (uint u = i; u < a.width; u += 256u) {
    float sv[5];
#pragma unroll
    for (uint j = 0; j < 5u; ++j) sv[j] = j < a.depth ? st[j * a.width + u] : 0.0f;
#pragma unroll
    for (uint k = 0; k < 4u; ++k) {
      if (k >= kn) break;
      const float qv = qs[k][u];
#pragma unroll
      for (uint j = 0; j < 5u; ++j) if (j < a.depth) sums[k][j] = fma(sv[j], qv, sums[k][j]);
    }
  }
#pragma unroll
  for (uint k = 0; k < 4u; ++k) {
    if (k >= kn) break;
#pragma unroll
    for (uint j = 0; j < 5u; ++j) {
      if (j >= a.depth) break;
      const float sum = simd_sum(sums[k][j]);
      if (i % 32u == 0u) part[k][j][i / 32u] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (i < kn) {
    const uint k = i;
    for (uint j = 0; j < a.depth; ++j) {
      float total = 0.0f;
      for (uint s = 0; s < 8u; ++s) total += part[k][j][s];
      w[k][j] = total * a.scale;
    }
    float peak = w[k][0];
    for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
    float total = 0.0f;
    for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
    for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < kn; ++k) {
    for (uint u = i; u < a.width; u += 256u) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], st[j * a.width + u], sum);
      out[k * a.width + u] = sum;
    }
  }
}

[[kernel]] void i2_batch_mix(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    constant i2_mix_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  i2_mix_rows(S, Q, Q + a.width, Q + 2u * a.width, Q + 3u * a.width, a, min(a.k, 4u), O, t, i, part, w);
}

[[kernel]] void i2_batch_mix4(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    const device float* K [[buffer(2)]], const device float* V [[buffer(3)]], const device float* F [[buffer(4)]],
    constant i2_mix_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  i2_mix_rows(S, Q, K, V, F, a, 4u, O, t, i, part, w);
}

struct i2_controls_args { uint h, rank, rows; uint xs[3], xb[3]; uint msg0, cols; };

[[kernel]] void i2_controls_norm(const device float* X0 [[buffer(0)]], const device float* X1 [[buffer(1)]],
    const device float* X2 [[buffer(2)]], const device float* MSG [[buffer(3)]],
    constant i2_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint controller = group / (2u * a.rows), rem = group % (2u * a.rows);
  const uint row = rem / 2u, which = rem % 2u;
  const device float* cur = controller == 0u ? X0 : controller == 1u ? X1 : X2;
  const device float* src = which == 0u ? cur + size_t(a.xb[controller] + row * a.xs[controller]) * a.h
    : MSG + size_t(a.msg0 + controller + row * 4u) * a.h;
  device float* dst = OUT + (size_t(controller) * a.rows + row) * 2u * a.h + which * a.h;
  float sum = 0.0f;
  if ((a.h & 3u) == 0u) {
    const uint h4 = a.h >> 2u;
    const device float4* src4 = reinterpret_cast<const device float4*>(src);
    device float4* dst4 = reinterpret_cast<device float4*>(dst);
    for (uint i = tid; i < h4; i += nth) {
      const float4 v = src4[i];
      sum = fma(v.w, v.w, fma(v.z, v.z, fma(v.y, v.y, fma(v.x, v.x, sum))));
    }
    sum = simd_sum(sum);
    if (tid % 32u == 0u) part[tid / 32u] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nth / 32u; ++i) total += part[i];
    const float inv = 1.0f / sqrt(total / float(a.h) + 1e-6f);
    for (uint i = tid; i < h4; i += nth) dst4[i] = src4[i] * inv;
    return;
  }
  for (uint i = tid; i < a.h; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < nth / 32u; ++i) total += part[i];
  const float inv = 1.0f / sqrt(total / float(a.h) + 1e-6f);
  for (uint i = tid; i < a.h; i += nth) dst[i] = src[i] * inv;
}

inline float4 i2_ctl_load4(const device float* p) {
  return float4(*reinterpret_cast<const device packed_float4*>(p));
}

inline float4 i2_ctl_load4(const device bfloat* p) {
  const uint2 u = *reinterpret_cast<const device uint2*>(p);
  return float4(as_type<float>(u.x << 16u), as_type<float>(u.x & 0xffff0000u),
                as_type<float>(u.y << 16u), as_type<float>(u.y & 0xffff0000u));
}

template <typename W>
[[kernel]] void i2_controls_down_t(const device float* X [[buffer(0)]], const device W* W0 [[buffer(1)]],
    const device W* W1 [[buffer(2)]], const device W* W2 [[buffer(3)]],
    constant i2_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u;
  if (col >= a.rank || group.y >= a.rows) return;
  const uint width = 2u * a.h;
  const device float* x = X + (size_t(group.z) * a.rows + group.y) * width;
  const device W* w = (group.z == 0u ? W0 : group.z == 1u ? W1 : W2) + size_t(col) * width;
  float acc = 0.0f;
  uint base = 0u;
  for (; base + 128u <= width; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
    const float4 wv = i2_ctl_load4(w + k);
    acc = fma(xv.x, wv.x, acc); acc = fma(xv.y, wv.y, acc);
    acc = fma(xv.z, wv.z, acc); acc = fma(xv.w, wv.w, acc);
  }
  for (uint k = base + lane; k < width; k += 32u) acc = fma(x[k], float(w[k]), acc);
  acc = simd_sum(acc);
  if (lane == 0u) OUT[(size_t(group.z) * a.rows + group.y) * a.rank + col] = acc;
}

template [[host_name("i2_controls_down")]] [[kernel]] void i2_controls_down_t<float>(
    const device float* X [[buffer(0)]], const device float* W0 [[buffer(1)]],
    const device float* W1 [[buffer(2)]], const device float* W2 [[buffer(3)]],
    constant i2_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]);
template [[host_name("i2_controls_down_bf16")]] [[kernel]] void i2_controls_down_t<bfloat>(
    const device float* X [[buffer(0)]], const device bfloat* W0 [[buffer(1)]],
    const device bfloat* W1 [[buffer(2)]], const device bfloat* W2 [[buffer(3)]],
    constant i2_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]);

#define I2_CONTROLS_APPLY_ARGS(W) const device float* X [[buffer(0)]], const device W* G0 [[buffer(1)]], \
    const device W* G1 [[buffer(2)]], const device W* G2 [[buffer(3)]], const device W* F0 [[buffer(4)]], \
    const device W* F1 [[buffer(5)]], const device W* F2 [[buffer(6)]], const device float* C0 [[buffer(7)]], \
    const device float* C1 [[buffer(8)]], const device float* C2 [[buffer(9)]], const device float* MSG [[buffer(10)]], \
    device float* O0 [[buffer(11)]], device float* O1 [[buffer(12)]], device float* O2 [[buffer(13)]], \
    constant i2_controls_args& a [[buffer(14)]], uint3 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]

template <uint C, typename W>
[[kernel]] void i2_controls_apply_cols(I2_CONTROLS_APPLY_ARGS(W)) {
  const uint cols = C == 0u ? a.cols : C;
  const uint col0 = (group.x * 8u + tid / 32u) * cols, lane = tid % 32u, j = group.z;
  if (col0 >= a.h || group.y >= a.rows) return;
  const device float* x = X + (size_t(j) * a.rows + group.y) * a.rank;
  const device float* cur = (j == 0u ? C0 : j == 1u ? C1 : C2) + size_t(a.xb[j] + group.y * a.xs[j]) * a.h;
  const device float* msg = MSG + size_t(a.msg0 + j + group.y * 4u) * a.h;
  device float* out = (j == 0u ? O0 : j == 1u ? O1 : O2) + size_t(group.y) * a.h;
  if (a.rank == 256u) {
    float xv[8];
#pragma unroll
    for (uint s = 0u; s < 8u; ++s) xv[s] = x[lane + s * 32u];
    for (uint col = col0; col < min(col0 + cols, a.h); ++col) {
      const device W* wg = (j == 0u ? G0 : j == 1u ? G1 : G2) + size_t(col) * 256u;
      const device W* wf = (j == 0u ? F0 : j == 1u ? F1 : F2) + size_t(col) * 256u;
      float g = 0.0f, f = 0.0f;
#pragma unroll
      for (uint s = 0u; s < 8u; ++s) {
        g = fma(xv[s], float(wg[lane + s * 32u]), g);
        f = fma(xv[s], float(wf[lane + s * 32u]), f);
      }
      g = simd_sum(g); f = simd_sum(f);
      if (lane == 0u) out[col] = cur[col] - precise::tanh(f) * cur[col] + precise::tanh(g) * msg[col];
    }
    return;
  }
  for (uint col = col0; col < min(col0 + cols, a.h); ++col) {
    const device W* wg = (j == 0u ? G0 : j == 1u ? G1 : G2) + size_t(col) * a.rank;
    const device W* wf = (j == 0u ? F0 : j == 1u ? F1 : F2) + size_t(col) * a.rank;
    float g = 0.0f, f = 0.0f;
    uint base = 0u;
    for (; base + 128u <= a.rank; base += 128u) {
      const uint k = base + lane * 4u;
      const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
      const float4 gv = i2_ctl_load4(wg + k);
      const float4 fv = i2_ctl_load4(wf + k);
      g = fma(xv.x, gv.x, g); g = fma(xv.y, gv.y, g); g = fma(xv.z, gv.z, g); g = fma(xv.w, gv.w, g);
      f = fma(xv.x, fv.x, f); f = fma(xv.y, fv.y, f); f = fma(xv.z, fv.z, f); f = fma(xv.w, fv.w, f);
    }
    for (uint k = base + lane; k < a.rank; k += 32u) {
      g = fma(x[k], float(wg[k]), g); f = fma(x[k], float(wf[k]), f);
    }
    g = simd_sum(g); f = simd_sum(f);
    if (lane == 0u) out[col] = cur[col] - precise::tanh(f) * cur[col] + precise::tanh(g) * msg[col];
  }
}

template [[host_name("i2_controls_apply")]] [[kernel]] void i2_controls_apply_cols<1, float>(I2_CONTROLS_APPLY_ARGS(float));
template [[host_name("i2_controls_apply_rows")]] [[kernel]] void i2_controls_apply_cols<0, float>(I2_CONTROLS_APPLY_ARGS(float));
template [[host_name("i2_controls_apply_bf16")]] [[kernel]] void i2_controls_apply_cols<1, bfloat>(I2_CONTROLS_APPLY_ARGS(bfloat));
template [[host_name("i2_controls_apply_rows_bf16")]] [[kernel]] void i2_controls_apply_cols<0, bfloat>(I2_CONTROLS_APPLY_ARGS(bfloat));

struct i2_qkv_args { uint kv, heads, dim, rel, extent, layout, dyn, tau; float norm, q_threshold, rel_threshold, eps, floor, alpha; };

inline float i2_qkv_conv(const thread float* hist, uint pos, const device bfloat* W, constant i2_qkv_args& a, uint c,
    const device float* D) {
  const uint channels = a.kv * a.dim;
  float sum = hist[0];
  for (uint k = 0; k < 4u; ++k) {
    const uint lag = 3u - k;
    if (lag > pos) continue;
    sum = fma(hist[lag], float(W[a.layout ? (size_t)(3u - k) * channels + c : (size_t)c * 4u + k]), sum);
  }
  if (a.dyn) for (uint lag = 0; lag < 4u; ++lag) if (lag <= pos) sum = fma(hist[lag], D[(c / a.dim) * 4u + lag], sum);
  return sum * a.norm;
}

inline float i2_qkv_total(float sum, threadgroup float* part, uint tid) {
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < 8u; ++i) total += part[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

inline float2 i2_qkv_rows(float kc, float vc, bool live, constant i2_qkv_args& a, threadgroup float* part, uint tid) {
  const float kt = i2_qkv_total(live ? fma(kc, kc, 0.0f) : 0.0f, part, tid);
  const float clip = clamp(a.q_threshold / (sqrt(kt) + 1e-6f), 0.0f, 1.0f);
  const float vt = i2_qkv_total(live ? fma(vc, vc, 0.0f) : 0.0f, part, tid);
  return float2(kc * clip, vc * (1.0f / sqrt(vt / float(a.dim) + a.eps)));
}

inline void i2_qkv_query(device float* Q, const device bfloat* QS, constant i2_qkv_args& a, uint row, uint h,
    threadgroup float* part, uint tid) {
  device float* x = Q + ((size_t)row * a.heads + h) * a.dim;
  const bool live = tid < a.dim;
  const float v = live ? x[tid] * float(QS[(size_t)h * a.dim + tid]) : 0.0f;
  const float total = i2_qkv_total(live ? fma(v, v, 0.0f) : 0.0f, part, tid);
  const float clip = clamp(a.q_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  if (live) x[tid] = v * clip;
}

inline void i2_qkv_rel(const device float* R, const device bfloat* RS, const device bfloat* P, constant i2_qkv_args& a,
    device float* O, uint row, uint block, uint pos, threadgroup float (&rc)[2][16], uint tid) {
  const uint cols = a.extent + 1u, first = block * 256u, h0 = first / cols;
  if (tid < 64u) {
    const uint s = tid / 32u, lane = tid % 32u, h = h0 + s;
    const bool live = h < a.heads && lane < a.rel;
    const float v = live ? R[((size_t)row * a.heads + h) * a.rel + lane] * float(RS[(size_t)h * a.rel + lane]) : 0.0f;
    const float total = simd_sum(live ? fma(v, v, 0.0f) : 0.0f);
    const float clip = clamp(a.rel_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
    if (lane < 16u) rc[s][lane] = v * clip;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint i = first + tid;
  if (i >= cols * a.heads) return;
  const uint h = i / cols, dist = i % cols;
  const size_t out = (size_t)row * cols * a.heads + i;
  if (dist == a.extent) { O[out] = 0.0f; return; }
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(pos) + 1.0f) / a.floor)) : 1.0f;
  float sum = 0.0f;
  for (uint u = 0; u < a.rel; ++u) sum = fma(float(P[u * a.extent + dist]), rc[h - h0][u], sum);
  O[out] = sum * tau;
}

struct i2_batch_qkv_row { device float* k_ring; device float* v_ring; device half* keys; device half* values; uint pos, pad; };

[[kernel]] void i2_batch_qkv(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    const device bfloat* KW [[buffer(2)]], const device bfloat* VW [[buffer(3)]], const device float* KD [[buffer(4)]],
    const device float* VD [[buffer(5)]], device float* Q [[buffer(6)]], const device bfloat* QS [[buffer(7)]],
    const device float* R [[buffer(8)]], const device bfloat* RS [[buffer(9)]], const device bfloat* P [[buffer(10)]],
    constant i2_qkv_args& a [[buffer(11)]], device float* REL [[buffer(12)]],
    constant i2_batch_qkv_row* rows [[buffer(13)]], device float* KC [[buffer(14)]], device float* VC [[buffer(15)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  threadgroup float rc[2][16];
  const uint row = group.y, pos = rows[row].pos;
  if (group.x >= a.kv + a.heads) {
    i2_qkv_rel(R, RS, P, a, REL, row, group.x - a.kv - a.heads, pos, rc, tid);
    return;
  }
  if (group.x >= a.kv) {
    i2_qkv_query(Q, QS, a, row, group.x - a.kv, part, tid);
    return;
  }
  const uint count = a.kv * a.dim, c = group.x * a.dim + tid;
  const bool live = tid < a.dim;
  float kc = 0.0f, vc = 0.0f;
  if (live) {
    float kh[4], vh[4];
    const size_t at = (size_t)row * count + c;
    kh[0] = K[at];
    vh[0] = V[at];
    for (uint lag = 1; lag < 4u; ++lag) {
      const size_t r = lag <= pos ? ((pos - lag) % 4u) * count + c : 0;
      kh[lag] = lag <= pos ? rows[row].k_ring[r] : 0.0f;
      vh[lag] = lag <= pos ? rows[row].v_ring[r] : 0.0f;
    }
    kc = i2_qkv_conv(kh, pos, KW, a, c, KD + (size_t)row * a.kv * 4u);
    vc = i2_qkv_conv(vh, pos, VW, a, c, VD + (size_t)row * a.kv * 4u);
    rows[row].k_ring[(pos % 4u) * count + c] = kh[0];
    rows[row].v_ring[(pos % 4u) * count + c] = vh[0];
  }
  const float2 kv = i2_qkv_rows(kc, vc, live, a, part, tid);
  if (!live) return;
  KC[(size_t)row * count + c] = kv.x;
  VC[(size_t)row * count + c] = vc;
  rows[row].keys[c] = kv.x;
  rows[row].values[c] = kv.y;
}
