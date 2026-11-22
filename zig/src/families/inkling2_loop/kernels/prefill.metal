#include <metal_stdlib>
using namespace metal;

struct i2p_dense_args { uint n, k, rows, xs, xb; };
struct i2p_cap_args { uint n, k, rows; float cap; };
struct i2p_router_args { uint n, k, rows, xs, xb; };
struct i2p_pick_args { uint experts, topk, rows; float route_scale; };
struct i2p_embed_args { uint width, rows; float eps; };
struct i2p_rms_args { uint width, rows, xs, xb, ys, yb, wb; float eps; };
struct i2p_ctl_args { uint h, rows, cxs, cxb, mxs, mxb, ys, yb; };
struct i2p_gather_args { uint width, depth, stride; };
struct i2p_mix_args { uint width, depth, k, rows; float scale; };
struct i2p_clip_args { uint heads, width, rows; float threshold; };
struct i2p_conv_args { uint channels, heads, ring, base, rows, layout; float norm; };
struct i2p_cstore_args { uint channels, ring, base, rows; };
struct i2p_kv_args { uint count, base, rows, wring; };
struct i2p_rel_args { uint heads, rel, extent, rows, base, tau; float floor, alpha; };
struct i2p_attn_args { uint heads, kv, dim, rel, extent, window, base, rows, rmode; float scale, floor, alpha; };
struct i2p_moe_args { uint k, n, slots, topk; float gate_cap, up_cap; };
struct i2p_combine_args { uint width, topk, rows; };

inline float i2p_bf(const device bfloat* p, size_t i) { return float(p[i]); }

[[kernel]] void i2p_dense(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_dense_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.n || p.y >= a.rows) return;
  const device bfloat* w = W + (size_t)p.x * a.k;
  const device float* x = X + (size_t)(a.xb + p.y * a.xs) * a.k;
  float acc = 0.0f;
  for (uint k = 0; k < a.k; ++k) acc = fma(x[k], i2p_bf(w, k), acc);
  O[(size_t)p.y * a.n + p.x] = acc;
}

[[kernel]] void i2p_dense_cap(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_cap_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.n || p.y >= a.rows) return;
  const device bfloat* w = W + (size_t)p.x * a.k;
  const device float* x = X + (size_t)p.y * a.k;
  float acc = 0.0f;
  for (uint k = 0; k < a.k; ++k) acc = fma(x[k], i2p_bf(w, k), acc);
  O[(size_t)p.y * a.n + p.x] = a.cap * precise::tanh(acc / a.cap);
}

[[kernel]] void i2p_cap_exact(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_cap_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x * 32u + sg;
  if (row >= a.n) return;
  const device bfloat* w = W + (size_t)row * a.k;
  const device float* x = X + (size_t)tg.y * a.k;
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u) acc = fma(x[k], i2p_bf(w, k), acc);
  acc = simd_sum(acc);
  if (lane == 0u) O[(size_t)tg.y * a.n + row] = a.cap * precise::tanh(acc / a.cap);
}

[[kernel]] void i2p_router_exact(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_router_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg.x * 32u + sg;
  if (row >= a.n) return;
  const device bfloat* w = W + (size_t)row * a.k;
  const device float* x = X + (size_t)(a.xb + tg.y * a.xs) * a.k;
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u) acc = fma(x[k], i2p_bf(w, k), acc);
  acc = simd_sum(acc);
  if (lane == 0u) O[(size_t)tg.y * a.n + row] = acc;
}

struct i2p_gemm_args { uint n, k, rows, xs, xb; float cap; };

[[kernel]] void i2p_gemm(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2p_gemm_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  constexpr int BM = 32, BN = 64, BK = 32, AP = 40, BP = 72;
  const int sg = int(ti) / 32;
  const int m0 = int(tg.y) * BM, n0 = int(tg.x) * BN;
  threadgroup float x[BM * AP];
  threadgroup float w[BK * BP];
  threadgroup float out[BM * BP];
  simdgroup_matrix<float, 8, 8> acc[4] = {simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f), simdgroup_matrix<float, 8, 8>(0.0f),
    simdgroup_matrix<float, 8, 8>(0.0f)};
  for (int k0 = 0; k0 < int(a.k); k0 += BK) {
    for (int e = int(ti); e < BM * BK; e += 256) {
      int r = e / BK, k = e % BK;
      int gr = m0 + r, gk = k0 + k;
      x[r * AP + k] = gr < int(a.rows) && gk < int(a.k)
        ? X[size_t(a.xb + uint(gr) * a.xs) * a.k + uint(gk)] : 0.0f;
    }
    for (int e = int(ti); e < BK * BN; e += 256) {
      int k = e / BN, n = e % BN;
      int gn = n0 + n, gk = k0 + k;
      w[k * BP + n] = gn < int(a.n) && gk < int(a.k)
        ? float(W[size_t(gn) * a.k + uint(gk)]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < BK; k += 8) {
      simdgroup_matrix<float, 8, 8> b;
      simdgroup_load(b, w + k * BP + sg * 8, ulong(BP));
      for (int r = 0; r < 4; ++r) {
        simdgroup_matrix<float, 8, 8> a8;
        simdgroup_load(a8, x + r * 8 * AP + k, ulong(AP));
        simdgroup_multiply_accumulate(acc[r], a8, b, acc[r]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int r = 0; r < 4; ++r) simdgroup_store(acc[r], out + r * 8 * BP + sg * 8, ulong(BP));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = int(ti); e < BM * BN; e += 256) {
    int r = e / BN, n = e % BN;
    if (m0 + r < int(a.rows) && n0 + n < int(a.n)) {
      float v = out[r * BP + n];
      if (a.cap > 0.0f) v = a.cap * precise::tanh(v / a.cap);
      O[size_t(m0 + r) * a.n + uint(n0 + n)] = v;
    }
  }
}

[[kernel]] void i2p_routepick(const device float* L [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant i2p_pick_args& a [[buffer(2)]], device int* picks [[buffer(3)]], device float* wts [[buffer(4)]],
    uint t [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (t >= a.rows) return;
  const uint E = a.experts, K = a.topk;
  const device float* l = L + (size_t)t * E;
  float probs[4], sel[4];
  for (uint j = 0; j < 4; ++j) {
    const uint e = lane + 32u * j;
    probs[j] = e < E ? 1.0f / (1.0f + precise::exp(-l[e])) : 0.0f;
    sel[j] = e < E ? probs[j] + bias[e] : NAN;
  }
  float denom = 0.0f;
  for (uint k = 0; k < K; ++k) {
    float bv = 0.0f;
    uint bi = E;
    for (uint j = 0; j < 4; ++j) {
      const uint e = lane + 32u * j;
      if (e < E && !isnan(sel[j]) && (bi == E || sel[j] > bv)) { bv = sel[j]; bi = e; }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
      const float ov = simd_shuffle_down(bv, off);
      const uint oi = simd_shuffle_down(bi, off);
      if (lane + off < 32u && oi != E && (bi == E || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    }
    bi = simd_broadcast_first(bi);
    if (lane == 0) picks[(size_t)t * K + k] = int(bi);
    if (bi != E) {
      const uint j = bi / 32u;
      const float p = simd_shuffle(probs[j], ushort(bi % 32u));
      if (lane == bi % 32u) sel[j] = NAN;
      denom += p;
      if (lane == 0) wts[(size_t)t * K + k] = p;
    } else {
      if (lane == 0) wts[(size_t)t * K + k] = NAN;
    }
  }
  if (lane == 0) for (uint k = 0; k < K; ++k) wts[(size_t)t * K + k] = wts[(size_t)t * K + k] / (denom + 1e-20f) * a.route_scale;
}

[[kernel]] void i2p_embed(const device uint* ids [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant i2p_embed_args& a [[buffer(3)]], device float* X [[buffer(4)]],
    uint blk [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device bfloat* src = W + (size_t)ids[blk] * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) {
    const float v = i2p_bf(src, i);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  device float* dst = X + (size_t)blk * a.width;
  for (uint i = t; i < a.width; i += nth) dst[i] = i2p_bf(src, i) * inv * i2p_bf(w, i);
}

[[kernel]] void i2p_rms(const device float* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant i2p_rms_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint blk [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* src = X + (size_t)(a.xb + blk * a.xs) * a.width;
  device float* dst = O + (size_t)(a.yb + blk * a.ys) * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  if (a.wb) {
    for (uint i = t; i < a.width; i += nth) dst[i] = src[i] * inv * i2p_bf(w, i);
  } else {
    for (uint i = t; i < a.width; i += nth) dst[i] = src[i] * inv;
  }
}

[[kernel]] void i2p_ctl(const device float* CUR [[buffer(0)]], const device float* MSG [[buffer(1)]],
    const device float* GAIN [[buffer(2)]], const device float* FORGET [[buffer(3)]],
    constant i2p_ctl_args& a [[buffer(4)]], device float* O [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.h || p.y >= a.rows) return;
  const float c = CUR[(size_t)(a.cxb + p.y * a.cxs) * a.h + p.x];
  const float m = MSG[(size_t)(a.mxb + p.y * a.mxs) * a.h + p.x];
  const float g = precise::tanh(GAIN[(size_t)p.y * a.h + p.x]);
  const float f = precise::tanh(FORGET[(size_t)p.y * a.h + p.x]);
  O[(size_t)(a.yb + p.y * a.ys) * a.h + p.x] = c - f * c + g * m;
}

[[kernel]] void i2p_gather(const device float* BLK [[buffer(0)]], const device float* CUR [[buffer(1)]],
    constant i2p_gather_args& a [[buffer(2)]], device float* ST [[buffer(3)]],
    uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  for (uint j = 0; j < a.depth; ++j) {
    const device float* src = j + 1u < a.depth
      ? BLK + ((size_t)t * a.stride + (a.depth - 2u - j)) * a.width
      : CUR + (size_t)t * a.width;
    device float* dst = ST + ((size_t)t * a.depth + j) * a.width;
    for (uint u = i; u < a.width; u += nth) dst[u] = src[u];
  }
}

[[kernel]] void i2p_mix(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    constant i2p_mix_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float w[4][5];
  const device float* st = S + (size_t)t * a.depth * a.width;
  device float* out = O + (size_t)t * a.k * a.width;
  for (uint k = 0; k < a.k; ++k) {
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint u = i; u < a.width; u += nth) sum = fma(st[j * a.width + u], Q[k * a.width + u], sum);
      sum = simd_sum(sum);
      if (i % 32u == 0u) part[i / 32u] = sum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
      if (i == 0u) w[k][j] = total * a.scale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i == 0u) {
      float peak = w[k][0];
      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = i; u < a.width; u += nth) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], st[j * a.width + u], sum);
      out[k * a.width + u] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void i2p_mix4(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    const device float* K [[buffer(2)]], const device float* V [[buffer(3)]], const device float* F [[buffer(4)]],
    constant i2p_mix_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    uint t [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float w[4][5];
  const device float* st = S + (size_t)t * a.depth * a.width;
  device float* out = O + (size_t)t * 4u * a.width;
  for (uint k = 0; k < 4u; ++k) {
    const device float* q = k == 0 ? Q : k == 1 ? K : k == 2 ? V : F;
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint u = i; u < a.width; u += nth) sum = fma(st[j * a.width + u], q[u], sum);
      sum = simd_sum(sum);
      if (i % 32u == 0u) part[i / 32u] = sum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
      if (i == 0u) w[k][j] = total * a.scale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i == 0u) {
      float peak = w[k][0];
      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint u = i; u < a.width; u += nth) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], st[j * a.width + u], sum);
      out[k * a.width + u] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

[[kernel]] void i2p_clip_scaled(const device float* X [[buffer(0)]], const device bfloat* S [[buffer(1)]],
    constant i2p_clip_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint h = g / a.rows, t = g % a.rows;
  const size_t at = ((size_t)t * a.heads + h) * a.width;
  const device bfloat* sc = S + (size_t)h * a.width;
  float sum = 0.0f;
  for (uint u = i; u < a.width; u += nth) {
    const float v = X[at + u] * i2p_bf(sc, u);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (i % 32u == 0u) part[i / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float clip = clamp(a.threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  for (uint u = i; u < a.width; u += nth) O[at + u] = X[at + u] * i2p_bf(sc, u) * clip;
}

[[kernel]] void i2p_clip(const device float* X [[buffer(0)]],
    constant i2p_clip_args& a [[buffer(1)]], device float* O [[buffer(2)]],
    uint g [[threadgroup_position_in_grid]], uint i [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const uint h = g / a.rows, t = g % a.rows;
  const size_t at = ((size_t)t * a.heads + h) * a.width;
  float sum = 0.0f;
  for (uint u = i; u < a.width; u += nth) sum = fma(X[at + u], X[at + u], sum);
  sum = simd_sum(sum);
  if (i % 32u == 0u) part[i / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float clip = clamp(a.threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  for (uint u = i; u < a.width; u += nth) O[at + u] = X[at + u] * clip;
}

[[kernel]] void i2p_conv(const device float* X [[buffer(0)]], const device float* RING [[buffer(1)]],
    const device bfloat* W [[buffer(2)]], const device float* D [[buffer(3)]],
    constant i2p_conv_args& a [[buffer(4)]], device float* O [[buffer(5)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels || p.y >= a.rows) return;
  const uint c = p.x, t = p.y, pos = a.base + t;
  const device float* x = X + (size_t)t * a.channels;
  float sum = x[c];
  for (uint k = 0; k < 4u; ++k) {
    const int at = int(pos) + int(k) - 3;
    if (at < 0) continue;
    float v;
    if (at == int(pos)) v = x[c];
    else if ((uint)at >= a.base) v = X[(size_t)((uint)at - a.base) * a.channels + c];
    else v = RING[((uint)at % a.ring) * a.channels + c];
    const size_t tap = a.layout ? (size_t)(3u - k) * a.channels + c : (size_t)c * 4u + k;
    sum = fma(v, i2p_bf(W, tap), sum);
  }
  if (a.heads > 0u) {
    const uint per = a.channels / a.heads;
    const uint head = c / per;
    for (uint lag = 0; lag < 4u; ++lag) {
      if (lag > pos) continue;
      float v;
      if (lag == 0u) v = x[c];
      else if (pos - lag >= a.base) v = X[(size_t)(pos - lag - a.base) * a.channels + c];
      else v = RING[((pos - lag) % a.ring) * a.channels + c];
      sum = fma(v, D[(size_t)t * a.heads * 4u + head * 4u + lag], sum);
    }
  }
  O[(size_t)t * a.channels + c] = sum * a.norm;
}

[[kernel]] void i2p_conv_store(const device float* X [[buffer(0)]], device float* RING [[buffer(1)]],
    constant i2p_cstore_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels || p.y >= a.rows) return;
  for (uint q = p.y + a.ring; q < a.rows; q += a.ring) {
    if ((a.base + q) % a.ring == (a.base + p.y) % a.ring) return;
  }
  RING[((a.base + p.y) % a.ring) * a.channels + p.x] = X[(size_t)p.y * a.channels + p.x];
}

[[kernel]] void i2p_kv_store(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    device float* KEYS [[buffer(2)]], device float* VALUES [[buffer(3)]], constant i2p_kv_args& a [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.count || p.y >= a.rows) return;
  const uint r = a.wring ? (a.base + p.y) % a.wring : a.base + p.y;
  KEYS[(size_t)r * a.count + p.x] = K[(size_t)p.y * a.count + p.x];
  VALUES[(size_t)r * a.count + p.x] = V[(size_t)p.y * a.count + p.x];
}

[[kernel]] void i2p_relbias(const device bfloat* P [[buffer(0)]], const device float* R [[buffer(1)]],
    constant i2p_rel_args& a [[buffer(2)]], device float* O [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  const uint cols = a.extent + 1u;
  if (p.x >= cols * a.heads || p.y >= a.rows) return;
  const uint h = p.x / cols, dist = p.x % cols;
  const uint pos = a.base + p.y;
  if (dist == a.extent) {
    O[(size_t)p.y * cols * a.heads + p.x] = 0.0f;
    return;
  }
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(pos) + 1.0f) / a.floor)) : 1.0f;
  float sum = 0.0f;
  const device float* r = R + ((size_t)p.y * a.heads + h) * a.rel;
  for (uint u = 0; u < a.rel; ++u) sum = fma(i2p_bf(P, u * a.extent + dist), r[u], sum);
  O[(size_t)p.y * cols * a.heads + p.x] = sum * tau;
}

[[kernel]] void i2p_attn(const device float* Q [[buffer(0)]], const device float* KEYS [[buffer(1)]],
    const device float* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2p_attn_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    const device float* KC [[buffer(7)]], const device float* VC [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint h = tg.x, t = tg.y;
  if (h >= a.heads || t >= a.rows) return;
  const uint D = a.dim, per = max(1u, D / 32u);
  const uint kv = h / (a.heads / a.kv);
  const uint pos = a.base + t;
  const float tau = 1.0f + a.alpha * precise::log(max(1.0f, (float(pos) + 1.0f) / a.floor));
  const float use_tau = a.window ? 1.0f : tau;
  float qv[4];
  const device float* q = Q + (size_t)t * a.heads * D;
  for (uint j = 0; j < per; ++j) qv[j] = (lane + 32u * j < D) ? q[h * D + lane + 32u * j] * a.scale * use_tau : 0.0f;
  const float sk = sink[(size_t)t * a.heads + h];
  const uint lo = a.window ? (pos + 1u > a.window ? pos + 1u - a.window : 0u) : 0u;
  const device float* b = B + (size_t)t * a.heads * (a.extent + 1u);
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint j = lo; j <= pos; ++j) {
    const uint col = min(pos - j, a.extent);
    const uint rj = (a.rmode && a.window) ? j % a.window : j;
    float dot = 0.0f;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) {
      const size_t at = (j < a.base ? (size_t)rj : (size_t)(j - a.base)) * a.kv + kv;
      const float kk = (j < a.base ? KEYS[at * D + lane + 32u * u] : KC[at * D + lane + 32u * u]);
      dot = fma(qv[u], kk, dot);
    }
    const float score = simd_sum(dot) + b[h * (a.extent + 1u) + col] - sk;
    const float peak = max(top, score);
    const float rescale = precise::exp(top - peak);
    const float w = precise::exp(score - peak);
    mass = mass * rescale + w;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) {
      const size_t at = (j < a.base ? (size_t)rj : (size_t)(j - a.base)) * a.kv + kv;
      const float vv = (j < a.base ? VALUES[at * D + lane + 32u * u] : VC[at * D + lane + 32u * u]);
      acc[u] = acc[u] * rescale + w * vv;
    }
    top = peak;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) O[(size_t)t * a.heads * D + h * D + lane + 32u * u] = acc[u] / mass;
}

[[kernel]] void i2p_attn_tiled(const device float* Q [[buffer(0)]], const device float* KEYS [[buffer(1)]],
    const device float* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2p_attn_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    const device float* KC [[buffer(7)]], const device float* VC [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint ti [[thread_index_in_threadgroup]]) {
  const uint sg = ti / 32u, lane = ti % 32u, kv = tg.x, row0 = tg.y * 8u;
  const uint t = row0 + sg % 8u, h = kv * 3u + sg / 8u;
  const uint pos = a.base + t;
  const float tau = 1.0f + a.alpha * precise::log(max(1.0f, (float(pos) + 1.0f) / a.floor));
  const float use_tau = a.window ? 1.0f : tau;
  float qv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  if (t < a.rows) for (uint u = 0; u < 4u; ++u)
    qv[u] = Q[((size_t)t * a.heads + h) * 128u + lane + 32u * u] * a.scale * use_tau;
  const float sk = t < a.rows ? sink[(size_t)t * a.heads + h] : 0.0f;
  const uint lo = a.window && pos + 1u > a.window ? pos + 1u - a.window : 0u;
  const uint first = a.base + row0;
  const uint lo0 = a.window && first + 1u > a.window ? first + 1u - a.window : 0u;
  const uint last = a.base + min(row0 + 7u, a.rows - 1u);
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  threadgroup float keys[8 * 128], values[8 * 128];
  for (uint j0 = lo0 / 8u * 8u; j0 <= last; j0 += 8u) {
    for (uint e = ti; e < 8u * 128u; e += 768u) {
      const uint j = j0 + e / 128u, d = e % 128u;
      float kk = 0.0f, vv = 0.0f;
      if (j <= last) {
        const uint rj = a.rmode && a.window ? j % a.window : j;
        const size_t at = ((j < a.base ? (size_t)rj : (size_t)(j - a.base)) * a.kv + kv) * 128u + d;
        kk = j < a.base ? KEYS[at] : KC[at];
        vv = j < a.base ? VALUES[at] : VC[at];
      }
      keys[e] = kk; values[e] = vv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < a.rows) for (uint x = 0; x < 8u; ++x) {
      const uint j = j0 + x;
      if (j < lo || j > pos) continue;
      float dot = 0.0f;
      for (uint u = 0; u < 4u; ++u) dot = fma(qv[u], keys[x * 128u + lane + 32u * u], dot);
      const uint col = min(pos - j, a.extent);
      const float score = simd_sum(dot) + B[((size_t)t * a.heads + h) * (a.extent + 1u) + col] - sk;
      const float peak = max(top, score);
      const float rescale = precise::exp(top - peak);
      const float w = precise::exp(score - peak);
      mass = mass * rescale + w;
      for (uint u = 0; u < 4u; ++u) acc[u] = acc[u] * rescale + w * values[x * 128u + lane + 32u * u];
      top = peak;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t < a.rows) for (uint u = 0; u < 4u; ++u)
    O[((size_t)t * a.heads + h) * 128u + lane + 32u * u] = acc[u] / mass;
}

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

inline void i2p_table_pair(const device uchar* wg, const device uchar* wu, const device float* x,
    uint k0, const device float* codes, thread float& dg, thread float& du) {
#ifdef I2P_MXFP8
  const ulong ag[4] = { *(const device ulong*)(wg), *(const device ulong*)(wg + 8),
                        *(const device ulong*)(wg + 16), *(const device ulong*)(wg + 24) };
  const ulong au[4] = { *(const device ulong*)(wu), *(const device ulong*)(wu + 8),
                        *(const device ulong*)(wu + 16), *(const device ulong*)(wu + 24) };
  for (uint q = 0; q < 4u; ++q) {
    for (uint j = 0; j < 8u; ++j) {
      const float xv = x[k0 + q * 8u + j];
      dg = fma(codes[i2p_lane_byte(ag[q], j)], xv, dg);
      du = fma(codes[i2p_lane_byte(au[q], j)], xv, du);
    }
  }
#else
  const ulong ag = *(const device ulong*)(wg);
  const ulong au = *(const device ulong*)(wu);
  for (uint j = 0; j < 8u; ++j) {
    const uint ab = i2p_lane_byte(ag, j);
    const uint ub = i2p_lane_byte(au, j);
    const float x0 = x[k0 + 2u * j];
    const float x1 = x[k0 + 2u * j + 1u];
    dg = fma(codes[ab & 15u], x0, dg);
    dg = fma(codes[ab >> 4u], x1, dg);
    du = fma(codes[ub & 15u], x0, du);
    du = fma(codes[ub >> 4u], x1, du);
  }
#endif
}

inline void i2p_table_row(const device uchar* w, const device float* x,
    uint k0, const device float* codes, thread float& d) {
#ifdef I2P_MXFP8
  const ulong a[4] = { *(const device ulong*)(w), *(const device ulong*)(w + 8),
                       *(const device ulong*)(w + 16), *(const device ulong*)(w + 24) };
  for (uint q = 0; q < 4u; ++q) {
    for (uint j = 0; j < 8u; ++j) d = fma(codes[i2p_lane_byte(a[q], j)], x[k0 + q * 8u + j], d);
  }
#else
  const ulong a = *(const device ulong*)(w);
  for (uint j = 0; j < 8u; ++j) {
    const uint b = i2p_lane_byte(a, j);
    d = fma(codes[b & 15u], x[k0 + 2u * j], d);
    d = fma(codes[b >> 4u], x[k0 + 2u * j + 1u], d);
  }
#endif
}

[[kernel]] void i2p_gu(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2p_moe_args& a [[buffer(6)]], device float* ACT [[buffer(7)]],
    const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x;
  const uint slot = tg.y;
  if (c >= a.n || slot >= a.slots) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = X + (size_t)(slot / a.topk) * a.k;
  const size_t row = (size_t)e * a.n * 2u + 2u * c;
  const device uchar* wg = W + row * I2P_CODE_STRIDE(a.k);
  const device uchar* wu = wg + I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  float dg = 0.0f;
  float du = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2P_GROUP;
    float pg = 0.0f;
    float pu = 0.0f;
    i2p_table_pair(wg + I2P_BYTE_OFF(k0), wu + I2P_BYTE_OFF(k0), x, k0, CODES, pg, pu);
    dg = fma(i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g), pg, dg);
    du = fma(i2p_decode_scale(S, (row + 1u) * I2P_SCALE_STRIDE(a.k) + g), pu, du);
  }
  dg = simd_sum(dg);
  du = simd_sum(du);
  if (lane == 0u) {
#ifdef I2P_MXFP8
    ACT[(size_t)slot * a.n + c] = i2p_situ(dg, du, a.gate_cap, a.up_cap);
#else
    const float global = amax[e] / den[0];
    ACT[(size_t)slot * a.n + c] = i2p_situ(dg * global, du * global, a.gate_cap, a.up_cap);
#endif
  }
}

[[kernel]] void i2p_down(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* PICKS [[buffer(5)]],
    constant i2p_moe_args& a [[buffer(6)]], device float* OUT [[buffer(7)]],
    const device float* CODES [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint r = tg.x;
  const uint slot = tg.y;
  if (r >= a.n || slot >= a.slots) return;
  const uint e = uint(PICKS[slot]);
  const device float* x = ACT + (size_t)slot * a.k;
  const size_t row = (size_t)e * a.n + r;
  const device uchar* w = W + row * I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  float d = 0.0f;
  for (uint g = lane; g < groups; g += 32u) {
    const uint k0 = g * I2P_GROUP;
    float p = 0.0f;
    i2p_table_row(w + I2P_BYTE_OFF(k0), x, k0, CODES, p);
    d = fma(i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g), p, d);
  }
  d = simd_sum(d);
  if (lane == 0u) {
#ifdef I2P_MXFP8
    OUT[(size_t)slot * a.n + r] = d;
#else
    OUT[(size_t)slot * a.n + r] = d * (amax[e] / den[0]);
#endif
  }
}

struct i2p_moeg_args { uint k, n, slots, topk, experts; float gate_cap, up_cap; };

[[kernel]] void i2p_moe_zero(device uint* CNT [[buffer(0)]], constant uint& E [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i < E) CNT[i] = 0u;
}

[[kernel]] void i2p_moe_count(const device int* PICKS [[buffer(0)]], device atomic_uint* CNT [[buffer(1)]],
    constant uint& slots [[buffer(2)]], uint s [[thread_position_in_grid]]) {
  if (s >= slots) return;
  uint e = uint(PICKS[s]);
  atomic_fetch_add_explicit(CNT + e, 1u, memory_order_relaxed);
}

[[kernel]] void i2p_moe_offsets(const device uint* CNT [[buffer(0)]], device int* OFF [[buffer(1)]],
    device uint* CUR [[buffer(2)]], constant uint& E [[buffer(3)]],
    uint t [[thread_index_in_threadgroup]]) {
  threadgroup uint counts[128];
  if (t < E) counts[t] = CNT[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0u) {
    uint sum = 0u;
    for (uint i = 0u; i < E; ++i) {
      OFF[i] = int(sum);
      CUR[i] = sum;
      sum += counts[i];
    }
    OFF[E] = int(sum);
  }
}

[[kernel]] void i2p_moe_sort(const device int* PICKS [[buffer(0)]], device atomic_uint* CUR [[buffer(1)]],
    device int* ROW_OF [[buffer(2)]], constant uint& slots [[buffer(3)]],
    uint s [[thread_position_in_grid]]) {
  if (s >= slots) return;
  uint e = uint(PICKS[s]);
  uint pos = atomic_fetch_add_explicit(CUR + e, 1u, memory_order_relaxed);
  ROW_OF[pos] = int(s);
}

[[kernel]] void i2p_gu_grouped(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ROW_OF [[buffer(6)]], constant i2p_moeg_args& a [[buffer(7)]],
    device float* ACT [[buffer(8)]], const device float* CODES [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint c = tg.x;
  const uint e = tg.y;
  if (c >= a.n || e >= a.experts) return;
  const int start = OFF[e];
  const int end = OFF[e + 1u];
  if (start >= end || end - start >= I2P_MATRIX_THRESHOLD) return;
  const size_t row = (size_t)e * a.n * 2u + 2u * c;
  const device uchar* wg_base = W + row * I2P_CODE_STRIDE(a.k);
  const device uchar* wu_base = wg_base + I2P_CODE_STRIDE(a.k);
  const uint groups = a.k / I2P_GROUP;
  const uint per = (groups + 31u) / 32u;
  ulong wg_cache[8];
  ulong wu_cache[8];
#ifdef I2P_MXFP8
  ulong wg1_cache[8];
  ulong wg2_cache[8];
  ulong wg3_cache[8];
  ulong wu1_cache[8];
  ulong wu2_cache[8];
  ulong wu3_cache[8];
#endif
  float scg_cache[8];
  float scu_cache[8];
  for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
    const uint k0 = g * I2P_GROUP;
#ifdef I2P_MXFP8
    const device ulong* pg = (const device ulong*)(wg_base + k0);
    const device ulong* pu = (const device ulong*)(wu_base + k0);
    wg_cache[idx] = pg[0]; wg1_cache[idx] = pg[1]; wg2_cache[idx] = pg[2]; wg3_cache[idx] = pg[3];
    wu_cache[idx] = pu[0]; wu1_cache[idx] = pu[1]; wu2_cache[idx] = pu[2]; wu3_cache[idx] = pu[3];
#else
    wg_cache[idx] = *(const device ulong*)(wg_base + I2P_BYTE_OFF(k0));
    wu_cache[idx] = *(const device ulong*)(wu_base + I2P_BYTE_OFF(k0));
#endif
    scg_cache[idx] = i2p_decode_scale(S, row * I2P_SCALE_STRIDE(a.k) + g);
    scu_cache[idx] = i2p_decode_scale(S, (row + 1u) * I2P_SCALE_STRIDE(a.k) + g);
  }
#ifdef I2P_MXFP8
  const float has_global = 0.0f;
#else
  const float global = amax[e] / den[0];
#endif
  for (int oi = start; oi < end; ++oi) {
    const uint slot = uint(ROW_OF[oi]);
    const device float* x = X + (size_t)(slot / a.topk) * a.k;
    float dg = 0.0f;
    float du = 0.0f;
    for (uint idx = 0u, g = lane; g < groups; ++idx, g += 32u) {
      const uint k0 = g * I2P_GROUP;
      float pg = 0.0f;
      float pu = 0.0f;
#ifdef I2P_MXFP8
      const ulong ag[4] = { wg_cache[idx], wg1_cache[idx], wg2_cache[idx], wg3_cache[idx] };
      const ulong au[4] = { wu_cache[idx], wu1_cache[idx], wu2_cache[idx], wu3_cache[idx] };
      for (uint q = 0u; q < 4u; ++q) {
        for (uint j = 0u; j < 8u; ++j) {
          const float xv = x[k0 + q * 8u + j];
          pg = fma(CODES[i2p_lane_byte(ag[q], j)], xv, pg);
          pu = fma(CODES[i2p_lane_byte(au[q], j)], xv, pu);
        }
      }
#else
      const ulong ag = wg_cache[idx];
      const ulong au = wu_cache[idx];
      for (uint j = 0u; j < 8u; ++j) {
        const uint ab = i2p_lane_byte(ag, j);
        const uint ub = i2p_lane_byte(au, j);
        const float x0 = x[k0 + 2u * j];
        const float x1 = x[k0 + 2u * j + 1u];
        pg = fma(CODES[ab & 15u], x0, pg);
        pg = fma(CODES[ab >> 4u], x1, pg);
        pu = fma(CODES[ub & 15u], x0, pu);
        pu = fma(CODES[ub >> 4u], x1, pu);
      }
#endif
      dg = fma(scg_cache[idx], pg, dg);
      du = fma(scu_cache[idx], pu, du);
    }
    dg = simd_sum(dg);
    du = simd_sum(du);
    if (lane == 0u) {
#ifdef I2P_MXFP8
      ACT[(size_t)slot * a.n + c] = i2p_situ(dg, du, a.gate_cap, a.up_cap);
#else
      ACT[(size_t)slot * a.n + c] = i2p_situ(dg * global, du * global, a.gate_cap, a.up_cap);
#endif
    }
  }
}

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

[[kernel]] void i2p_combine(const device float* DOWN [[buffer(0)]], const device float* wts [[buffer(1)]],
    constant i2p_combine_args& a [[buffer(2)]], device float* OUT [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  float sum = 0.0f;
  for (uint j = 0; j < a.topk; ++j) sum = fma(wts[(size_t)p.y * a.topk + j], DOWN[((size_t)p.y * a.topk + j) * a.width + p.x], sum);
  OUT[(size_t)p.y * a.width + p.x] = sum;
}
