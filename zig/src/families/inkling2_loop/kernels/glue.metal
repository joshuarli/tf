// Inkling 2 Mini's dense glue on Metal: one token at a time, f32 activations, the checkpoint's bf16 weights widened
// per read. The arithmetic mirrors campaign/ref/inkling2_loop_ref.py, itself a transcription of llama.cpp 40b7a546.
#include <metal_stdlib>
using namespace metal;

struct i2_mv_args { uint m, k; float cap; };
struct i2_rms_args { uint width, blocks; float eps; };
struct i2_mix_args { uint width, depth, k; float scale; };
struct i2_clip_args { uint heads, width; float threshold; };
struct i2_clip3_args { uint heads, kv, dim, rel; float q_threshold, rel_threshold; };
struct i2_conv_args { uint channels, heads, ring, pos, layout; float norm; };
struct i2_attn_args { uint heads, kv, dim, rel, extent, window, pos, tau; float scale, floor, alpha; };
struct i2_pair_args { uint count; };
struct i2_route_args { uint experts, topk; float route_scale; };

inline float i2_bf(const device bfloat* p, size_t i) { return float(p[i]); }

// out[row] = X . W[row]: a warp an output row of the stored bf16 [m, k] matrix.
[[kernel]] void i2_mv(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2_mv_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 32u + sg;
  if (row >= a.m) return;
  const device bfloat* w = W + (size_t)row * a.k;
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u) acc = fma(X[k], i2_bf(w, k), acc);
  acc = simd_sum(acc);
  if (lane == 0u) O[row] = acc;
}

// The unembedding's row: cap * tanh(X . W[row] / cap).
[[kernel]] void i2_mv_cap(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    constant i2_mv_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = tg * 32u + sg;
  if (row >= a.m) return;
  const device bfloat* w = W + (size_t)row * a.k;
  float acc = 0.0f;
  for (uint k = lane; k < a.k; k += 32u) acc = fma(X[k], i2_bf(w, k), acc);
  acc = simd_sum(acc);
  if (lane == 0u) O[row] = a.cap * precise::tanh(acc / a.cap);
}

// rms_norm of `a.blocks` consecutive blocks of `a.width` values: block b from threadgroup b, weighted unless
// `a.blocks` is zero (the depth mix's states carry no weight). One eps for both paths.
[[kernel]] void i2_rms(const device float* X [[buffer(0)]], const device bfloat* w [[buffer(1)]],
    constant i2_rms_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint blk [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* src = X + (size_t)blk * a.width;
  device float* dst = O + (size_t)blk * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  if (a.blocks == 0u) {
    for (uint i = t; i < a.width; i += nth) dst[i] = src[i] * inv;
  } else {
    const device bfloat* bw = w + (size_t)blk * a.width;
    for (uint i = t; i < a.width; i += nth) dst[i] = src[i] * inv * i2_bf(bw, i);
  }
}

[[kernel]] void i2_rms_gather(const device float* X0 [[buffer(0)]], const device float* X1 [[buffer(1)]],
    const device float* X2 [[buffer(2)]], const device float* X3 [[buffer(3)]],
    const device float* X4 [[buffer(4)]], constant i2_rms_args& a [[buffer(5)]],
    device float* O [[buffer(6)]], uint blk [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* src = blk == 0u ? X0 : blk == 1u ? X1 : blk == 2u ? X2 : blk == 3u ? X3 : X4;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nth) O[(size_t)blk * a.width + i] = src[i] * inv;
}

[[kernel]] void i2_rms_weighted3(const device float* X0 [[buffer(0)]], const device float* X1 [[buffer(1)]],
    const device float* X2 [[buffer(2)]], const device bfloat* W0 [[buffer(3)]],
    const device bfloat* W1 [[buffer(4)]], const device bfloat* W2 [[buffer(5)]],
    constant i2_rms_args& a [[buffer(6)]], device float* O [[buffer(7)]],
    uint blk [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device float* src = blk == 0u ? X0 : blk == 1u ? X1 : X2;
  const device bfloat* weight = blk == 0u ? W0 : blk == 1u ? W1 : W2;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nth) O[(size_t)blk * a.width + i] = src[i] * inv * float(weight[i]);
}

[[kernel]] void i2_rms_kv_store(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    device half* KEYS [[buffer(2)]], device half* VALUES [[buffer(3)]],
    constant i2_rms_args& a [[buffer(4)]], uint blk [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const size_t base = (size_t)blk * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(V[base + i], V[base + i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nth) {
    KEYS[base + i] = K[base + i];
    VALUES[base + i] = V[base + i] * inv;
  }
}

// A token's residual: rms_norm(bf16 embedding row ids[0], embed_norm, eps).
[[kernel]] void i2_embed(const device uint* ids [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    const device bfloat* w [[buffer(2)]], constant i2_rms_args& a [[buffer(3)]], device float* X [[buffer(4)]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const device bfloat* src = W + (size_t)ids[0] * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) {
    const float v = i2_bf(src, i);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float inv = 1.0f / sqrt(total / float(a.width) + a.eps);
  for (uint i = t; i < a.width; i += nth) X[i] = i2_bf(src, i) * inv * i2_bf(w, i);
}

// The depth mix: O[k] = sum_j softmax_j(S[j] . Q[k] * scale) * S[j] over `a.depth` states, each `a.width` wide.
[[kernel]] void i2_mix(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    constant i2_mix_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float w[4][5];
  for (uint k = 0; k < a.k; ++k) {
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = t; i < a.width; i += nth) sum = fma(S[j * a.width + i], Q[k * a.width + i], sum);
      sum = simd_sum(sum);
      if (t % 32u == 0u) part[t / 32u] = sum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
      if (t == 0u) w[k][j] = total * a.scale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
      float peak = w[k][0];
      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < a.width; i += nth) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], S[j * a.width + i], sum);
      O[k * a.width + i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The same with the four query rows in four buffers: Q, K, V, FFN.
[[kernel]] void i2_mix4(const device float* S [[buffer(0)]], const device float* Q [[buffer(1)]],
    const device float* K [[buffer(2)]], const device float* V [[buffer(3)]], const device float* F [[buffer(4)]],
    constant i2_mix_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  threadgroup float w[4][5];
  for (uint k = 0; k < 4u; ++k) {
    const device float* q = k == 0 ? Q : k == 1 ? K : k == 2 ? V : F;
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = t; i < a.width; i += nth) sum = fma(S[j * a.width + i], q[i], sum);
      sum = simd_sum(sum);
      if (t % 32u == 0u) part[t / 32u] = sum;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float total = 0.0f;
      for (uint s = 0; s < nth / 32u; ++s) total += part[s];
      if (t == 0u) w[k][j] = total * a.scale;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0u) {
      float peak = w[k][0];
      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < a.width; i += nth) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], S[j * a.width + i], sum);
      O[k * a.width + i] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// q * min(1, threshold / (|q| + 1e-6)) over the learned per-(head, dim) scale; threadgroup h is head h.
[[kernel]] void i2_clip_scaled(const device float* X [[buffer(0)]], const device bfloat* S [[buffer(1)]],
    constant i2_clip_args& a [[buffer(2)]], device float* O [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const size_t at = (size_t)head * a.width;
  const device bfloat* sc = S + at;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) {
    const float v = X[at + i] * i2_bf(sc, i);
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float clip = clamp(a.threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  for (uint i = t; i < a.width; i += nth) O[at + i] = X[at + i] * i2_bf(sc, i) * clip;
}

[[kernel]] void i2_clip3(device float* Q [[buffer(0)]], const device bfloat* QS [[buffer(1)]],
    device float* R [[buffer(2)]], const device bfloat* RS [[buffer(3)]], device float* K [[buffer(4)]],
    constant i2_clip3_args& a [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const bool q = group < a.heads;
  const bool r = !q && group < 2u * a.heads;
  const uint head = q ? group : r ? group - a.heads : group - 2u * a.heads;
  const uint width = r ? a.rel : a.dim;
  const size_t at = (size_t)head * width;
  device float* x = q ? Q : r ? R : K;
  const device bfloat* scale = q ? QS + at : r ? RS + at : QS;
  float sum = 0.0f;
  for (uint i = t; i < width; i += nth) {
    const float v = r || q ? x[at + i] * float(scale[i]) : x[at + i];
    sum = fma(v, v, sum);
  }
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float threshold = r ? a.rel_threshold : a.q_threshold;
  const float clip = clamp(threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  for (uint i = t; i < width; i += nth) {
    const float v = r || q ? x[at + i] * float(scale[i]) : x[at + i];
    x[at + i] = v * clip;
  }
}

// The same clip with no learned scale (K's, after its convolution).
[[kernel]] void i2_clip(const device float* X [[buffer(0)]], constant i2_clip_args& a [[buffer(1)]],
    device float* O [[buffer(2)]], uint head [[threadgroup_position_in_grid]],
    uint t [[thread_position_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
  threadgroup float part[32];
  const size_t at = (size_t)head * a.width;
  float sum = 0.0f;
  for (uint i = t; i < a.width; i += nth) sum = fma(X[at + i], X[at + i], sum);
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < nth / 32u; ++s) total += part[s];
  const float clip = clamp(a.threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
  for (uint i = t; i < a.width; i += nth) O[at + i] = X[at + i] * clip;
}

// The short convolution for one token: (x + sum_k w[k] * x[pos + k - 3] + sum_lag dyn[h, lag] * x[pos - lag]) / norm.
// RING keeps this site's earlier tokens at slot (pos - n) % ring. `layout` names the stored tap order: 0 is a
// transposed store `[channels, taps]` (tap k of channel c at c*taps + k), 1 a full layer's `static_w` `[taps,
// channels]`, which the reference reverses, so tap k reads row 3 - k.
[[kernel]] void i2_conv(const device float* X [[buffer(0)]], device float* RING [[buffer(1)]],
    const device bfloat* W [[buffer(2)]], const device float* D [[buffer(3)]],
    constant i2_conv_args& a [[buffer(4)]], device float* O [[buffer(5)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels) return;
  const uint c = p.x;
  float sum = X[c];
  for (uint k = 0; k < 4u; ++k) {
    const int at = int(a.pos) + int(k) - 3;
    if (at < 0) continue;
    const float v = at == int(a.pos) ? X[c] : RING[(uint(at) % a.ring) * a.channels + c];
    const size_t tap = a.layout ? (size_t)(3u - k) * a.channels + c : (size_t)c * 4u + k;
    sum = fma(v, i2_bf(W, tap), sum);
  }
  if (a.heads > 0u) {
    const uint per = a.channels / a.heads;
    const uint head = c / per;
    for (uint lag = 0; lag < 4u; ++lag) {
      if (lag > a.pos) continue;
      const float v = lag == 0u ? X[c] : RING[((a.pos - lag) % a.ring) * a.channels + c];
      sum = fma(v, D[head * 4u + lag], sum);
    }
  }
  RING[(a.pos % a.ring) * a.channels + c] = X[c];
  O[c] = sum * a.norm;
}

// Record this token's raw input at slot pos % ring; after i2_conv, which reads the earlier slots.
[[kernel]] void i2_conv_store(const device float* X [[buffer(0)]], device float* RING [[buffer(1)]],
    constant i2_conv_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.channels) RING[(a.pos % a.ring) * a.channels + i] = X[i];
}

// This token's convolved key and normed value into its slot's cache row, for later positions to read.
[[kernel]] void i2_kv_store(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    device float* KEYS [[buffer(2)]], device float* VALUES [[buffer(3)]], constant i2_pair_args& a [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  KEYS[i] = K[i];
  VALUES[i] = V[i];
}

// Relative-position attention for one token, one head a simdgroup: (q . k) / sqrt(D) + B[h, dist] - sink over the
// causal window, with the zero logit the reference's `exp(-peak)` denominator adds. KEYS and VALUES are this slot's
// cache, row j the key/value of the sequence's position j.
[[kernel]] void i2_attn(const device float* Q [[buffer(0)]], const device float* KEYS [[buffer(1)]],
    const device float* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2_attn_args& a [[buffer(5)]], device float* O [[buffer(6)]],
    uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint D = a.dim, per = max(1u, D / 32u);
  const uint kv = h / (a.heads / a.kv);
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  float qv[4];
  for (uint j = 0; j < per; ++j) qv[j] = (lane + 32u * j < D) ? Q[h * D + lane + 32u * j] * a.scale * tau : 0.0f;
  const float sk = sink[h];
  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
  float top = 0.0f, mass = 1.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint j = lo; j <= a.pos; ++j) {
    const uint col = min(a.pos - j, a.extent);
    const size_t at = ((size_t)j * a.kv + kv) * D;
    float dot = 0.0f;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) dot = fma(qv[u], KEYS[at + lane + 32u * u], dot);
    const float score = simd_sum(dot) + B[h * (a.extent + 1u) + col] - sk;
    const float peak = max(top, score);
    const float rescale = precise::exp(top - peak);
    const float w = precise::exp(score - peak);
    mass = mass * rescale + w;
    for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * rescale + w * VALUES[at + lane + 32u * u];
    top = peak;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) O[h * D + lane + 32u * u] = acc[u] / mass;
}

// The relative bias row for one token: B[h, dist] = sum_r proj[r, dist] * R[h, r] * tau for `dist` in [0, extent),
// and 0 at `extent`. The stored projection has `extent` columns; the row above the last distance is the zero it lacks.
[[kernel]] void i2_relbias(const device bfloat* P [[buffer(0)]], const device float* R [[buffer(1)]],
    constant i2_attn_args& a [[buffer(2)]], device float* O [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  const uint cols = a.extent + 1u;
  if (i >= cols * a.heads) return;
  const uint h = i / cols, dist = i % cols;
  if (dist == a.extent) {
    O[i] = 0.0f;
    return;
  }
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  float sum = 0.0f;
  for (uint r = 0; r < a.rel; ++r) sum = fma(i2_bf(P, r * a.extent + dist), R[h * a.rel + r], sum);
  O[i] = sum * tau;
}

// o = a * sigmoid(b).
[[kernel]] void i2_sig_mul(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], device float* O [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  O[i] = A[i] / (1.0f + precise::exp(-B[i]));
}

// a = a - b.
[[kernel]] void i2_sub(device float* A [[buffer(0)]], const device float* B [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  A[i] = A[i] - B[i];
}

// a += b.
[[kernel]] void i2_add(device float* A [[buffer(0)]], const device float* B [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  A[i] += B[i];
}

// o = a * b, elementwise.
[[kernel]] void i2_mul(const device float* A [[buffer(0)]], const device float* B [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], device float* O [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  O[i] = A[i] * B[i];
}

// tanh in place, for a controller's width-wide gates.
// A loop controller's output from its raw gain and forget projections: cur - tanh(forget) * cur + tanh(gain) * message.
[[kernel]] void i2_ctl(const device float* CUR [[buffer(0)]], const device float* MSG [[buffer(1)]],
    const device float* GAIN [[buffer(2)]], const device float* FORGET [[buffer(3)]],
    constant i2_pair_args& a [[buffer(4)]], device float* O [[buffer(5)]], uint i [[thread_position_in_grid]]) {
  if (i >= a.count) return;
  const float gain = precise::tanh(GAIN[i]);
  const float forget = precise::tanh(FORGET[i]);
  O[i] = CUR[i] - forget * CUR[i] + gain * MSG[i];
}

[[kernel]] void i2_copy(const device float* X [[buffer(0)]], device float* O [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) O[i] = X[i];
}

// OUT[j * count + i] = X[i] for every one of `rows` rows: the expert input, one copy a routing pick.
struct i2_bcast_args { uint count, rows; };

[[kernel]] void i2_bcast(const device float* X [[buffer(0)]], constant i2_bcast_args& a [[buffer(1)]],
    device float* O [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.count || p.y >= a.rows) return;
  O[(size_t)p.y * a.count + p.x] = X[p.x];
}

[[kernel]] void i2_zero(device float* O [[buffer(0)]], constant i2_pair_args& a [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
  if (i < a.count) O[i] = 0.0f;
}

// One token's top-k by sigmoid(logits) with the gate bias on the selection only; weights over the unbiased picks.
// One simdgroup: each lane holds experts lane, lane+32, ...; every round reduces (value, index) across lanes with
// ties to the lower index, the order the serial scan kept, and the weight denominator sums in pick order.
[[kernel]] void i2_route(const device float* L [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant i2_route_args& a [[buffer(2)]], device int* picks [[buffer(3)]], device float* wts [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint E = a.experts, K = a.topk;
  float probs[4], sel[4];
  for (uint j = 0; j < 4; ++j) {
    const uint e = lane + 32u * j;
    probs[j] = e < E ? 1.0f / (1.0f + precise::exp(-L[e])) : 0.0f;
    sel[j] = e < E ? probs[j] + bias[e] : NAN;
  }
  float picked[16];
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
    if (lane == 0) picks[k] = int(bi);
    if (bi != E) {
      const uint j = bi / 32u;
      const float p = simd_shuffle(probs[j], ushort(bi % 32u));
      if (lane == bi % 32u) sel[j] = NAN;
      denom += p;
      picked[k] = p;
    } else {
      picked[k] = NAN;
    }
  }
  if (lane == 0) for (uint k = 0; k < K; ++k) wts[k] = picked[k] / (denom + 1e-20f) * a.route_scale;
}

// f32 to bf16 over a row, for a check tool's write-back.
[[kernel]] void i2_to_bf16(const device float* X [[buffer(0)]], device bfloat* O [[buffer(1)]],
    constant i2_pair_args& a [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < a.count) O[i] = bfloat(X[i]);
}

// The expert plan for one token: offsets[e] is expert e's first member in order[], which names its pick slot.
struct i2_plan_args { uint slots, experts; };

[[kernel]] void i2_plan(const device int* picks [[buffer(0)]], constant i2_plan_args& a [[buffer(1)]],
    device int* offsets [[buffer(2)]], device int* order [[buffer(3)]], uint t [[thread_position_in_grid]]) {
  if (t != 0u) return;
  const uint E = a.experts, slots = a.slots;
  for (uint e = 0; e <= E; ++e) offsets[e] = 0;
  for (uint i = 0; i < slots; ++i) offsets[uint(picks[i]) + 1u] += 1;
  for (uint e = 0; e < E; ++e) offsets[e + 1u] += offsets[e];
  int at[128];
  for (uint e = 0; e < E; ++e) at[e] = offsets[e];
  for (uint i = 0; i < slots; ++i) {
    const uint e = uint(picks[i]);
    order[uint(at[e])] = int(i);
    at[e] += 1;
  }
}

// out[c] = sum_j wts[j] * down[j, c], the picks weighted by the normalized route scale.
struct i2_combine_args { uint width, topk; };

[[kernel]] void i2_combine(const device float* DOWN [[buffer(0)]], const device float* wts [[buffer(1)]],
    constant i2_combine_args& a [[buffer(2)]], device float* OUT [[buffer(3)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width) return;
  float sum = 0.0f;
  for (uint j = 0; j < a.topk; ++j) sum = fma(wts[j], DOWN[(size_t)j * a.width + p.x], sum);
  OUT[p.x] = sum;
}
