#include <metal_stdlib>
using namespace metal;

struct i2_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };

[[kernel]] void i2_attn_split(const device float* Q [[buffer(0)]], const device float* KEYS [[buffer(1)]],
    const device float* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2_attn_parts& a [[buffer(5)]], device float* PART [[buffer(6)]],
    uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint nth [[threads_per_threadgroup]]) {
  const uint G = a.heads / a.kv;
  const uint D = a.dim, per = max(1u, D / 32u);
  const uint kv = tgp / a.splits, s = tgp % a.splits, h = kv * G + sg;
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
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
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint block = js; block < je; block += 8u) {
    const uint j = block + lane / 4u;
    const uint rj = a.window ? (j % a.window) : j;
    const size_t at = ((size_t)rj * a.kv + kv) * D;
    float dot = 0.0f;
    if (j < je) {
      for (uint d = (lane % 4u) * 4u; d < D; d += 16u) {
        const float4 q = *(const device float4*)(Q + h * D + d) * (a.scale * tau);
        const float4 k = *(const device float4*)(KEYS + at + d);
        dot = fma(q.x, k.x, dot);
        dot = fma(q.y, k.y, dot);
        dot = fma(q.z, k.z, dot);
        dot = fma(q.w, k.w, dot);
      }
    }
    dot += simd_shuffle_xor(dot, 1u);
    dot += simd_shuffle_xor(dot, 2u);
    const float score = j < je ? dot + B[h * (a.extent + 1u) + min(a.pos - j, a.extent)] - sk : -INFINITY;
    for (uint k = 0; k < min(8u, je - block); ++k) {
      const float sc = simd_shuffle(score, k * 4u);
      const float peak = max(top, sc);
      const float rescale = precise::exp(top - peak);
      const float w = precise::exp(sc - peak);
      mass = mass * rescale + w;
      const uint row = a.window ? ((block + k) % a.window) : block + k;
      const size_t vbase = ((size_t)row * a.kv + kv) * D;
      for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) acc[u] = acc[u] * rescale + w * VALUES[vbase + lane + 32u * u];
      top = peak;
    }
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  for (uint u = 0; u < per; ++u) if (lane + 32u * u < D) PART[base + 2u + lane + 32u * u] = acc[u];
}

[[kernel]] void i2_attn_combine(const device float* PART [[buffer(0)]],
    constant i2_attn_parts& a [[buffer(1)]], device float* O [[buffer(2)]],
    uint h [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
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
