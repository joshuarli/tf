#include <metal_stdlib>
using namespace metal;

struct i2_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };

[[kernel]] void i2_attn_split(const device float* Q [[buffer(0)]], const device half* KEYS [[buffer(1)]],
    const device half* VALUES [[buffer(2)]], const device float* B [[buffer(3)]], const device float* sink [[buffer(4)]],
    constant i2_attn_parts& a [[buffer(5)]], device float* PART [[buffer(6)]],
    uint tgp [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint nth [[threads_per_threadgroup]]) {
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
  threadgroup float4 KB4[32];
  threadgroup float4 VB4[32];
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

struct i2_batch_row { device half* keys; device half* values; i2_attn_parts args; };

[[kernel]] void i2_batch_split(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2_batch_row* rows [[buffer(3)]], device float* PART [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 nth2 [[threads_per_threadgroup]]) {
  const uint tid = tid2.x, nth = nth2.x;
  constant i2_attn_parts& a = rows[tg.y].args;
  const device half* KEYS = rows[tg.y].keys;
  const device half* VALUES = rows[tg.y].values;
  Q += size_t(tg.y) * a.heads * a.dim;
  B += size_t(tg.y) * a.heads * (a.extent + 1u);
  sink += size_t(tg.y) * a.heads;
  PART += size_t(tg.y) * a.heads * 16u * (a.dim + 2u);
  const uint tgp = tg.x;
  if (tgp >= a.kv * a.splits) return;
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
  threadgroup float4 KB4[32];
  threadgroup float4 VB4[32];
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


[[kernel]] void i2_batch_combine(const device float* PART [[buffer(0)]], constant i2_batch_row* rows [[buffer(1)]],
    device float* O [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  constant i2_attn_parts& a = rows[tg.y].args;
  const uint h = tg.x;
  PART += size_t(tg.y) * a.heads * 16u * (a.dim + 2u);
  O += size_t(tg.y) * a.heads * a.dim;
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

struct i2_controls_args { uint h, rank, rows; uint xs[3], xb[3]; };

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
    : MSG + size_t(controller + row * 4u) * a.h;
  device float* dst = OUT + (size_t(controller) * a.rows + row) * 2u * a.h + which * a.h;
  float sum = 0.0f;
  for (uint i = tid; i < a.h; i += nth) sum = fma(src[i], src[i], sum);
  sum = simd_sum(sum);
  if (tid % 32u == 0u) part[tid / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint i = 0; i < nth / 32u; ++i) total += part[i];
  const float inv = 1.0f / sqrt(total / float(a.h) + 1e-6f);
  for (uint i = tid; i < a.h; i += nth) dst[i] = src[i] * inv;
}

[[kernel]] void i2_controls_down(const device float* X [[buffer(0)]], const device float* W0 [[buffer(1)]],
    const device float* W1 [[buffer(2)]], const device float* W2 [[buffer(3)]],
    constant i2_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u;
  if (col >= a.rank || group.y >= a.rows) return;
  const uint width = 2u * a.h;
  const device float* x = X + (size_t(group.z) * a.rows + group.y) * width;
  const device float* w = (group.z == 0u ? W0 : group.z == 1u ? W1 : W2) + size_t(col) * width;
  float acc = 0.0f;
  uint base = 0u;
  for (; base + 128u <= width; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
    const float4 wv = float4(*reinterpret_cast<const device packed_float4*>(w + k));
    acc = fma(xv.x, wv.x, acc); acc = fma(xv.y, wv.y, acc);
    acc = fma(xv.z, wv.z, acc); acc = fma(xv.w, wv.w, acc);
  }
  for (uint k = base + lane; k < width; k += 32u) acc = fma(x[k], w[k], acc);
  acc = simd_sum(acc);
  if (lane == 0u) OUT[(size_t(group.z) * a.rows + group.y) * a.rank + col] = acc;
}

[[kernel]] void i2_controls_apply(const device float* X [[buffer(0)]],
    const device float* G0 [[buffer(1)]], const device float* G1 [[buffer(2)]], const device float* G2 [[buffer(3)]],
    const device float* F0 [[buffer(4)]], const device float* F1 [[buffer(5)]], const device float* F2 [[buffer(6)]],
    const device float* C0 [[buffer(7)]], const device float* C1 [[buffer(8)]], const device float* C2 [[buffer(9)]],
    const device float* MSG [[buffer(10)]], device float* O0 [[buffer(11)]], device float* O1 [[buffer(12)]],
    device float* O2 [[buffer(13)]], constant i2_controls_args& a [[buffer(14)]],
    uint3 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u, lane = tid % 32u, j = group.z;
  if (col >= a.h || group.y >= a.rows) return;
  const device float* x = X + (size_t(j) * a.rows + group.y) * a.rank;
  const device float* wg = (j == 0u ? G0 : j == 1u ? G1 : G2) + size_t(col) * a.rank;
  const device float* wf = (j == 0u ? F0 : j == 1u ? F1 : F2) + size_t(col) * a.rank;
  const device float* cur = (j == 0u ? C0 : j == 1u ? C1 : C2) + size_t(a.xb[j] + group.y * a.xs[j]) * a.h;
  const device float* msg = MSG + size_t(j + group.y * 4u) * a.h;
  device float* out = (j == 0u ? O0 : j == 1u ? O1 : O2) + size_t(group.y) * a.h;
  float g = 0.0f, f = 0.0f;
  uint base = 0u;
  for (; base + 128u <= a.rank; base += 128u) {
    const uint k = base + lane * 4u;
    const float4 xv = float4(*reinterpret_cast<const device packed_float4*>(x + k));
    const float4 gv = float4(*reinterpret_cast<const device packed_float4*>(wg + k));
    const float4 fv = float4(*reinterpret_cast<const device packed_float4*>(wf + k));
    g = fma(xv.x, gv.x, g); g = fma(xv.y, gv.y, g); g = fma(xv.z, gv.z, g); g = fma(xv.w, gv.w, g);
    f = fma(xv.x, fv.x, f); f = fma(xv.y, fv.y, f); f = fma(xv.z, fv.z, f); f = fma(xv.w, fv.w, f);
  }
  for (uint k = base + lane; k < a.rank; k += 32u) {
    g = fma(x[k], wg[k], g); f = fma(x[k], wf[k], f);
  }
  g = simd_sum(g); f = simd_sum(f);
  if (lane == 0u) out[col] = cur[col] - precise::tanh(f) * cur[col] + precise::tanh(g) * msg[col];
}
