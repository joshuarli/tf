#include <metal_stdlib>
using namespace metal;

struct i2pr_mix_args { uint width, depth, stride, k, norm; float scale, eps; };

constant constexpr uint I2PR_THREADS = 256u;
constant constexpr uint I2PR_PER = 9u;
constant constexpr uint I2PR_DEPTH = 5u;

inline float i2pr_total(float sum, threadgroup float* part, uint t) {
  sum = simd_sum(sum);
  if (t % 32u == 0u) part[t / 32u] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (uint s = 0; s < I2PR_THREADS / 32u; ++s) total += part[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return total;
}

[[kernel]] void i2pr_mix_norm(const device float* BLK [[buffer(0)]], const device float* CUR [[buffer(1)]],
    const device float* Q0 [[buffer(2)]], const device float* Q1 [[buffer(3)]], const device float* Q2 [[buffer(4)]],
    const device float* Q3 [[buffer(5)]], constant i2pr_mix_args& a [[buffer(6)]], device float* MIX [[buffer(7)]],
    const device bfloat* W0 [[buffer(8)]], const device bfloat* W1 [[buffer(9)]], const device bfloat* W2 [[buffer(10)]],
    device float* O0 [[buffer(11)]], device float* O1 [[buffer(12)]], device float* O2 [[buffer(13)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]]) {
  threadgroup float part[I2PR_DEPTH][I2PR_THREADS / 32u];
  threadgroup float w[4][I2PR_DEPTH];
  float src[I2PR_DEPTH][I2PR_PER];
  float inv[I2PR_DEPTH];
  const uint sg = t / 32u, lane = t % 32u;
  for (uint j = 0; j < a.depth; ++j) {
    const device float* s = j + 1u < a.depth ? BLK + ((size_t)row * a.stride + (a.depth - 2u - j)) * a.width
                                             : CUR + (size_t)row * a.width;
    float sum = 0.0f;
    for (uint i = 0; i < I2PR_PER; ++i) {
      const float v = s[t + i * I2PR_THREADS];
      src[j][i] = v;
      sum = fma(v, v, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) part[j][sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < a.depth; ++j) {
    float total = 0.0f;
    for (uint s = 0; s < I2PR_THREADS / 32u; ++s) total += part[j][s];
    inv[j] = 1.0f / sqrt(total / float(a.width) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < a.k; ++k) {
    const device float* q = k == 0u ? Q0 : k == 1u ? Q1 : k == 2u ? Q2 : Q3;
    float qv[I2PR_PER];
    for (uint i = 0; i < I2PR_PER; ++i) qv[i] = q[t + i * I2PR_THREADS];
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = 0; i < I2PR_PER; ++i) sum = fma(src[j][i] * inv[j], qv[i], sum);
      sum = simd_sum(sum);
      if (lane == 0u) part[j][sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0u) {
      float peak = -INFINITY;
      for (uint j = 0; j < a.depth; ++j) {
        float total = 0.0f;
        for (uint s = 0; s < I2PR_THREADS / 32u; ++s) total += part[j][s];
        w[k][j] = total * a.scale;
        peak = max(peak, w[k][j]);
      }
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) {
        const float e = precise::exp(w[k][j] - peak);
        w[k][j] = e;
        total += e;
      }
      for (uint j = 0; j < a.depth; ++j) w[k][j] = w[k][j] / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float out[I2PR_PER];
    device float* mix = MIX + ((size_t)row * a.k + k) * a.width;
    float sum2 = 0.0f;
    for (uint i = 0; i < I2PR_PER; ++i) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], src[j][i] * inv[j], sum);
      out[i] = sum;
      mix[t + i * I2PR_THREADS] = sum;
      sum2 = fma(sum, sum, sum2);
    }
    if (a.norm == 0u) continue;
    sum2 = simd_sum(sum2);
    if (lane == 0u) part[0][sg] = sum2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < I2PR_THREADS / 32u; ++s) total += part[0][s];
    const float r = 1.0f / sqrt(total / float(a.width) + a.eps);
    const device bfloat* g = k == 0u ? W0 : k == 1u ? W1 : W2;
    device float* o = (k == 0u ? O0 : k == 1u ? O1 : O2) + (size_t)row * a.width;
    for (uint i = 0; i < I2PR_PER; ++i) o[t + i * I2PR_THREADS] = out[i] * r * float(g[t + i * I2PR_THREADS]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

struct i2pr_clip_args { uint heads, kv, items; float q_threshold, rel_threshold, eps; };

inline float i2pr_part(float v) {
  return simd_sum(fma(v, v, 0.0f));
}

[[kernel]] void i2pr_clips(device float* Q [[buffer(0)]], const device bfloat* QS [[buffer(1)]],
    device float* R [[buffer(2)]], const device bfloat* RS [[buffer(3)]], const device float* KC [[buffer(4)]],
    const device float* VC [[buffer(5)]], device half* KH [[buffer(6)]], device half* VH [[buffer(7)]],
    constant i2pr_clip_args& a [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const uint item = tg.x * 8u + sg, row = tg.y;
  if (item >= a.items) return;
  if (item < 2u * a.heads) {
    const bool rel = item >= a.heads;
    const uint h = rel ? item - a.heads : item;
    if (rel) {
      device float* x = R + ((size_t)row * a.heads + h) * 16u;
      const device bfloat* sc = RS + (size_t)h * 16u;
      const float v = lane < 16u ? x[lane] * float(sc[lane]) : 0.0f;
      float total = 0.0f;
      total += i2pr_part(v);
      const float clip = clamp(a.rel_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
      if (lane < 16u) x[lane] = x[lane] * float(sc[lane]) * clip;
      return;
    }
    device float* x = Q + ((size_t)row * a.heads + h) * 128u;
    const device bfloat* sc = QS + (size_t)h * 128u;
    float total = 0.0f;
    for (uint s = 0; s < 4u; ++s) total += i2pr_part(x[lane + 32u * s] * float(sc[lane + 32u * s]));
    const float clip = clamp(a.q_threshold / (sqrt(total) + 1e-6f), 0.0f, 1.0f);
    for (uint s = 0; s < 4u; ++s) x[lane + 32u * s] = x[lane + 32u * s] * float(sc[lane + 32u * s]) * clip;
    return;
  }
  const uint h = item - 2u * a.heads;
  const size_t at = ((size_t)row * a.kv + h) * 128u;
  float kt = 0.0f;
  float vt = 0.0f;
  for (uint s = 0; s < 4u; ++s) {
    kt += i2pr_part(KC[at + lane + 32u * s]);
    vt += i2pr_part(VC[at + lane + 32u * s]);
  }
  const float clip = clamp(a.q_threshold / (sqrt(kt) + 1e-6f), 0.0f, 1.0f);
  const float inv = 1.0f / sqrt(vt / 128.0f + a.eps);
  for (uint s = 0; s < 4u; ++s) {
    KH[at + lane + 32u * s] = KC[at + lane + 32u * s] * clip;
    VH[at + lane + 32u * s] = VC[at + lane + 32u * s] * inv;
  }
}

struct i2pr_ring_args { uint channels, ring, base, rows, tail; };

[[kernel]] void i2pr_ring_store(const device float* X [[buffer(0)]], device float* RING [[buffer(1)]],
    constant i2pr_ring_args& a [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels || p.y >= a.tail) return;
  const uint y = a.rows - a.tail + p.y;
  RING[((a.base + y) % a.ring) * a.channels + p.x] = X[(size_t)y * a.channels + p.x];
}

struct i2pr_kv_args { uint count, base, rows, wring; };

[[kernel]] void i2pr_kv_store(const device half4* K [[buffer(0)]], const device half4* V [[buffer(1)]],
    device half4* KEYS [[buffer(2)]], device half4* VALUES [[buffer(3)]], constant i2pr_kv_args& a [[buffer(4)]],
    uint2 p [[thread_position_in_grid]]) {
  const uint quads = a.count / 4u;
  if (p.x >= quads || p.y >= a.rows) return;
  if (a.wring && p.y + a.wring < a.rows) return;
  const uint r = a.wring ? (a.base + p.y) % a.wring : a.base + p.y;
  KEYS[(size_t)r * quads + p.x] = K[(size_t)p.y * quads + p.x];
  VALUES[(size_t)r * quads + p.x] = V[(size_t)p.y * quads + p.x];
}

struct i2pr_conv_args { uint channels, heads, ring, base, rows, layout; float norm; uint out; };

[[kernel]] void i2pr_conv_add(const device float* X [[buffer(0)]], const device float* RING [[buffer(1)]],
    const device bfloat* W [[buffer(2)]], constant i2pr_conv_args& a [[buffer(3)]], device float* O [[buffer(4)]],
    device float* RES [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
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
    sum = fma(v, float(W[tap]), sum);
  }
  const float value = sum * a.norm;
  const size_t at = (size_t)t * a.channels + c;
  if (a.out) O[at] = value;
  RES[at] = RES[at] + value;
}

inline void i2pr_mix_row(const device float* BLK, const device float* CUR, const device float* Q0,
    const device float* Q1, const device float* Q2, const device float* Q3, constant i2pr_mix_args& a, device float* MIX,
    const device bfloat* W0, const device bfloat* W1, const device bfloat* W2, device float* O0, device float* O1,
    device float* O2, uint row, uint t, uint sg, uint lane, threadgroup float (&part)[4][5][8],
    threadgroup float (&w)[4][5], threadgroup float (&inv)[5]) {
  float src[5][9];
  for (uint j = 0; j < a.depth; ++j) {
    const device float* s = j + 1u < a.depth ? BLK + ((size_t)row * a.stride + (a.depth - 2u - j)) * a.width
                                             : CUR + (size_t)row * a.width;
    float sum = 0.0f;
    for (uint i = 0; i < 9u; ++i) {
      src[j][i] = s[t + i * 256u];
      sum = fma(src[j][i], src[j][i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) part[0][j][sg] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.depth) {
    float total = 0.0f;
    for (uint s = 0; s < 8u; ++s) total += part[0][t][s];
    inv[t] = 1.0f / sqrt(total / float(a.width) + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < a.depth; ++j) for (uint i = 0; i < 9u; ++i) src[j][i] = src[j][i] * inv[j];
  for (uint k = 0; k < a.k; ++k) {
    const device float* q = k == 0u ? Q0 : k == 1u ? Q1 : k == 2u ? Q2 : Q3;
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = 0; i < 9u; ++i) sum = fma(src[j][i], q[t + i * 256u], sum);
      sum = simd_sum(sum);
      if (lane == 0u) part[k][j][sg] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < a.k) {
    for (uint j = 0; j < a.depth; ++j) {
      float total = 0.0f;
      for (uint s = 0; s < 8u; ++s) total += part[t][j][s];
      w[t][j] = total * a.scale;
    }
    float peak = w[t][0];
    for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[t][j]);
    float total = 0.0f;
    for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[t][j] - peak);
    for (uint j = 0; j < a.depth; ++j) w[t][j] = precise::exp(w[t][j] - peak) / total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float out[3][9];
  for (uint k = 0; k < a.k; ++k) {
    device float* mix = MIX + ((size_t)row * a.k + k) * a.width;
    float sum2 = 0.0f;
    for (uint i = 0; i < 9u; ++i) {
      float sum = 0.0f;
      for (uint j = 0; j < a.depth; ++j) sum = fma(w[k][j], src[j][i], sum);
      mix[t + i * 256u] = sum;
      if (k < 3u) out[k][i] = sum;
      sum2 = fma(sum, sum, sum2);
    }
    if (a.norm == 0u) continue;
    sum2 = simd_sum(sum2);
    if (lane == 0u) part[0][k][sg] = sum2;
  }
  if (a.norm == 0u) return;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < a.k; ++k) {
    float total = 0.0f;
    for (uint s = 0; s < 8u; ++s) total += part[0][k][s];
    const float r = 1.0f / sqrt(total / float(a.width) + a.eps);
    const device bfloat* g = k == 0u ? W0 : k == 1u ? W1 : W2;
    device float* o = (k == 0u ? O0 : k == 1u ? O1 : O2) + (size_t)row * a.width;
    for (uint i = 0; i < 9u; ++i) o[t + i * 256u] = out[k][i] * r * float(g[t + i * 256u]);
  }
}

[[kernel]] void i2pr_batch_mix(const device float* BLK [[buffer(0)]], const device float* CUR [[buffer(1)]],
    const device float* Q0 [[buffer(2)]], const device float* Q1 [[buffer(3)]], const device float* Q2 [[buffer(4)]],
    const device float* Q3 [[buffer(5)]], constant i2pr_mix_args& a [[buffer(6)]], device float* MIX [[buffer(7)]],
    const device bfloat* W0 [[buffer(8)]], const device bfloat* W1 [[buffer(9)]], const device bfloat* W2 [[buffer(10)]],
    device float* O0 [[buffer(11)]], device float* O1 [[buffer(12)]], device float* O2 [[buffer(13)]],
    uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  threadgroup float inv[5];
  i2pr_mix_row(BLK, CUR, Q0, Q1, Q2, Q3, a, MIX, W0, W1, W2, O0, O1, O2, row, t, sg, lane, part, w, inv);
}

[[kernel]] void i2pr_batch_mix2(const device float* BLK0 [[buffer(0)]], const device float* CUR0 [[buffer(1)]],
    const device float* Q00 [[buffer(2)]], const device float* Q01 [[buffer(3)]], const device float* Q02 [[buffer(4)]],
    const device float* Q03 [[buffer(5)]], device float* MIX0 [[buffer(6)]], const device float* BLK1 [[buffer(7)]],
    const device float* CUR1 [[buffer(8)]], const device float* Q10 [[buffer(9)]], const device float* Q11 [[buffer(10)]],
    const device float* Q12 [[buffer(11)]], const device float* Q13 [[buffer(12)]], device float* MIX1 [[buffer(13)]],
    constant i2pr_mix_args* args [[buffer(14)]], uint2 tg [[threadgroup_position_in_grid]],
    uint t [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[4][5][8];
  threadgroup float w[4][5];
  threadgroup float inv[5];
  const device bfloat* none = (const device bfloat*)Q00;
  if (tg.y == 0u) i2pr_mix_row(BLK0, CUR0, Q00, Q01, Q02, Q03, args[0], MIX0, none, none, none, MIX0, MIX0, MIX0, tg.x, t, sg, lane, part, w, inv);
  else i2pr_mix_row(BLK1, CUR1, Q10, Q11, Q12, Q13, args[1], MIX1, none, none, none, MIX1, MIX1, MIX1, tg.x, t, sg, lane, part, w, inv);
}
