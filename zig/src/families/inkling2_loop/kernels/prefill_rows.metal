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
  threadgroup float part[I2PR_THREADS / 32u];
  threadgroup float w[4][I2PR_DEPTH];
  float src[I2PR_DEPTH][I2PR_PER];
  float inv[I2PR_DEPTH];
  for (uint j = 0; j < a.depth; ++j) {
    const device float* s = j + 1u < a.depth ? BLK + ((size_t)row * a.stride + (a.depth - 2u - j)) * a.width
                                             : CUR + (size_t)row * a.width;
    float sum = 0.0f;
    for (uint i = 0; i < I2PR_PER; ++i) {
      const float v = s[t + i * I2PR_THREADS];
      src[j][i] = v;
      sum = fma(v, v, sum);
    }
    const float total = i2pr_total(sum, part, t);
    inv[j] = 1.0f / sqrt(total / float(a.width) + 1e-6f);
  }
  for (uint k = 0; k < a.k; ++k) {
    const device float* q = k == 0u ? Q0 : k == 1u ? Q1 : k == 2u ? Q2 : Q3;
    for (uint j = 0; j < a.depth; ++j) {
      float sum = 0.0f;
      for (uint i = 0; i < I2PR_PER; ++i) sum = fma(src[j][i] * inv[j], q[t + i * I2PR_THREADS], sum);
      const float total = i2pr_total(sum, part, t);
      if (t == 0u) w[k][j] = total * a.scale;
    }
    if (t == 0u) {
      float peak = w[k][0];
      for (uint j = 1; j < a.depth; ++j) peak = max(peak, w[k][j]);
      float total = 0.0f;
      for (uint j = 0; j < a.depth; ++j) total += precise::exp(w[k][j] - peak);
      for (uint j = 0; j < a.depth; ++j) w[k][j] = precise::exp(w[k][j] - peak) / total;
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
    const float total = i2pr_total(sum2, part, t);
    const float r = 1.0f / sqrt(total / float(a.width) + a.eps);
    const device bfloat* g = k == 0u ? W0 : k == 1u ? W1 : W2;
    device float* o = (k == 0u ? O0 : k == 1u ? O1 : O2) + (size_t)row * a.width;
    for (uint i = 0; i < I2PR_PER; ++i) o[t + i * I2PR_THREADS] = out[i] * r * float(g[t + i * I2PR_THREADS]);
  }
}
