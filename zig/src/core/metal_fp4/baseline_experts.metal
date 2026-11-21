// The routed expert projections: gate/up over the checkpoint's interleaved stored rows, then down.
// One expert a threadgroup row, 32 output columns a threadgroup, 2 rows by 2 columns a thread; every member of the
// expert (up to 32 at a time) is folded through the same staged weight tile. NVFP4 is the default decoder.
#include <metal_stdlib>
using namespace metal;

struct i2_moe { uint k, n, rows; float gate_cap, up_cap; };

// An e4m3 code's value. 0x7f/0xff are NaN, as the reference's table reads them.
inline float i2_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

// An e2m1 nibble's value.
inline float i2_e2m1(uchar c) {
  const float t[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
  const float v = t[c & 7u];
  return (c & 8u) ? -v : v;
}

// Expert `e`'s stored row `row` at input `k`, decoded as the checkpoint stores it.
inline float i2_weight(const device uchar* W, const device uchar* S, const device float* amax,
                       const device float* den, uint e, uint row, uint k, uint K, uint rows) {
  const size_t base = (size_t)e * rows + row;
#ifdef I2_MXFP8
  return i2_e4m3(W[base * K + k]) * exp2(float(int(S[base * (K / 32u) + k / 32u]) - 127));
#else
  const uchar b = W[base * (K / 2u) + k / 2u];
  const float code = (k & 1u) ? i2_e2m1(b >> 4) : i2_e2m1(b & 15u);
  return code * i2_e4m3(S[base * (K / 16u) + k / 16u]) * (amax[e] / den[0]);
#endif
}

// act[slot, col] = situ(gate, up) for each member of expert tg.y: one token, so every member shares the input
// row X, and ORDER names the pick slot its activation lands in.
[[kernel]] void i2_gu(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ORDER [[buffer(6)]], constant i2_moe& a [[buffer(7)]], device float* ACT [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint e = tg.y, col0 = tg.x * 32u;
  const uint from = uint(OFF[e]), to = uint(OFF[e + 1u]);
  const uint tx = tid.x % 16u, ty = tid.x / 16u;
  threadgroup float xs[32][9];
  threadgroup float wg[32][9];
  threadgroup float wu[32][9];
  for (uint base = from; base < to; base += 32u) {
    const uint cnt = min(32u, to - base);
    float ag[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    float au[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    for (uint k0 = 0; k0 < a.k; k0 += 8u) {
      for (uint i = tid.x; i < 256u; i += 256u) {
        const uint r = i / 8u, kk = i % 8u, gk = k0 + kk;
        xs[r][kk] = (r < cnt && gk < a.k) ? X[gk] : 0.0f;
        const uint col = col0 + r;
        const bool live = col < a.n && gk < a.k;
        wg[r][kk] = live ? i2_weight(W, S, amax, den, e, 2u * col, gk, a.k, a.rows) : 0.0f;
        wu[r][kk] = live ? i2_weight(W, S, amax, den, e, 2u * col + 1u, gk, a.k, a.rows) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < 8u; ++kk) {
        const float x0 = xs[ty * 2u][kk], x1 = xs[ty * 2u + 1u][kk];
        for (uint c = 0; c < 2u; ++c) {
          const uint lc = tx * 2u + c;
          ag[0][c] = fma(x0, wg[lc][kk], ag[0][c]);
          au[0][c] = fma(x0, wu[lc][kk], au[0][c]);
          ag[1][c] = fma(x1, wg[lc][kk], ag[1][c]);
          au[1][c] = fma(x1, wu[lc][kk], au[1][c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 2u; ++r) for (uint c = 0; c < 2u; ++c) {
      const uint member = ty * 2u + r;
      const uint col = col0 + tx * 2u + c;
      if (member >= cnt || col >= a.n) continue;
      const uint slot = uint(ORDER[base + member]);
      const float g = ag[r][c], u = au[r][c];
      const float gate = a.gate_cap * precise::tanh(g / a.gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
      const float up = a.up_cap > 0.0f ? a.up_cap * precise::tanh(u / a.up_cap) : u;
      ACT[(size_t)slot * a.n + col] = gate * up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// out[slot, col] = act[slot, :] . W2[e, col, :] for each member of expert tg.y.
[[kernel]] void i2_down(const device float* ACT [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* amax [[buffer(3)]],
    const device float* den [[buffer(4)]], const device int* OFF [[buffer(5)]],
    const device int* ORDER [[buffer(6)]], constant i2_moe& a [[buffer(7)]], device float* OUT [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]) {
  const uint e = tg.y, col0 = tg.x * 32u;
  const uint from = uint(OFF[e]), to = uint(OFF[e + 1u]);
  const uint tx = tid.x % 16u, ty = tid.x / 16u;
  threadgroup float xs[32][9];
  threadgroup float ws[32][9];
  for (uint base = from; base < to; base += 32u) {
    const uint cnt = min(32u, to - base);
    float acc[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
    for (uint k0 = 0; k0 < a.k; k0 += 8u) {
      for (uint i = tid.x; i < 256u; i += 256u) {
        const uint r = i / 8u, kk = i % 8u, gk = k0 + kk;
        xs[r][kk] = (r < cnt && gk < a.k) ? ACT[(size_t)uint(ORDER[base + r]) * a.k + gk] : 0.0f;
        const uint col = col0 + r;
        ws[r][kk] = (col < a.n && gk < a.k) ? i2_weight(W, S, amax, den, e, col, gk, a.k, a.rows) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint kk = 0; kk < 8u; ++kk) {
        const float x0 = xs[ty * 2u][kk], x1 = xs[ty * 2u + 1u][kk];
        for (uint c = 0; c < 2u; ++c) {
          const float w0 = ws[tx * 2u + c][kk];
          acc[0][c] = fma(x0, w0, acc[0][c]);
          acc[1][c] = fma(x1, w0, acc[1][c]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 2u; ++r) for (uint c = 0; c < 2u; ++c) {
      const uint member = ty * 2u + r;
      const uint col = col0 + tx * 2u + c;
      if (member >= cnt || col >= a.n) continue;
      OUT[(size_t)uint(ORDER[base + member]) * a.n + col] = acc[r][c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
