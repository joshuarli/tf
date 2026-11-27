#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

typedef vec<half, 8> i2n_h8;
typedef vec<float, 8> i2n_f8;

inline short2 i2n_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

inline void i2n_mma(thread i2n_f8& lo, thread i2n_f8& hi, thread const i2n_h8& a, thread const i2n_h8& b0,
    thread const i2n_h8& b1) {
  using namespace mpp::tensor_ops;
  constexpr auto shape = matmul2d_descriptor(16, 32, 16, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<shape, execution_simdgroup> op;
  auto left = op.template get_left_input_cooperative_tensor<half, half, float>();
  auto right = op.template get_right_input_cooperative_tensor<half, half, float>();
  auto acc = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>,
      metal::remove_addrspace_t<decltype(right)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    left[e] = a[e];
    right[e] = b0[e];
    right[8 + e] = b1[e];
    acc[e] = lo[e];
    acc[8 + e] = hi[e];
  }
  op.run(left, right, acc);
  _Pragma("clang loop unroll(full)")
  for (short e = 0; e < 8; e++) {
    lo[e] = acc[e];
    hi[e] = acc[8 + e];
  }
}

inline half i2n_fp8_half(uchar bits) {
  const ushort v = bits & 127u;
  const ushort sign = ushort(bits & 128u) << 8u;
  const ushort u = (v << 7u) | (((v + 1u) >> 7u) << 14u) | sign;
  return as_type<half>(u) * 256.0h;
}

inline half4 i2n_fp4x4(uint w) {
  const ushort4 c = ushort4(w & 15u, (w >> 4u) & 15u, (w >> 8u) & 15u, (w >> 12u) & 15u);
  const half4 mag = as_type<half4>(ushort4((c & 7u) << 9u)) * 16384.0h;
  return as_type<half4>(as_type<ushort4>(mag) | ushort4((c & 8u) << 12u));
}

inline float i2n_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) / (1.0f + precise::exp(-g));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

template <uint S>
inline void i2n_reduce(thread i2n_f8& acc0, thread i2n_f8& acc1, threadgroup float* red, ushort sg, ushort lane) {
  if (S > 1u) {
    if (sg > 0u) {
      const uint base = (uint(sg) - 1u) * 16u;
      _Pragma("clang loop unroll(full)")
      for (uint e = 0u; e < 8u; ++e) {
        red[(base + e) * 32u + lane] = acc0[e];
        red[(base + 8u + e) * 32u + lane] = acc1[e];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
      for (uint o = 0u; o < S - 1u; ++o) {
        const uint base = o * 16u;
        _Pragma("clang loop unroll(full)")
        for (uint e = 0u; e < 8u; ++e) {
          acc0[e] += red[(base + e) * 32u + lane];
          acc1[e] += red[(base + 8u + e) * 32u + lane];
        }
      }
    }
  }
}

struct i2n_sort_args { uint slots, experts, grid_x_up, grid_x_down; };

[[kernel]] void i2n_moe_sort(const device int* PICKS [[buffer(0)]], constant i2n_sort_args& a [[buffer(1)]],
    device uint* META [[buffer(2)]], device uint* INDIRECT [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint cnt[128];
  threadgroup atomic_uint cur[128];
  threadgroup uint sg_sum[4];
  threadgroup uint sg_act[4];
  atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tid; s < a.slots; s += 128u) {
    atomic_fetch_add_explicit(&cnt[uint(PICKS[s])], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint c = tid < a.experts ? atomic_load_explicit(&cnt[tid], memory_order_relaxed) : 0u;
  const uint act = c > 0u ? 1u : 0u;
  const uint c_local = simd_prefix_exclusive_sum(c);
  const uint a_local = simd_prefix_exclusive_sum(act);
  if (lane == 31u) {
    sg_sum[sg] = c_local + c;
    sg_act[sg] = a_local + act;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint c_base = 0u, a_base = 0u;
  for (uint s = 0u; s < sg; ++s) {
    c_base += sg_sum[s];
    a_base += sg_act[s];
  }
  const uint off = c_base + c_local;
  const uint idx = a_base + a_local;
  if (act != 0u) {
    META[idx] = tid;
    META[128u + idx] = off;
    META[256u + idx] = c;
    atomic_store_explicit(&cur[tid], off, memory_order_relaxed);
  }
  if (tid == 127u) {
    const uint total_act = idx + act;
    INDIRECT[0] = a.grid_x_up;
    INDIRECT[1] = total_act;
    INDIRECT[2] = 1u;
    INDIRECT[3] = a.grid_x_down;
    INDIRECT[4] = total_act;
    INDIRECT[5] = 1u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint s = tid; s < a.slots; s += 128u) {
    const uint e = uint(PICKS[s]);
    const uint p = atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
    META[384u + p] = s;
  }
}

struct i2n_moe_args { uint k, n, slots, topk; float gate_cap, up_cap; };

template <bool GU, uint R>
inline void i2n_nvfp4_few_lut(const device float* X, const device uchar* W, const device uchar* S_buf,
    const device float* G, const device float* den, const device uint* META, constant i2n_moe_args& a,
    device float* Y, threadgroup float* red, uint expert, uint off, uint n0, ushort sg, ushort lane) {
  const uint tid = uint(sg) * 32u + uint(lane);
  threadgroup float* lut = red;
  threadgroup float* scales = red + 16u;
  if (tid < 16u) {
    const uint m = tid & 7u;
    lut[tid] = as_type<float>((m < 2u ? m * 0x3f000000u : 0x3f000000u + (m << 22u)) | ((tid & 8u) << 28u));
  }
  for (uint i = tid; i < 256u; i += 128u) scales[i] = float(i2n_fp8_half(uchar(i)));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint cols = GU ? 2u * a.n : a.n;
  const uint stride = a.k >> 1u;
  const uint groups = a.k >> 4u;
  const float global = G[expert] / den[0];
  uint slots[R];
  const device float* xs[R];
  _Pragma("clang loop unroll(full)")
  for (uint r = 0u; r < R; ++r) {
    slots[r] = META[384u + off + r];
    xs[r] = X + size_t(GU ? slots[r] / a.topk : slots[r]) * a.k;
  }
  const device uchar* We = W + size_t(expert) * cols * stride;
  const device uchar* Se = S_buf + size_t(expert) * cols * groups;
  if (GU) {
    const uint pair_base = (n0 >> 1u) + uint(sg) * 4u;
    for (uint pass = 0u; pass < 2u; ++pass) {
      const uint p0 = pair_base + pass * 2u;
      if (p0 >= a.n) break;
      const uint rows[2] = { 2u * p0, min(2u * (p0 + 1u), cols - 2u) };
      float dg[2][R] = {}, du[2][R] = {};
      for (uint g = lane; g < groups; g += 32u) {
        float4 xv[R][4];
        _Pragma("clang loop unroll(full)")
        for (uint r = 0u; r < R; ++r) {
          _Pragma("clang loop unroll(full)")
          for (uint q = 0u; q < 4u; ++q) xv[r][q] = *reinterpret_cast<const device float4*>(xs[r] + g * 16u + 4u * q);
        }
        _Pragma("clang loop unroll(full)")
        for (uint p = 0u; p < 2u; ++p) {
          const ulong wg = *reinterpret_cast<const device ulong*>(We + size_t(rows[p]) * stride + g * 8u);
          const ulong wu = *reinterpret_cast<const device ulong*>(We + size_t(rows[p] + 1u) * stride + g * 8u);
          float pg[R] = {}, pu[R] = {};
          _Pragma("clang loop unroll(full)")
          for (uint q = 0u; q < 4u; ++q) {
            const uint bg = uint(wg >> (16u * q)) & 0xffffu;
            const uint bu = uint(wu >> (16u * q)) & 0xffffu;
            const float g0 = lut[bg & 15u], g1 = lut[(bg >> 4u) & 15u], g2 = lut[(bg >> 8u) & 15u], g3 = lut[bg >> 12u];
            const float u0 = lut[bu & 15u], u1 = lut[(bu >> 4u) & 15u], u2 = lut[(bu >> 8u) & 15u], u3 = lut[bu >> 12u];
            _Pragma("clang loop unroll(full)")
            for (uint r = 0u; r < R; ++r) {
              pg[r] = fma(g3, xv[r][q].w, fma(g2, xv[r][q].z, fma(g1, xv[r][q].y, fma(g0, xv[r][q].x, pg[r]))));
              pu[r] = fma(u3, xv[r][q].w, fma(u2, xv[r][q].z, fma(u1, xv[r][q].y, fma(u0, xv[r][q].x, pu[r]))));
            }
          }
          const float sg_v = scales[Se[size_t(rows[p]) * groups + g]];
          const float su_v = scales[Se[size_t(rows[p] + 1u) * groups + g]];
          _Pragma("clang loop unroll(full)")
          for (uint r = 0u; r < R; ++r) {
            dg[p][r] = fma(sg_v, pg[r], dg[p][r]);
            du[p][r] = fma(su_v, pu[r], du[p][r]);
          }
        }
      }
      _Pragma("clang loop unroll(full)")
      for (uint p = 0u; p < 2u; ++p) {
        if (p0 + p >= a.n) break;
        _Pragma("clang loop unroll(full)")
        for (uint r = 0u; r < R; ++r) {
          const float gs = simd_sum(dg[p][r]), us = simd_sum(du[p][r]);
          if (lane == 0u) Y[size_t(slots[r]) * a.n + p0 + p] = i2n_situ(gs * global, us * global, a.gate_cap, a.up_cap);
        }
      }
    }
    return;
  }
  const uint row_base = n0 + uint(sg) * 8u;
  for (uint pass = 0u; pass < 2u; ++pass) {
    const uint r0 = row_base + pass * 4u;
    if (r0 >= a.n) break;
    float d[4][R] = {};
    for (uint g = lane; g < groups; g += 32u) {
      float4 xv[R][4];
      _Pragma("clang loop unroll(full)")
      for (uint r = 0u; r < R; ++r) {
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) xv[r][q] = *reinterpret_cast<const device float4*>(xs[r] + g * 16u + 4u * q);
      }
      _Pragma("clang loop unroll(full)")
      for (uint p = 0u; p < 4u; ++p) {
        const uint wr = min(r0 + p, a.n - 1u);
        const ulong w = *reinterpret_cast<const device ulong*>(We + size_t(wr) * stride + g * 8u);
        float pv[R] = {};
        _Pragma("clang loop unroll(full)")
        for (uint q = 0u; q < 4u; ++q) {
          const uint b = uint(w >> (16u * q)) & 0xffffu;
          const float c0 = lut[b & 15u], c1 = lut[(b >> 4u) & 15u], c2 = lut[(b >> 8u) & 15u], c3 = lut[b >> 12u];
          _Pragma("clang loop unroll(full)")
          for (uint r = 0u; r < R; ++r) {
            pv[r] = fma(c3, xv[r][q].w, fma(c2, xv[r][q].z, fma(c1, xv[r][q].y, fma(c0, xv[r][q].x, pv[r]))));
          }
        }
        const float sc = scales[Se[size_t(wr) * groups + g]];
        _Pragma("clang loop unroll(full)")
        for (uint r = 0u; r < R; ++r) d[p][r] = fma(sc, pv[r], d[p][r]);
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint p = 0u; p < 4u; ++p) {
      if (r0 + p >= a.n) break;
      _Pragma("clang loop unroll(full)")
      for (uint r = 0u; r < R; ++r) {
        const float v = simd_sum(d[p][r]);
        if (lane == 0u) Y[size_t(slots[r]) * a.n + r0 + p] = v * global;
      }
    }
  }
}

template <bool GU, bool AFFINE, uint S>
inline void i2n_expert_body(const device float* X, const device uchar* W, const device uchar* S_buf,
    const device uchar* G, const device float* den, const device uint* META, constant i2n_moe_args& a,
    device float* Y, threadgroup float* red, uint2 tg, ushort sg, ushort lane) {
  const uint expert = META[tg.y];
  const uint off = META[128u + tg.y];
  const uint cnt = META[256u + tg.y];
  const uint cols = GU ? 2u * a.n : a.n;
  const uint n0 = tg.x * 32u;
  if (!AFFINE && S == 4u) {
    if (cnt == 1u) { i2n_nvfp4_few_lut<GU, 1u>(X, W, S_buf, (const device float*)G, den, META, a, Y, red, expert, off, n0, sg, lane); return; }
    if (cnt == 2u) { i2n_nvfp4_few_lut<GU, 2u>(X, W, S_buf, (const device float*)G, den, META, a, Y, red, expert, off, n0, sg, lane); return; }
  }
  const uint groups = a.k >> 6u;
  const short2 home = i2n_home(lane);
  const uint fm = uint(home.y);
  const uint q4 = uint(home.x) >> 2u;
  const bool xlive0 = fm < cnt;
  const bool xlive1 = (fm + 8u) < cnt;
  const device float* xr0 = xlive0 ? X + size_t(GU ? META[384u + off + fm] / a.topk : META[384u + off + fm]) * a.k + 16u * q4 : nullptr;
  const device float* xr1 = xlive1 ? X + size_t(GU ? META[384u + off + fm + 8u] / a.topk : META[384u + off + fm + 8u]) * a.k + 16u * q4 : nullptr;
  const device uchar* wp[4];
  const device uchar* sp[4];
  const device bfloat* bp[4];
  bool wvalid[4];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0u; j < 4u; ++j) {
    const uint row = n0 + fm + 8u * j;
    wvalid[j] = row < cols;
    const size_t wrow = size_t(expert) * cols + (wvalid[j] ? row : 0u);
    wp[j] = W + wrow * (a.k >> 1u) + 8u * q4;
    sp[j] = S_buf + (AFFINE ? 2u * wrow * groups : wrow * (a.k >> 4u) + q4);
    bp[j] = (const device bfloat*)G + wrow * groups;
  }
  i2n_f8 acc0 = i2n_f8(0.0f), acc1 = i2n_f8(0.0f);
  for (uint g = sg; g < groups; g += S) {
    uint2 wv[4];
    half sh[4];
    float sf[4], bf[4];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 4u; ++j) {
      wv[j] = wvalid[j] ? *reinterpret_cast<const device uint2*>(wp[j] + 32u * g) : uint2(0u);
      if (AFFINE) {
        sf[j] = wvalid[j] ? float(((const device bfloat*)sp[j])[g]) : 0.0f;
        bf[j] = wvalid[j] ? float(bp[j][g]) : 0.0f;
      } else {
        sh[j] = wvalid[j] ? i2n_fp8_half(sp[j][4u * g]) : 0.0h;
      }
    }
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
      const float4 x0f = xlive0 ? *reinterpret_cast<const device float4*>(xr0 + 64u * g + 4u * q) : float4(0.0f);
      const float4 x1f = xlive1 ? *reinterpret_cast<const device float4*>(xr1 + 64u * g + 4u * q) : float4(0.0f);
      const half4 x0h = AFFINE ? half4(x0f) : half4(clamp(x0f, -65504.0f, 65504.0f));
      const half4 x1h = AFFINE ? half4(x1f) : half4(clamp(x1f, -65504.0f, 65504.0f));
      half4 bw[4];
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 4u; ++j) {
        const uint chunk = ((q < 2u ? wv[j].x : wv[j].y) >> (16u * (q & 1u))) & 0xffffu;
        if (AFFINE) {
          bw[j] = wvalid[j] ? half4(fma(float4((uint4(chunk) >> uint4(0u, 4u, 8u, 12u)) & 15u), sf[j], bf[j])) : half4(0.0h);
        } else {
          bw[j] = wvalid[j] ? i2n_fp4x4(chunk) * sh[j] : half4(0.0h);
        }
      }
      i2n_mma(acc0, acc1, i2n_h8(x0h, x1h), i2n_h8(bw[0], bw[1]), i2n_h8(bw[2], bw[3]));
    }
  }
  i2n_reduce<S>(acc0, acc1, red, sg, lane);
  if (sg != 0u) return;
  const float global = AFFINE ? 1.0f : ((const device float*)G)[expert] / den[0];
  const i2n_f8 vs[2] = { acc0, acc1 };
  _Pragma("clang loop unroll(full)")
  for (uint h = 0u; h < 2u; ++h) {
    const uint r = fm + 8u * h;
    if (r >= cnt) continue;
    const size_t slot = size_t(META[384u + off + r]);
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 2u; ++j) {
      const uint col = n0 + 16u * j + uint(home.x);
      if (GU) {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0u; e < 2u; ++e) {
          if (col / 2u + e < a.n) {
            Y[slot * a.n + col / 2u + e] = i2n_situ(vs[j][4u * h + 2u * e] * global,
                vs[j][4u * h + 2u * e + 1u] * global, a.gate_cap, a.up_cap);
          }
        }
      } else {
        _Pragma("clang loop unroll(full)")
        for (uint e = 0u; e < 4u; ++e) {
          if (col + e < a.n) Y[slot * a.n + col + e] = vs[j][4u * h + e] * global;
        }
      }
    }
  }
}

#define I2N_EXPERT(NAME, GU, AFFINE, S) \
[[kernel]] void NAME(const device float* X [[buffer(0)]], const device uchar* W [[buffer(1)]], \
    const device uchar* S_buf [[buffer(2)]], const device uchar* G [[buffer(3)]], const device float* den [[buffer(4)]], \
    const device uint* META [[buffer(5)]], constant i2n_moe_args& a [[buffer(6)]], device float* Y [[buffer(7)]], \
    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
  threadgroup float red[(S > 1u ? S - 1u : 1u) * 16u * 32u]; \
  i2n_expert_body<GU, AFFINE, S>(X, W, S_buf, G, den, META, a, Y, red, tg, sg, lane); \
}

I2N_EXPERT(i2n_nvfp4_up, true, false, 4)
I2N_EXPERT(i2n_nvfp4_down, false, false, 4)
I2N_EXPERT(i2n_aff4_up, true, true, 4)
I2N_EXPERT(i2n_aff4_down, false, true, 4)

struct i2n_dense_args { uint n, k, rows, xs, xb, bits; float cap; };

template <uint S>
inline void i2n_affine8_body(const device float* x, size_t step, uint rows, const device uint* w,
    const device bfloat* s, const device bfloat* b, uint row0, uint last, uint k,
    threadgroup float* red, ushort sg, ushort lane, thread i2n_f8& acc0, thread i2n_f8& acc1) {
  const uint words = k >> 2u, groups = k >> 6u;
  const short2 home = i2n_home(lane);
  const uint fm = uint(home.y);
  const uint q4 = uint(home.x) >> 2u;
  const bool xlive0 = fm < rows;
  const bool xlive1 = (fm + 8u) < rows;
  const device float* xr0 = xlive0 ? x + size_t(fm) * step + 16u * q4 : nullptr;
  const device float* xr1 = xlive1 ? x + size_t(fm + 8u) * step + 16u * q4 : nullptr;
  const device uint* wp[4];
  const device bfloat* sp[4];
  const device bfloat* bp[4];
  bool wvalid[4];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0u; j < 4u; ++j) {
    const uint wr = row0 + fm + 8u * j;
    wvalid[j] = wr <= last;
    const size_t wrow = wvalid[j] ? wr : last;
    wp[j] = w + wrow * words + 4u * q4;
    sp[j] = s + wrow * groups;
    bp[j] = b + wrow * groups;
  }
  acc0 = i2n_f8(0.0f);
  acc1 = i2n_f8(0.0f);
  const uint4 shift8 = uint4(0u, 8u, 16u, 24u);
  for (uint g = sg; g < groups; g += S) {
    uint4 wv[4];
    float sc[4], bi[4];
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 4u; ++j) {
      wv[j] = wvalid[j] ? *reinterpret_cast<const device uint4*>(wp[j] + 16u * g) : uint4(0u);
      sc[j] = wvalid[j] ? float(sp[j][g]) : 0.0f;
      bi[j] = wvalid[j] ? float(bp[j][g]) : 0.0f;
    }
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
      const half4 x0h = xlive0 ? half4(*reinterpret_cast<const device float4*>(xr0 + 64u * g + 4u * q)) : half4(0.0h);
      const half4 x1h = xlive1 ? half4(*reinterpret_cast<const device float4*>(xr1 + 64u * g + 4u * q)) : half4(0.0h);
      half4 bw[4];
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 4u; ++j) {
        const uint cw = wv[j][q];
        bw[j] = wvalid[j] ? half4(fma(float4((uint4(cw) >> shift8) & 255u), sc[j], bi[j])) : half4(0.0h);
      }
      i2n_mma(acc0, acc1, i2n_h8(x0h, x1h), i2n_h8(bw[0], bw[1]), i2n_h8(bw[2], bw[3]));
    }
  }
  i2n_reduce<S>(acc0, acc1, red, sg, lane);
}

template <uint S>
[[kernel]] void i2n_dense_t(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2n_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[(S > 1u ? S - 1u : 1u) * 16u * 32u];
  const uint row0 = group * 32u;
  i2n_f8 acc0, acc1;
  i2n_affine8_body<S>(x + size_t(a.xb) * a.k, size_t(a.xs) * a.k, a.rows, w, s, b, row0, a.n - 1u, a.k, red, sg, lane, acc0, acc1);
  if (sg != 0u) return;
  const short2 home = i2n_home(lane);
  const i2n_f8 vs[2] = { acc0, acc1 };
  _Pragma("clang loop unroll(full)")
  for (uint h = 0u; h < 2u; ++h) {
    const uint r = uint(home.y) + 8u * h;
    if (r >= a.rows) continue;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 2u; ++j) {
      const uint col = row0 + 16u * j + uint(home.x);
      _Pragma("clang loop unroll(full)")
      for (uint e = 0u; e < 4u; ++e) {
        if (col + e < a.n) {
          float v = vs[j][4u * h + e];
          if (a.cap > 0.0f) v = a.cap * precise::tanh(v / a.cap);
          y[size_t(r) * a.n + col + e] = v;
        }
      }
    }
  }
}

template [[host_name("i2n_dense4")]] [[kernel]] void i2n_dense_t<4>(const device float* x [[buffer(0)]],
    const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2n_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]);

template [[host_name("i2n_dense8")]] [[kernel]] void i2n_dense_t<8>(const device float* x [[buffer(0)]],
    const device uint* w [[buffer(1)]], const device bfloat* s [[buffer(2)]], const device bfloat* b [[buffer(3)]],
    constant i2n_dense_args& a [[buffer(4)]], device float* y [[buffer(5)]], uint group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]);

struct i2n_projection_args { uint k, full, rows; uint m[8]; };

[[kernel]] void i2n_projections(const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]],
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]], const device bfloat* sq [[buffer(4)]],
    const device bfloat* bq [[buffer(5)]], const device uint* wk [[buffer(6)]], const device bfloat* sk [[buffer(7)]],
    const device bfloat* bk [[buffer(8)]], const device uint* wv [[buffer(9)]], const device bfloat* sv [[buffer(10)]],
    const device bfloat* bv [[buffer(11)]], device float* oq [[buffer(12)]], device float* ok [[buffer(13)]],
    device float* ov [[buffer(14)]], device float* orl [[buffer(15)]], device float* og [[buffer(16)]],
    device float* os [[buffer(17)]], device float* okd [[buffer(18)]], device float* ovd [[buffer(19)]],
    constant i2n_projection_args& a [[buffer(20)]], uint group [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint S = 4u;
  threadgroup float red[(S - 1u) * 16u * 32u];
  uint which = 0u, first = group;
  for (; which < 8u; ++which) {
    const uint span = (a.m[which] + 31u) / 32u;
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
  const uint m = a.m[which], row0 = first * 32u;
  i2n_f8 acc0, acc1;
  i2n_affine8_body<S>(key ? xk : val ? xv : xq, a.k, a.rows,
      a.full && key ? wk : a.full && val ? wv : wq,
      a.full && key ? sk : a.full && val ? sv : sq,
      a.full && key ? bk : a.full && val ? bv : bq,
      offset + row0, offset + m - 1u, a.k, red, sg, lane, acc0, acc1);
  if (sg != 0u) return;
  device float* y = which == 0u ? oq : which == 1u ? ok : which == 2u ? ov : which == 3u ? orl
      : which == 4u ? og : which == 5u ? os : which == 6u ? okd : ovd;
  const short2 home = i2n_home(lane);
  const i2n_f8 vs[2] = { acc0, acc1 };
  _Pragma("clang loop unroll(full)")
  for (uint h = 0u; h < 2u; ++h) {
    const uint r = uint(home.y) + 8u * h;
    if (r >= a.rows) continue;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 2u; ++j) {
      const uint col = row0 + 16u * j + uint(home.x);
      _Pragma("clang loop unroll(full)")
      for (uint e = 0u; e < 4u; ++e) {
        if (col + e < m) y[size_t(r) * m + col + e] = vs[j][4u * h + e];
      }
    }
  }
}

inline half4 i2n_bf16x4(const device bfloat* p) {
  const uint2 u = *reinterpret_cast<const device uint2*>(p);
  return half4(float4(as_type<float>(u.x << 16u), as_type<float>(u.x & 0xffff0000u),
                      as_type<float>(u.y << 16u), as_type<float>(u.y & 0xffff0000u)));
}

struct i2n_controls_args { uint h, rank, rows; uint xs[3], xb[3]; uint msg0, cols; };

[[kernel]] void i2n_controls_down_bf16(const device float* X [[buffer(0)]], const device bfloat* W0 [[buffer(1)]],
    const device bfloat* W1 [[buffer(2)]], const device bfloat* W2 [[buffer(3)]],
    constant i2n_controls_args& a [[buffer(4)]], device float* OUT [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  constexpr uint S = 8u;
  threadgroup float red[(S - 1u) * 16u * 32u];
  const uint j_ctl = tg.y;
  const uint row0 = tg.x * 32u;
  const uint k = 2u * a.h;
  const uint groups = k >> 6u;
  const short2 home = i2n_home(lane);
  const uint fm = uint(home.y);
  const uint q4 = uint(home.x) >> 2u;
  const bool xlive0 = fm < a.rows;
  const bool xlive1 = (fm + 8u) < a.rows;
  const device float* xbase = X + size_t(j_ctl) * a.rows * k;
  const device float* xr0 = xlive0 ? xbase + size_t(fm) * k + 16u * q4 : nullptr;
  const device float* xr1 = xlive1 ? xbase + size_t(fm + 8u) * k + 16u * q4 : nullptr;
  const device bfloat* W = j_ctl == 0u ? W0 : j_ctl == 1u ? W1 : W2;
  const device bfloat* wp[4];
  bool wvalid[4];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0u; j < 4u; ++j) {
    const uint wr = row0 + fm + 8u * j;
    wvalid[j] = wr < a.rank;
    wp[j] = W + size_t(wvalid[j] ? wr : 0u) * k + 16u * q4;
  }
  i2n_f8 acc0 = i2n_f8(0.0f), acc1 = i2n_f8(0.0f);
  for (uint g = sg; g < groups; g += S) {
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
      const half4 x0h = xlive0 ? half4(*reinterpret_cast<const device float4*>(xr0 + 64u * g + 4u * q)) : half4(0.0h);
      const half4 x1h = xlive1 ? half4(*reinterpret_cast<const device float4*>(xr1 + 64u * g + 4u * q)) : half4(0.0h);
      half4 bw[4];
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 4u; ++j) {
        bw[j] = wvalid[j] ? i2n_bf16x4(wp[j] + 64u * g + 4u * q) : half4(0.0h);
      }
      i2n_mma(acc0, acc1, i2n_h8(x0h, x1h), i2n_h8(bw[0], bw[1]), i2n_h8(bw[2], bw[3]));
    }
  }
  i2n_reduce<S>(acc0, acc1, red, sg, lane);
  if (sg != 0u) return;
  const i2n_f8 vs[2] = { acc0, acc1 };
  _Pragma("clang loop unroll(full)")
  for (uint h = 0u; h < 2u; ++h) {
    const uint r = fm + 8u * h;
    if (r >= a.rows) continue;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 2u; ++j) {
      const uint col = row0 + 16u * j + uint(home.x);
      _Pragma("clang loop unroll(full)")
      for (uint e = 0u; e < 4u; ++e) {
        if (col + e < a.rank) OUT[(size_t(j_ctl) * a.rows + r) * a.rank + col + e] = vs[j][4u * h + e];
      }
    }
  }
}

[[kernel]] void i2n_controls_apply_bf16(const device float* X [[buffer(0)]], const device bfloat* G0 [[buffer(1)]],
    const device bfloat* G1 [[buffer(2)]], const device bfloat* G2 [[buffer(3)]], const device bfloat* F0 [[buffer(4)]],
    const device bfloat* F1 [[buffer(5)]], const device bfloat* F2 [[buffer(6)]], const device float* C0 [[buffer(7)]],
    const device float* C1 [[buffer(8)]], const device float* C2 [[buffer(9)]], const device float* MSG [[buffer(10)]],
    device float* O0 [[buffer(11)]], device float* O1 [[buffer(12)]], device float* O2 [[buffer(13)]],
    constant i2n_controls_args& a [[buffer(14)]], uint2 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const uint j_ctl = tg.y;
  const uint row0 = tg.x * 32u;
  const uint groups = a.rank >> 6u;
  const short2 home = i2n_home(lane);
  const uint fm = uint(home.y);
  const uint q4 = uint(home.x) >> 2u;
  const bool xlive0 = fm < a.rows;
  const bool xlive1 = (fm + 8u) < a.rows;
  const device float* xbase = X + size_t(j_ctl) * a.rows * a.rank;
  const device float* xr0 = xlive0 ? xbase + size_t(fm) * a.rank + 16u * q4 : nullptr;
  const device float* xr1 = xlive1 ? xbase + size_t(fm + 8u) * a.rank + 16u * q4 : nullptr;
  const device bfloat* WG = j_ctl == 0u ? G0 : j_ctl == 1u ? G1 : G2;
  const device bfloat* WF = j_ctl == 0u ? F0 : j_ctl == 1u ? F1 : F2;
  const device bfloat* gp[4];
  const device bfloat* fp[4];
  bool wvalid[4];
  _Pragma("clang loop unroll(full)")
  for (uint j = 0u; j < 4u; ++j) {
    const uint wr = row0 + fm + 8u * j;
    wvalid[j] = wr < a.h;
    const size_t woff = size_t(wvalid[j] ? wr : 0u) * a.rank + 16u * q4;
    gp[j] = WG + woff;
    fp[j] = WF + woff;
  }
  i2n_f8 ag0 = i2n_f8(0.0f), ag1 = i2n_f8(0.0f);
  i2n_f8 af0 = i2n_f8(0.0f), af1 = i2n_f8(0.0f);
  for (uint g = 0u; g < groups; ++g) {
    _Pragma("clang loop unroll(full)")
    for (uint q = 0u; q < 4u; ++q) {
      const half4 x0h = xlive0 ? half4(*reinterpret_cast<const device float4*>(xr0 + 64u * g + 4u * q)) : half4(0.0h);
      const half4 x1h = xlive1 ? half4(*reinterpret_cast<const device float4*>(xr1 + 64u * g + 4u * q)) : half4(0.0h);
      const i2n_h8 xv = i2n_h8(x0h, x1h);
      half4 bg[4], bf[4];
      _Pragma("clang loop unroll(full)")
      for (uint j = 0u; j < 4u; ++j) {
        bg[j] = wvalid[j] ? i2n_bf16x4(gp[j] + 64u * g + 4u * q) : half4(0.0h);
        bf[j] = wvalid[j] ? i2n_bf16x4(fp[j] + 64u * g + 4u * q) : half4(0.0h);
      }
      i2n_mma(ag0, ag1, xv, i2n_h8(bg[0], bg[1]), i2n_h8(bg[2], bg[3]));
      i2n_mma(af0, af1, xv, i2n_h8(bf[0], bf[1]), i2n_h8(bf[2], bf[3]));
    }
  }
  const device float* C = j_ctl == 0u ? C0 : j_ctl == 1u ? C1 : C2;
  device float* O = j_ctl == 0u ? O0 : j_ctl == 1u ? O1 : O2;
  const i2n_f8 vg[2] = { ag0, ag1 };
  const i2n_f8 vf[2] = { af0, af1 };
  _Pragma("clang loop unroll(full)")
  for (uint h = 0u; h < 2u; ++h) {
    const uint r = fm + 8u * h;
    if (r >= a.rows) continue;
    const device float* cur = C + size_t(a.xb[j_ctl] + r * a.xs[j_ctl]) * a.h;
    const device float* msg = MSG + size_t(a.msg0 + j_ctl + r * 4u) * a.h;
    device float* out = O + size_t(r) * a.h;
    _Pragma("clang loop unroll(full)")
    for (uint j = 0u; j < 2u; ++j) {
      const uint col = row0 + 16u * j + uint(home.x);
      _Pragma("clang loop unroll(full)")
      for (uint e = 0u; e < 4u; ++e) {
        if (col + e < a.h) {
          const float c_val = cur[col + e];
          const float m_val = msg[col + e];
          out[col + e] = c_val - precise::tanh(vf[j][4u * h + e]) * c_val + precise::tanh(vg[j][4u * h + e]) * m_val;
        }
      }
    }
  }
}
