#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct i2_aot_as_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2_aot_moeg_args { uint k, n, slots, topk, experts; float gate_cap, up_cap; };
struct i2_aot_t_dense_args { uint n, k, rows, xs, xb, bits; float cap; };
struct i2_aot_t_proj_args { uint k, full, rows; uint m[8]; };
struct i2_aot_t_moe_args { uint k, n, slots, topk; float gate_cap, up_cap; };
struct i2_aot_ctl_r_args { uint h, rank, rows; uint xs[3], xb[3]; uint msg0, cols; };
struct i2_aot_ctl1_r_args { uint k, m0, m1, h, cxs, cxb, mxs, mxb, ys, yb; };

typedef vec<half, 8> i2a_h8;
typedef vec<float, 8> i2a_f8;

inline float i2a_situ(float g, float u, float gate_cap, float up_cap) {
  const float gate = gate_cap * precise::tanh(g / gate_cap) * (1.0f / (1.0f + precise::exp(-g)));
  const float up = up_cap > 0.0f ? up_cap * precise::tanh(u / up_cap) : u;
  return gate * up;
}

inline float i2a_e4m3(uchar c) {
  if ((c & 127u) == 127u) return NAN;
  const uint e = (uint(c) >> 3) & 15u, man = uint(c) & 7u;
  const float v = e ? (1.0f + float(man) * 0.125f) * exp2(float(int(e) - 7)) : float(man) * (1.0f / 512.0f);
  return (c & 128u) ? -v : v;
}

inline half4 i2a_fp4x4(uint w) {
  const uint4 c = (uint4(w) >> uint4(0u, 4u, 8u, 12u)) & 15u;
  const uint4 mag = c & 7u;
  const uint4 bits = select(0x3800u + (mag << 9), mag * 0x3800u, mag < 2u) | ((c & 8u) << 12);
  return as_type<half4>(ushort4(bits));
}

inline short2 i2a_home(ushort l) {
  return short2(short((l & 8) + ((l & 1) << 2)), short(((l & 16) >> 2) | ((l >> 1) & 3)));
}

inline void i2a_mma(thread i2a_f8& lo, thread i2a_f8& hi, thread const i2a_h8& a, thread const i2a_h8& b0,
    thread const i2a_h8& b1) {
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

inline bool i2a_expert_tile(const device int* OFF, int experts, int least, int tile, ushort lane,
    thread int& expert, thread int& row, thread int& rows) {
  int before = 0;
  for (int base = 0; base < experts; base += 32) {
    const int e = base + int(lane);
    const int first = e < experts ? OFF[e] : 0, last = e < experts ? OFF[e + 1] : 0;
    const int count = last - first >= least ? (last - first + 63) / 64 : 0;
    const int upto = before + simd_prefix_inclusive_sum(count);
    const int owner = simd_sum(int(upto <= tile));
    if (owner < 32) {
      expert = base + owner;
      row = simd_shuffle(first, ushort(owner)) + (tile - simd_shuffle(upto - count, ushort(owner))) * 64;
      rows = min(64, simd_shuffle(last, ushort(owner)) - row);
      return true;
    }
    before = simd_shuffle(upto, ushort(31));
  }
  return false;
}

[[kernel]] void i2_aot_as_prepare_sb(device const uint* w [[buffer(0)]], device const bfloat2* sb [[buffer(1)]],
    constant i2_aot_as_args& a [[buffer(2)]], device half* out_w [[buffer(3)]],
    device atomic_uint* overflow [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i < a.n * a.k / 4u) {
    const uint row = i / (a.k / 4u), k = (i % (a.k / 4u)) * 4u, per = 32u / a.bits;
    const uint packed = w[(size_t)row * (a.k / per) + k / per];
    const uint shift = (k % per) * a.bits, mask = (1u << a.bits) - 1u;
    const size_t group = (size_t)row * (a.k / 64u) + k / 64u;
    const bfloat2 pair = sb[group];
    const uint4 offsets = uint4(0u, a.bits, 2u * a.bits, 3u * a.bits);
    const float4 code = float4((uint4(packed >> shift) >> offsets) & mask);
    const float4 value = fma(code, float4(float(pair.x)), float4(float(pair.y)));
    if (any(abs(value) > 65504.0f)) atomic_fetch_add_explicit(overflow, 1u, memory_order_relaxed);
    *reinterpret_cast<device half4*>(out_w + (size_t)row * a.k + k) = half4(value);
  }
}

[[kernel]] void i2_aot_unpack_gf2(const device float2* gf [[buffer(0)]], constant uint& count [[buffer(1)]],
    device float* out_g [[buffer(2)]], device float* out_f [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < count) {
    const float2 v = gf[i];
    out_g[i] = v.x;
    out_f[i] = v.y;
  }
}

[[kernel]] void i2_aot_expert_units_up_packed(const device half* X [[buffer(0)]], const device uchar* W [[buffer(1)]],
    const device uchar* S [[buffer(2)]], const device float* G [[buffer(3)]], const device float* den [[buffer(4)]],
    const device int* OFF [[buffer(5)]], const device int* ROW_OF [[buffer(6)]],
    constant i2_aot_moeg_args& a [[buffer(7)]], device half* ACT [[buffer(8)]],
    device atomic_uint* overflow [[buffer(9)]], uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  constexpr int BK = 64, PAD = 72;
  threadgroup half w[64 * PAD];
  int expert = 0, row0 = 0, rows = 0;
  if (!i2a_expert_tile(OFF, int(a.experts), 1, int(tg.y), lane, expert, row0, rows)) return;
  const int cols = 2 * int(a.n), K = int(a.k), n0 = int(tg.x) * 64;
  const int t = int(sg) * 32 + int(lane);
  const int tm = 32 * (int(sg) / 2), tn = 32 * (int(sg) % 2);
  const short2 home = i2a_home(lane);
  const device half* xr[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    _Pragma("clang loop unroll(full)")
    for (short h = 0; h < 2; h++) {
      const int i = tm + 16 * m + home.y + 8 * h;
      const uint slot = uint(ROW_OF[row0 + (i < rows ? i : 0)]);
      xr[m][h] = X + size_t(slot / a.topk) * size_t(K) + home.x;
    }
  }
  const int wr = t / 2, wk = (t % 2) * 32;
  const int col = n0 + wr;
  const bool wvalid = col < cols;
  const int c_unit = col >> 1;
  const int is_up = col & 1;
  const size_t groups = size_t(K / 16);
  const size_t base = (size_t(expert) * size_t(a.n) + size_t(c_unit)) * groups;
  threadgroup half* mine = w + wr * PAD + wk;
  i2a_f8 acc[2][2];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    acc[m][0] = i2a_f8(0.0f);
    acc[m][1] = i2a_f8(0.0f);
  }
  uint over = 0u;
  for (int k0 = 0; k0 < K; k0 += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t g0 = size_t((k0 + wk) / 16);
    const ulong w0 = wvalid ? *(const device ulong*)(W + (base + g0) * 16u + size_t(is_up) * 8u) : 0ul;
    const ulong w1 = wvalid ? *(const device ulong*)(W + (base + g0 + 1u) * 16u + size_t(is_up) * 8u) : 0ul;
    const half s0 = wvalid ? half(i2a_e4m3(S[(base + g0) * 2u + size_t(is_up)])) : 0.0h;
    const half s1 = wvalid ? half(i2a_e4m3(S[(base + g0 + 1u) * 2u + size_t(is_up)])) : 0.0h;
    const uint c0 = uint(w0), c1 = uint(w0 >> 32u), c2 = uint(w1), c3 = uint(w1 >> 32u);
    *(threadgroup half4*)(mine + 0) = i2a_fp4x4(c0) * s0;
    *(threadgroup half4*)(mine + 4) = i2a_fp4x4(c0 >> 16u) * s0;
    *(threadgroup half4*)(mine + 8) = i2a_fp4x4(c1) * s0;
    *(threadgroup half4*)(mine + 12) = i2a_fp4x4(c1 >> 16u) * s0;
    *(threadgroup half4*)(mine + 16) = i2a_fp4x4(c2) * s1;
    *(threadgroup half4*)(mine + 20) = i2a_fp4x4(c2 >> 16u) * s1;
    *(threadgroup half4*)(mine + 24) = i2a_fp4x4(c3) * s1;
    *(threadgroup half4*)(mine + 28) = i2a_fp4x4(c3 >> 16u) * s1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    _Pragma("clang loop unroll(full)")
    for (short q = 0; q < 4; q++) {
      const threadgroup half* b = w + (tn + home.y) * PAD + 16 * q + home.x;
      const i2a_h8 b0 = i2a_h8(*(const threadgroup half4*)b, *(const threadgroup half4*)(b + 8 * PAD));
      const i2a_h8 b1 = i2a_h8(*(const threadgroup half4*)(b + 16 * PAD), *(const threadgroup half4*)(b + 24 * PAD));
      _Pragma("clang loop unroll(full)")
      for (short m = 0; m < 2; m++) {
        const i2a_h8 x = i2a_h8(*(const device half4*)(xr[m][0] + k0 + 16 * q), *(const device half4*)(xr[m][1] + k0 + 16 * q));
        i2a_mma(acc[m][0], acc[m][1], x, b0, b1);
      }
    }
  }
  const float global = G[expert] / den[0];
  _Pragma("clang loop unroll(full)")
  for (short m = 0; m < 2; m++) {
    _Pragma("clang loop unroll(full)")
    for (short h = 0; h < 2; h++) {
      const int r = tm + 16 * m + home.y + 8 * h;
      if (r >= rows) continue;
      const size_t slot = size_t(ROW_OF[row0 + r]);
      _Pragma("clang loop unroll(full)")
      for (short j = 0; j < 2; j++) {
        const int ocol = n0 + tn + 16 * j + home.x;
        const i2a_f8 v = acc[m][j];
        _Pragma("clang loop unroll(full)")
        for (short e = 0; e < 2; e++) {
          if (ocol / 2 + e >= int(a.n)) continue;
          const float value = i2a_situ(v[4 * h + 2 * e] * global, v[4 * h + 2 * e + 1] * global, a.gate_cap, a.up_cap);
          over += uint(abs(value) > 65504.0f);
          ACT[slot * a.n + ocol / 2 + e] = half(clamp(value, -65504.0f, 65504.0f));
        }
      }
    }
  }
  if (over) atomic_fetch_add_explicit(overflow, over, memory_order_relaxed);
}

struct i2_aot_kv_perm_args { uint rows, stride_rows, kv, dim; };

[[kernel]] void i2_aot_kv_to_perm(const device half* src [[buffer(0)]], device half* dst [[buffer(1)]],
    constant i2_aot_kv_perm_args& a [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
  if (p.x >= a.dim || p.y >= a.kv || p.z >= a.rows) return;
  const size_t s = ((size_t)p.z * a.kv + p.y) * a.dim + p.x;
  const size_t d = ((size_t)p.y * a.stride_rows + p.z) * a.dim + (p.x & 31u) * 4u + (p.x >> 5u);
  dst[d] = src[s];
}

inline void i2a_lane(uint lane, thread uint& fm, thread uint& fn) {
  const uint qid = lane >> 2;
  fm = (qid & 4u) | ((lane >> 1) & 3u);
  fn = ((qid & 2u) << 1) | ((lane & 1u) << 1);
}

inline thread float2& i2a_elements(thread simdgroup_matrix<float, 8, 8>& m) {
  return reinterpret_cast<thread float2&>(m.thread_elements());
}

inline void i2a_mma_f32(thread float2& c, float2 a, float2 b) {
  simdgroup_matrix<float, 8, 8> A, B, C, D;
  i2a_elements(A) = a;
  i2a_elements(B) = b;
  i2a_elements(C) = c;
  simdgroup_multiply_accumulate(D, A, B, C);
  c = i2a_elements(D);
}

template <uint N>
inline float2 i2a_tree(thread float2 (&v)[N]) {
#pragma unroll
  for (uint width = 1; width < N; width <<= 1)
#pragma unroll
    for (uint i = 0; i < N; i += 2u * width) v[i] = v[i] + v[i + width];
  return v[0];
}

template <uint SG, uint CL, uint N>
inline void i2a_partials(thread float2 (&acc)[CL][N], threadgroup float2* part, uint sg, uint lane) {
#pragma unroll
  for (uint n = 0; n < N; ++n) {
    float2 v[CL];
#pragma unroll
    for (uint l = 0; l < CL; ++l) v[l] = acc[l][n];
    part[(sg * N + n) * 32u + lane] = i2a_tree(v);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <uint SG, uint N>
inline float2 i2a_total(threadgroup const float2* part, uint n, uint lane) {
  float2 v[SG];
#pragma unroll
  for (uint i = 0; i < SG; ++i) v[i] = part[(i * N + n) * 32u + lane];
  return i2a_tree(v);
}

template <uint SG, uint T, uint G>
inline void i2a_affine_sb(const device float* x, size_t step, uint rows, const device uint* w,
    const device bfloat2* sb, uint row0, uint last, uint k, uint sg, uint lane, threadgroup float2* part,
    thread float2 (&out)[G][T]) {
  constexpr uint CL = 32u / SG;
  uint fm, fn;
  i2a_lane(lane, fm, fn);
  const uint words = k / 4u, groups = k / 64u, chunks = words / 64u;
  const uint half_word = 32u * (fn >> 2), shift = 8u * (fn & 3u), half_k = 128u * (fm >> 2) + (fm & 3u);
  bool live[2u * G];
  const device float* xr[2u * G];
#pragma unroll
  for (uint i = 0; i < 2u * G; ++i) {
    live[i] = 8u * (i / 2u) + fn + i % 2u < rows;
    xr[i] = x + (live[i] ? 8u * (i / 2u) + fn + i % 2u : 0u) * step;
  }
  size_t row[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) row[t] = min(row0 + t * 8u + fm, last);
  float2 acc[CL][G * T] = {};
  for (uint c = 0; c < chunks; ++c) {
    const uint lead = sg * CL + 64u * c;
    const uint gi = (lead + half_word) / 16u;
    float scale[T], bias[T];
#pragma unroll
    for (uint t = 0; t < T; ++t) {
      const bfloat2 pair = sb[row[t] * groups + gi];
      scale[t] = float(pair.x);
      bias[t] = float(pair.y);
    }
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint kk = 4u * (lead + l) + half_k;
      float2 bx[G];
#pragma unroll
      for (uint g = 0; g < G; ++g) bx[g] = float2(live[2u * g] ? xr[2u * g][kk] : 0.0f, live[2u * g + 1u] ? xr[2u * g + 1u][kk] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const uint packed = w[row[t] * words + lead + l + half_word];
        const float2 code = float2(float((packed >> shift) & 255u), float((packed >> (shift + 8u)) & 255u));
        const float2 weight = fma(code, float2(scale[t]), float2(bias[t]));
#pragma unroll
        for (uint g = 0; g < G; ++g) i2a_mma_f32(acc[l][g * T + t], weight, bx[g]);
      }
    }
  }
  i2a_partials<SG, CL, G * T>(acc, part, sg, lane);
#pragma unroll
  for (uint g = 0; g < G; ++g)
#pragma unroll
    for (uint t = 0; t < T; ++t) out[g][t] = i2a_total<SG, G * T>(part, g * T + t, lane);
}

template <uint SG, uint T, uint G>
[[kernel]] void i2_aot_t_dense_sb(const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]],
    const device bfloat2* sb [[buffer(2)]], constant i2_aot_t_dense_args& a [[buffer(3)]],
    device float* y [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float2 part[SG * G * T * 32];
  const uint row0 = group * 8u * T;
  float2 out[G][T];
  i2a_affine_sb<SG, T, G>(x + (size_t)a.xb * a.k, (size_t)a.xs * a.k, a.rows, w, sb, row0, a.n - 1u, a.k, sg, lane, part, out);
  if (sg != 0u) return;
  uint fm, fn;
  i2a_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint col = row0 + t * 8u + fm;
    if (col >= a.n) break;
#pragma unroll
    for (uint g = 0; g < G; ++g) {
      float2 v = out[g][t];
      if (a.cap > 0.0f) v = float2(a.cap * precise::tanh(v.x / a.cap), a.cap * precise::tanh(v.y / a.cap));
      if (8u * g + fn < a.rows) y[(size_t)(8u * g + fn) * a.n + col] = v.x;
      if (8u * g + fn + 1u < a.rows) y[(size_t)(8u * g + fn + 1u) * a.n + col] = v.y;
    }
  }
}

template [[host_name("i2_aot_t_dense_8_2_sb")]] [[kernel]] void i2_aot_t_dense_sb<8, 2, 1>(
    const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat2* sb [[buffer(2)]],
    constant i2_aot_t_dense_args& a [[buffer(3)]], device float* y [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);

template [[host_name("i2_aot_t_dense16_8_1_sb")]] [[kernel]] void i2_aot_t_dense_sb<8, 1, 2>(
    const device float* x [[buffer(0)]], const device uint* w [[buffer(1)]], const device bfloat2* sb [[buffer(2)]],
    constant i2_aot_t_dense_args& a [[buffer(3)]], device float* y [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);

#define I2A_PROJ_ARGS const device float* xq [[buffer(0)]], const device float* xk [[buffer(1)]], \
    const device float* xv [[buffer(2)]], const device uint* wq [[buffer(3)]], const device bfloat2* sbq [[buffer(4)]], \
    const device uint* wk [[buffer(5)]], const device bfloat2* sbk [[buffer(6)]], \
    const device uint* wv [[buffer(7)]], const device bfloat2* sbv [[buffer(8)]], \
    device float* oq [[buffer(9)]], device float* ok [[buffer(10)]], device float* ov [[buffer(11)]], \
    device float* orl [[buffer(12)]], device float* og [[buffer(13)]], device float* os [[buffer(14)]], \
    device float* okd [[buffer(15)]], device float* ovd [[buffer(16)]], \
    constant i2_aot_t_proj_args& a [[buffer(17)]], uint group [[threadgroup_position_in_grid]], \
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]

template <uint SG, uint T, uint G>
[[kernel]] void i2_aot_t_proj_sb(I2A_PROJ_ARGS) {
  threadgroup float2 part[SG * G * T * 32];
  uint which = 0u, first = group;
  for (; which < 8u; ++which) {
    const uint span = (a.m[which] + 8u * T - 1u) / (8u * T);
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
  const uint m = a.m[which], row0 = first * 8u * T;
  float2 out[G][T];
  i2a_affine_sb<SG, T, G>(key ? xk : val ? xv : xq, a.k, a.rows, a.full && key ? wk : a.full && val ? wv : wq,
      a.full && key ? sbk : a.full && val ? sbv : sbq, offset + row0, offset + m - 1u, a.k, sg, lane, part, out);
  if (sg != 0u) return;
  device float* y = which == 0u ? oq : which == 1u ? ok : which == 2u ? ov : which == 3u ? orl
      : which == 4u ? og : which == 5u ? os : which == 6u ? okd : ovd;
  uint fm, fn;
  i2a_lane(lane, fm, fn);
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint row = row0 + t * 8u + fm;
    if (row >= m) break;
#pragma unroll
    for (uint g = 0; g < G; ++g) {
      if (8u * g + fn < a.rows) y[(size_t)(8u * g + fn) * m + row] = out[g][t].x;
      if (8u * g + fn + 1u < a.rows) y[(size_t)(8u * g + fn + 1u) * m + row] = out[g][t].y;
    }
  }
}

template [[host_name("i2_aot_t_proj_8_2_sb")]] [[kernel]] void i2_aot_t_proj_sb<8, 2, 1>(I2A_PROJ_ARGS);
template [[host_name("i2_aot_t_proj16_8_1_sb")]] [[kernel]] void i2_aot_t_proj_sb<8, 1, 2>(I2A_PROJ_ARGS);

template <uint Q>
inline uint i2a_e_run(const device int* PICKS, uint slots, uint slot, uint lane, thread uint (&member)[8]) {
  const int e = PICKS[slot];
  uint mask[Q], rank = 0u;
#pragma unroll
  for (uint q = 0; q < Q; ++q) {
    const uint at = lane + 32u * q, base = 32u * q;
    mask[q] = uint(ulong(simd_vote::vote_t(simd_ballot(at < slots && PICKS[at] == e))));
    const uint below = slot >= base + 32u ? 0xffffffffu : (slot <= base ? 0u : (1u << (slot - base)) - 1u);
    rank += popcount(mask[q] & below);
    mask[q] &= ~below;
  }
  if (rank % 8u != 0u) return 0u;
  uint count = 0u;
#pragma unroll
  for (uint i = 0; i < 8u; ++i) {
    uint s = 32u * Q;
#pragma unroll
    for (uint q = 0; q < Q; ++q) {
      if (s == 32u * Q && mask[q] != 0u) {
        s = 32u * q + ctz(mask[q]);
        mask[q] &= mask[q] - 1u;
      }
    }
    member[i] = s;
    count += s < 32u * Q ? 1u : 0u;
  }
  return count;
}

template <uint R>
inline void i2a_e_few_gu(const device float* X, const device ulong2* We, const device uchar2* Se,
    const thread uint (&member)[8], uint r0, uint sg, threadgroup const float* lut, threadgroup const float* scales,
    uint lane, float global, constant i2_aot_t_moe_args& a, device float* Y) {
  constexpr uint PER = 2u;
  const device float* x[R];
  uint slot[R];
#pragma unroll
  for (uint r = 0; r < R; ++r) {
    slot[r] = member[r];
    x[r] = X + (size_t)(member[r] / a.topk) * a.k;
  }
  const uint first = r0 / 2u + sg * PER;
  const uint groups = a.k / 16u;
  for (uint j = 0; j < PER; ++j) {
    const uint unit = first + j;
    if (unit >= a.n) break;
    float dg[R] = {}, du[R] = {};
    const device ulong2* wgu = We + (size_t)unit * groups;
    const device uchar2* sgu = Se + (size_t)unit * groups;
    for (uint g = lane; g < groups; g += 32u) {
      const uint k0 = g * 16u;
      const ulong2 agu = wgu[g];
      const uchar2 scu = sgu[g];
      float pg[R] = {}, pu[R] = {};
      for (uint bi = 0; bi < 8u; ++bi) {
        const uint ab = uint(agu.x >> (8u * bi)) & 255u, ub = uint(agu.y >> (8u * bi)) & 255u;
        const float cg0 = lut[ab & 15u], cg1 = lut[ab >> 4u], cu0 = lut[ub & 15u], cu1 = lut[ub >> 4u];
#pragma unroll
        for (uint r = 0; r < R; ++r) {
          const float x0 = x[r][k0 + 2u * bi], x1 = x[r][k0 + 2u * bi + 1u];
          pg[r] = fma(cg1, x1, fma(cg0, x0, pg[r]));
          pu[r] = fma(cu1, x1, fma(cu0, x0, pu[r]));
        }
      }
      const float sgv = scales[scu.x], suv = scales[scu.y];
#pragma unroll
      for (uint r = 0; r < R; ++r) dg[r] = fma(sgv, pg[r], dg[r]), du[r] = fma(suv, pu[r], du[r]);
    }
#pragma unroll
    for (uint r = 0; r < R; ++r) {
      const float gsum = simd_sum(dg[r]), usum = simd_sum(du[r]);
      if (lane == 0u) Y[(size_t)slot[r] * a.n + unit] = i2a_situ(gsum * global, usum * global, a.gate_cap, a.up_cap);
    }
  }
}

template <uint Q>
[[kernel]] void i2_aot_e_gu_packed(const device float* X [[buffer(0)]], const device ulong2* W [[buffer(1)]],
    const device uchar2* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_t_moe_args& a [[buffer(5)]],
    device float* Y [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint SG = 4u, T = 2u, CL = 32u / SG;
  threadgroup float lut[16];
  threadgroup float scales[256];
  threadgroup float2 part[SG * T * 32];
  const uint slot = tg.y;
  if (slot >= a.slots) return;
  uint member[8];
  const uint count = i2a_e_run<Q>(PICKS, a.slots, slot, lane, member);
  if (count == 0u) return;
  if (tid < 16u) lut[tid] = E2M1[tid];
  for (uint i = tid; i < 256u; i += 32u * SG) scales[i] = E4M3[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int e = PICKS[slot];
  const uint groups = a.k / 16u;
  const uint wrows = 2u * a.n;
  const uint r0 = tg.x * 8u * T;
  const float global = gscale[e];
  const device ulong2* We = W + (size_t)uint(e) * a.n * groups;
  const device uchar2* Se = S + (size_t)uint(e) * a.n * groups;
  if (count == 1u) return i2a_e_few_gu<1>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  if (count == 2u) return i2a_e_few_gu<2>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  if (count == 3u) return i2a_e_few_gu<3>(X, We, Se, member, r0, sg, lut, scales, lane, global, a, Y);
  uint fm, fn;
  i2a_lane(lane, fm, fn);
  const bool live0 = fn < count, live1 = fn + 1u < count;
  const device float* x0 = X + (size_t)(live0 ? member[fn] / a.topk : 0u) * a.k;
  const device float* x1 = X + (size_t)(live1 ? member[fn + 1u] / a.topk : 0u) * a.k;
  float2 acc[CL][T] = {};
  uint unit[T], is_up[T];
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint rw = min(r0 + t * 8u + fm, wrows - 1u);
    unit[t] = rw >> 1u;
    is_up[t] = rw & 1u;
  }
  const uint nb = fn >> 1;
  for (uint base = sg * CL; base < groups; base += 32u) {
#pragma unroll
    for (uint l = 0; l < CL; ++l) {
      const uint gi = base + l;
      if (gi >= groups) break;
      const uint k0 = gi * 16u + fm;
      const float2 b0 = float2(live0 ? x0[k0] : 0.0f, live1 ? x1[k0] : 0.0f);
      const float2 b1 = float2(live0 ? x0[k0 + 8u] : 0.0f, live1 ? x1[k0 + 8u] : 0.0f);
#pragma unroll
      for (uint t = 0; t < T; ++t) {
        const size_t idx = (size_t)unit[t] * groups + gi;
        const ulong2 agu = We[idx];
        const uchar2 scu = Se[idx];
        const ulong word = is_up[t] ? agu.y : agu.x;
        const uchar scb = is_up[t] ? scu.y : scu.x;
        const uint c0 = uint(word >> (8u * nb)) & 255u, c1 = uint(word >> (8u * (4u + nb))) & 255u;
        const float2 a0 = float2(lut[c0 & 15u], lut[c0 >> 4u]), a1 = float2(lut[c1 & 15u], lut[c1 >> 4u]);
        float2 p = float2(0.0f);
        i2a_mma_f32(p, a0, b0);
        i2a_mma_f32(p, a1, b1);
        acc[l][t] = fma(float2(scales[scb]), p, acc[l][t]);
      }
    }
  }
  i2a_partials<SG, CL, T>(acc, part, sg, lane);
  if (sg != 0u) return;
#pragma unroll
  for (uint t = 0; t < T; ++t) {
    const uint r = r0 + t * 8u + fm;
    const float2 sum = i2a_total<SG, T>(part, t, lane);
    const size_t y0 = (size_t)member[fn] * a.n, y1 = (size_t)member[fn + 1u] * a.n;
    const float2 up = float2(simd_shuffle_xor(sum.x, 2u), simd_shuffle_xor(sum.y, 2u));
    if ((fm & 1u) == 0u && r < wrows) {
      if (live0) Y[y0 + r / 2u] = i2a_situ(sum.x * global, up.x * global, a.gate_cap, a.up_cap);
      if (live1) Y[y1 + r / 2u] = i2a_situ(sum.y * global, up.y * global, a.gate_cap, a.up_cap);
    }
  }
}

template [[host_name("i2_aot_e_gu_packed")]] [[kernel]] void i2_aot_e_gu_packed<2>(
    const device float* X [[buffer(0)]], const device ulong2* W [[buffer(1)]],
    const device uchar2* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_t_moe_args& a [[buffer(5)]],
    device float* Y [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);

template [[host_name("i2_aot_e_gu16_packed")]] [[kernel]] void i2_aot_e_gu_packed<4>(
    const device float* X [[buffer(0)]], const device ulong2* W [[buffer(1)]],
    const device uchar2* S [[buffer(2)]], const device float* gscale [[buffer(3)]],
    const device int* PICKS [[buffer(4)]], constant i2_aot_t_moe_args& a [[buffer(5)]],
    device float* Y [[buffer(6)]], const device float* E2M1 [[buffer(7)]],
    const device float* E4M3 [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]);

[[kernel]] void i2_aot_ctl_apply3_gf2_rows(const device float* X [[buffer(0)]], const device float2* GF0 [[buffer(1)]],
    const device float2* GF1 [[buffer(2)]], const device float2* GF2 [[buffer(3)]], const device float* C0 [[buffer(4)]],
    const device float* C1 [[buffer(5)]], const device float* C2 [[buffer(6)]], const device float* MSG [[buffer(7)]],
    device float* O0 [[buffer(8)]], device float* O1 [[buffer(9)]], device float* O2 [[buffer(10)]],
    constant i2_aot_ctl_r_args& a [[buffer(11)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint row = group.y, lane = tid % 32u, j = group.z;
  if (col >= a.h || row >= a.rows) return;
  const device float* x = X + ((size_t)j * a.rows + row) * a.rank;
  const device float2* gf = (j == 0u ? GF0 : j == 1u ? GF1 : GF2) + (size_t)col * a.rank;
  const device float* cur = (j == 0u ? C0 : j == 1u ? C1 : C2) + (size_t)(row * a.xs[j] + a.xb[j]) * a.h;
  const device float* msg = MSG + (size_t)(row * 4u + a.msg0 + j) * a.h;
  device float* out = (j == 0u ? O0 : j == 1u ? O1 : O2) + (size_t)row * a.h;
  const float x0 = x[lane], x1 = x[lane + 32u];
  const float2 gf0 = gf[lane], gf1 = gf[lane + 32u];
  const float g = simd_sum(fma(x1, gf1.x, fma(x0, gf0.x, 0.0f)));
  const float f = simd_sum(fma(x1, gf1.y, fma(x0, gf0.y, 0.0f)));
  if (lane == 0u) {
    const float cv = cur[col];
    out[col] = cv - precise::tanh(f) * cv + precise::tanh(g) * msg[col];
  }
}

[[kernel]] void i2_aot_ctl_apply1_gf2_rows(const device float* X [[buffer(0)]], const device float2* GF [[buffer(1)]],
    const device float* CUR [[buffer(2)]], const device float* MSG [[buffer(3)]], device float* OUT [[buffer(4)]],
    constant i2_aot_ctl1_r_args& a [[buffer(5)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  const uint col = group.x * 8u + tid / 32u;
  const uint row = group.y, lane = tid % 32u;
  if (col >= a.m0) return;
  const device float* x = X + (size_t)row * a.k;
  const device float2* gf = GF + (size_t)col * a.k;
  const device float* cur = CUR + (size_t)(row * a.cxs + a.cxb) * a.h;
  const device float* msg = MSG + (size_t)(row * a.mxs + a.mxb) * a.h;
  device float* out = OUT + (size_t)(row * a.ys + a.yb) * a.h;
  const float x0 = x[lane], x1 = x[lane + 32u];
  const float2 gf0 = gf[lane], gf1 = gf[lane + 32u];
  const float g = simd_sum(fma(x1, gf1.x, fma(x0, gf0.x, 0.0f)));
  const float f = simd_sum(fma(x1, gf1.y, fma(x0, gf0.y, 0.0f)));
  if (lane == 0u) {
    const float cv = cur[col];
    out[col] = cv - precise::tanh(f) * cv + precise::tanh(g) * msg[col];
  }
}
