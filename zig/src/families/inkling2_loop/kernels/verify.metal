// Each kernel keeps the bits of the batch kernel it twins; a row's ancestors come from captures, not the cache.

inline float i2v_bf(const device bfloat* p, size_t i) { return float(p[i]); }

struct i2v_conv_row { device float* ring; uint pos, start; uchar back[4]; };
struct i2v_conv_args { uint channels, heads, ring, pos, layout; float norm; uint residual, topk; };

inline float i2v_input(const device float* X, const device float* WTS, uint row, uint c, constant i2v_conv_args& a) {
  if (a.topk == 0u) return X[(size_t)row * a.channels + c];
  float sum = 0.0f;
  for (uint j = 0; j < a.topk; ++j) sum = fma(WTS[(size_t)row * a.topk + j], X[((size_t)row * a.topk + j) * a.channels + c], sum);
  return sum;
}

[[kernel]] void i2v_conv(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    const device float* D [[buffer(2)]], constant i2v_conv_args& a [[buffer(3)]],
    device float* O [[buffer(4)]], constant i2v_conv_row* rows [[buffer(5)]], device float* CAP [[buffer(6)]],
    const device float* WTS [[buffer(7)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels) return;
  const uint c = p.x, pos = rows[p.y].pos, start = rows[p.y].start;
  constant uchar* back = rows[p.y].back;
  device float* ring = rows[p.y].ring;
  const float xc = i2v_input(X, WTS, p.y, c, a);
  float sum = xc;
  for (uint k = 0; k < 4u; ++k) {
    const int at = int(pos) + int(k) - 3;
    if (at < 0) continue;
    const float v = at == int(pos) ? xc
      : (uint(at) >= start ? i2v_input(X, WTS, back[pos - uint(at) - 1u], c, a) : ring[(uint(at) % a.ring) * a.channels + c]);
    const size_t tap = a.layout ? (size_t)(3u - k) * a.channels + c : (size_t)c * 4u + k;
    sum = fma(v, i2v_bf(W, tap), sum);
  }
  if (a.heads > 0u) {
    const uint head = c / (a.channels / a.heads);
    for (uint lag = 0; lag < 4u; ++lag) {
      if (lag > pos) continue;
      const float v = lag == 0u ? xc
        : (pos - lag >= start ? i2v_input(X, WTS, back[lag - 1u], c, a) : ring[((pos - lag) % a.ring) * a.channels + c]);
      sum = fma(v, D[(size_t)p.y * a.heads * 4u + head * 4u + lag], sum);
    }
  }
  CAP[(size_t)p.y * a.channels + c] = xc;
  const size_t out = (size_t)p.y * a.channels + c;
  const float value = sum * a.norm;
  if (a.residual) O[out] += value; else O[out] = value;
}

struct i2v_rel_args { uint heads, kv, dim, rel, extent, window, pos, tau; float scale, floor, alpha, eps; };

[[kernel]] void i2v_store_rel(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    const device float* R [[buffer(2)]], const device bfloat* P [[buffer(3)]],
    constant i2v_rel_args& a [[buffer(4)]], device float* O [[buffer(5)]],
    constant uint* POS [[buffer(6)]], device half* KH [[buffer(7)]], device half* VH [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  const uint row = group.y, count = a.kv * a.dim;
  if (group.x < a.kv) {
    const size_t at = (size_t)row * count + group.x * a.dim;
    float sum = 0.0f;
    for (uint i = tid; i < a.dim; i += 256u) sum = fma(V[at + i], V[at + i], sum);
    sum = simd_sum(sum);
    if (tid % 32u == 0u) part[tid / 32u] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8u; ++i) total += part[i];
    const float inv = 1.0f / sqrt(total / float(a.dim) + a.eps);
    for (uint i = tid; i < a.dim; i += 256u) {
      KH[at + i] = K[at + i];
      VH[at + i] = V[at + i] * inv;
    }
    return;
  }
  const uint i = (group.x - a.kv) * 256u + tid;
  const uint cols = a.extent + 1u;
  if (i >= cols * a.heads) return;
  const uint h = i / cols, dist = i % cols;
  const size_t out = (size_t)row * cols * a.heads + i;
  if (dist == a.extent) { O[out] = 0.0f; return; }
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(POS[row]) + 1.0f) / a.floor)) : 1.0f;
  float sum = 0.0f;
  const device float* r = R + ((size_t)row * a.heads + h) * a.rel;
  for (uint u = 0; u < a.rel; ++u) sum = fma(i2v_bf(P, u * a.extent + dist), r[u], sum);
  O[out] = sum * tau;
}

// i2_split_serial with positions from `gpos` on read from the group's captured rows, KH/VH row path[j - gpos].
inline void i2v_split_serial(const device float* Q, const device half* KEYS, const device half* VALUES,
    const device half* KH, const device half* VH, constant uchar* path, uint gpos,
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
    const bool fresh = j >= gpos;
    const uint rj = a.window ? (j % a.window) : j;
    const size_t at = fresh ? ((size_t)path[j - gpos] * a.kv + kv) * D : ((size_t)rj * a.kv + kv) * D;
    const device half* kp = fresh ? KH : KEYS;
    const device half* vp = fresh ? VH : VALUES;
    const float bias = B[h * (a.extent + 1u) + col] - sk;
    for (uint v = tid; v < D / 4u; v += nth) {
      KB4[v] = float4(*reinterpret_cast<const device packed_half4*>(kp + at + v * 4u));
      VB4[v] = float4(*reinterpret_cast<const device packed_half4*>(vp + at + v * 4u));
    }
    for (uint i = (D / 4u) * 4u + tid; i < D; i += nth) {
      KB[i] = kp[at + i];
      VB[i] = vp[at + i];
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

// i2_split_blocks with the same source rule for each staged row.
inline void i2v_split_blocks(const device float* Q, const device half* KEYS, const device half* VALUES,
    const device half* KH, const device half* VH, constant uchar* path, uint gpos,
    const device float* B, const device float* sink, constant i2_attn_parts& a, device float* PART,
    uint tgp, uint tid, uint sg, uint lane, threadgroup half4* KS4, threadgroup half4* VS4) {
  constexpr uint D = 128u, G = 3u, N = 96u, R = 32u, CELLS = (R * 32u + N - 1u) / N;
  const uint kv = tgp / a.splits, s = tgp % a.splits, h = kv * G + sg;
  const float tau = a.tau ? 1.0f + a.alpha * precise::log(max(1.0f, (float(a.pos) + 1.0f) / a.floor)) : 1.0f;
  float qv[4];
  for (uint j = 0; j < 4u; ++j) qv[j] = Q[h * D + lane + 32u * j] * a.scale * tau;
  const float sk = sink[h];
  const uint lo = a.window ? (a.pos + 1u > a.window ? a.pos + 1u - a.window : 0u) : 0u;
  const size_t base = ((size_t)h * a.splits + s) * (D + 2u);
  const uint js = lo + s * a.chunk, je = min(js + a.chunk, a.pos + 1u);
  float top = -INFINITY, mass = 0.0f, acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (uint j0 = js; j0 < je; j0 += R) {
    const uint n = min(R, je - j0);
    const uint first = a.window ? j0 % a.window : j0;
    half4 kc[CELLS], vc[CELLS];
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      const uint r = min(e / 32u, n - 1u), j = first + r;
      const uint rj = a.window && j >= a.window ? j - a.window : j;
      const bool fresh = j0 + r >= gpos;
      const size_t at = fresh ? ((size_t)path[j0 + r - gpos] * a.kv + kv) * D : ((size_t)rj * a.kv + kv) * D;
      const device half* kr = (fresh ? KH : KEYS) + at + e % 32u;
      const device half* vr = (fresh ? VH : VALUES) + at + e % 32u;
      kc[i] = half4(kr[0], kr[32], kr[64], kr[96]);
      vc[i] = half4(vr[0], vr[32], vr[64], vr[96]);
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
      float dot = 0.0f;
      dot = fma(qv[0], kk.x, dot);
      dot = fma(qv[1], kk.y, dot);
      dot = fma(qv[2], kk.z, dot);
      dot = fma(qv[3], kk.w, dot);
      p[r] = dot;
    }
    const float score = i2_rows_sum(p, lane) + own;
    float run = lane < n ? score : -INFINITY;
    for (ushort off = 1; off < 32; off <<= 1) {
      const float other = simd_shuffle_up(run, off);
      if (lane >= off) run = max(other, run);
    }
    const float peak = max(top, run);
    const float before = lane == 0u ? top : simd_shuffle_up(peak, ushort(1));
    const float rescales = precise::exp(before - peak);
    const float weights = precise::exp(score - peak);
    if (n == R) {
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        const float rescale = simd_shuffle(rescales, ushort(r));
        const float w = simd_shuffle(weights, ushort(r));
        mass = mass * rescale + w;
        const float4 vv = float4(VS4[r * 32u + lane]);
        acc[0] = acc[0] * rescale + w * vv.x;
        acc[1] = acc[1] * rescale + w * vv.y;
        acc[2] = acc[2] * rescale + w * vv.z;
        acc[3] = acc[3] * rescale + w * vv.w;
      }
    } else {
      for (uint r = 0; r < n; ++r) {
        const float rescale = simd_shuffle(rescales, ushort(r));
        const float w = simd_shuffle(weights, ushort(r));
        mass = mass * rescale + w;
        const float4 vv = float4(VS4[r * 32u + lane]);
        acc[0] = acc[0] * rescale + w * vv.x;
        acc[1] = acc[1] * rescale + w * vv.y;
        acc[2] = acc[2] * rescale + w * vv.z;
        acc[3] = acc[3] * rescale + w * vv.w;
      }
    }
    top = simd_shuffle(peak, ushort(n - 1u));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) {
    PART[base] = top;
    PART[base + 1u] = mass;
  }
  for (uint u = 0; u < 4u; ++u) PART[base + 2u + lane + 32u * u] = acc[u];
}

struct i2v_row { device half* keys; device half* values; i2_attn_parts args; uint gpos; uchar path[16]; };

[[kernel]] void i2v_split(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2v_row* rows [[buffer(3)]], device float* PART [[buffer(4)]],
    const device half* KH [[buffer(5)]], const device half* VH [[buffer(6)]],
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
  const uint gpos = rows[tg.y].gpos;
  constant uchar* path = rows[tg.y].path;
  if (i2_blocked(a, nth2.x)) i2v_split_blocks(Q, rows[tg.y].keys, rows[tg.y].values, KH, VH, path, gpos, B, sink, a, PART, tg.x, tid2.x, sg, lane, KS4, VS4);
  else i2v_split_serial(Q, rows[tg.y].keys, rows[tg.y].values, KH, VH, path, gpos, B, sink, a, PART, tg.x, tid2.x, sg, lane, nth2.x, (threadgroup float4*)KS4, (threadgroup float4*)VS4);
}

[[kernel]] void i2v_split_shared(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2v_row* rows [[buffer(3)]], device float* PART [[buffer(4)]],
    const device half* KH [[buffer(5)]], const device half* VH [[buffer(6)]],
    constant uint& total_rows [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128u, G = 3u, N = 96u, R = 32u, CELLS = (R * 32u + N - 1u) / N;
  threadgroup half4 KS4[32 * 32];
  threadgroup half4 VS4[32 * 32];
  const uint r0 = tg.y * 4u;
  if (r0 >= total_rows) return;
  const uint m_cnt = min(4u, total_rows - r0);
  constant i2_attn_parts& a0 = rows[r0].args;
  const uint kv = tg.x / a0.splits, s = tg.x % a0.splits, h = kv * G + sg;
  if (tg.x >= a0.kv * a0.splits) return;
  const device half* KEYS = rows[r0].keys;
  const device half* VALUES = rows[r0].values;
  const uint gpos = rows[r0].gpos;
  const uint win = a0.window, ext = a0.extent, ext1 = ext + 1u;
  float qv[4][4], sk[4], top[4], mass[4], acc[4][4];
  uint js[4], je[4], pos[4];
  uint min_js = 0xffffffffu, max_je = 0u;
  for (uint m = 0u; m < m_cnt; ++m) {
    constant i2_attn_parts& am = rows[r0 + m].args;
    pos[m] = am.pos;
    const float tau = am.tau ? 1.0f + am.alpha * precise::log(max(1.0f, (float(am.pos) + 1.0f) / am.floor)) : 1.0f;
    const device float* qm = Q + size_t(r0 + m) * am.heads * D + h * D;
    for (uint j = 0u; j < 4u; ++j) qv[m][j] = qm[lane + 32u * j] * am.scale * tau;
    sk[m] = sink[size_t(r0 + m) * am.heads + h];
    const uint lo = win ? (am.pos + 1u > win ? am.pos + 1u - win : 0u) : 0u;
    js[m] = lo + s * am.chunk;
    je[m] = min(js[m] + am.chunk, am.pos + 1u);
    if (js[m] < je[m]) {
      min_js = min(min_js, js[m]);
      max_je = max(max_je, je[m]);
    }
    top[m] = -INFINITY;
    mass[m] = 0.0f;
    acc[m][0] = 0.0f; acc[m][1] = 0.0f; acc[m][2] = 0.0f; acc[m][3] = 0.0f;
  }
  const uint shared_end = min(max_je, gpos);
  for (uint j0 = min_js; j0 < shared_end; j0 += R) {
    const uint n = min(R, shared_end - j0);
    const uint first = win ? j0 % win : j0;
    half4 kc[CELLS], vc[CELLS];
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      const uint r = min(e / 32u, n - 1u), j = first + r;
      const uint rj = win && j >= win ? j - win : j;
      const size_t at = ((size_t)rj * a0.kv + kv) * D;
      const device half* kr = KEYS + at + e % 32u;
      const device half* vr = VALUES + at + e % 32u;
      kc[i] = half4(kr[0], kr[32], kr[64], kr[96]);
      vc[i] = half4(vr[0], vr[32], vr[64], vr[96]);
    }
#pragma unroll
    for (uint i = 0; i < CELLS; ++i) {
      const uint e = tid + i * N;
      if (e < R * 32u) {
        KS4[e] = kc[i];
        VS4[e] = vc[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint m = 0u; m < m_cnt; ++m) {
      const uint aj = max(j0, js[m]), bj = min(j0 + n, je[m]);
      if (aj >= bj) continue;
      const uint r_lo = aj - j0, r_hi = bj - j0;
      const device float* bm = B + size_t(r0 + m) * a0.heads * ext1 + h * ext1;
      float own = 0.0f;
      if (lane >= r_lo && lane < r_hi) own = bm[min(pos[m] - (j0 + lane), ext)] - sk[m];
      float p[32];
#pragma unroll
      for (uint r = 0; r < R; ++r) {
        const float4 kk = float4(KS4[r * 32u + lane]);
        float dot = 0.0f;
        dot = fma(qv[m][0], kk.x, dot);
        dot = fma(qv[m][1], kk.y, dot);
        dot = fma(qv[m][2], kk.z, dot);
        dot = fma(qv[m][3], kk.w, dot);
        p[r] = dot;
      }
      const float score = i2_rows_sum(p, lane) + own;
      float run = (lane >= r_lo && lane < r_hi) ? score : -INFINITY;
      for (ushort off = 1; off < 32; off <<= 1) {
        const float other = simd_shuffle_up(run, off);
        if (lane >= off) run = max(other, run);
      }
      const float peak = max(top[m], run);
      const float before = lane == r_lo ? top[m] : simd_shuffle_up(peak, ushort(1));
      const float rescales = precise::exp(before - peak);
      const float weights = precise::exp(score - peak);
      if (r_lo == 0u && r_hi == R) {
#pragma unroll
        for (uint r = 0; r < R; ++r) {
          const float rescale = simd_shuffle(rescales, ushort(r));
          const float w = simd_shuffle(weights, ushort(r));
          mass[m] = mass[m] * rescale + w;
          const float4 vv = float4(VS4[r * 32u + lane]);
          acc[m][0] = acc[m][0] * rescale + w * vv.x;
          acc[m][1] = acc[m][1] * rescale + w * vv.y;
          acc[m][2] = acc[m][2] * rescale + w * vv.z;
          acc[m][3] = acc[m][3] * rescale + w * vv.w;
        }
      } else {
        for (uint r = r_lo; r < r_hi; ++r) {
          const float rescale = simd_shuffle(rescales, ushort(r));
          const float w = simd_shuffle(weights, ushort(r));
          mass[m] = mass[m] * rescale + w;
          const float4 vv = float4(VS4[r * 32u + lane]);
          acc[m][0] = acc[m][0] * rescale + w * vv.x;
          acc[m][1] = acc[m][1] * rescale + w * vv.y;
          acc[m][2] = acc[m][2] * rescale + w * vv.z;
          acc[m][3] = acc[m][3] * rescale + w * vv.w;
        }
      }
      top[m] = simd_shuffle(peak, ushort(r_hi - 1u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (max_je > gpos) {
    for (uint m = 0u; m < m_cnt; ++m) {
      const uint fj0 = max(js[m], gpos);
      if (fj0 >= je[m]) continue;
      const uint fn = je[m] - fj0;
      constant uchar* path = rows[r0 + m].path;
      const device float* bm = B + size_t(r0 + m) * a0.heads * ext1 + h * ext1;
      for (uint j = fj0; j < je[m]; ++j) {
        const size_t at = ((size_t)path[j - gpos] * a0.kv + kv) * D;
        const device half* kr = KH + at + lane;
        const device half* vr = VH + at + lane;
        const float4 kk = float4(half4(kr[0], kr[32], kr[64], kr[96]));
        const float4 vv = float4(half4(vr[0], vr[32], vr[64], vr[96]));
        float dot = 0.0f;
        dot = fma(qv[m][0], kk.x, dot);
        dot = fma(qv[m][1], kk.y, dot);
        dot = fma(qv[m][2], kk.z, dot);
        dot = fma(qv[m][3], kk.w, dot);
        const float bias = bm[min(pos[m] - j, ext)] - sk[m];
        const float score = simd_sum(dot) + bias;
        const float peak = max(top[m], score);
        const float rescale = precise::exp(top[m] - peak);
        const float w = precise::exp(score - peak);
        mass[m] = mass[m] * rescale + w;
        acc[m][0] = acc[m][0] * rescale + w * vv.x;
        acc[m][1] = acc[m][1] * rescale + w * vv.y;
        acc[m][2] = acc[m][2] * rescale + w * vv.z;
        acc[m][3] = acc[m][3] * rescale + w * vv.w;
        top[m] = peak;
      }
      (void)fn;
    }
  }
  for (uint m = 0u; m < m_cnt; ++m) {
    device float* pm = PART + size_t(r0 + m) * a0.heads * 16u * (D + 2u);
    const size_t base = ((size_t)h * a0.splits + s) * (D + 2u);
    if (lane == 0u) {
      pm[base] = top[m];
      pm[base + 1u] = mass[m];
    }
    for (uint u = 0u; u < 4u; ++u) pm[base + 2u + lane + 32u * u] = acc[m][u];
  }
}

// Captures hold `stride` rows a slot: KV [slots][2][stride][kvw], CAP laid out as prefill.ConvCapture.
struct i2v_commit_row { device half* keys; device half* values; device float* conv[4]; uint pos, capacity, keep, conv_keep; };
struct i2v_commit_slot { uint local, locals_before, globals_before, pad; };
struct i2v_commit_args { uint kvw, hidden, rows, stride, slots, window, ring; };

[[kernel]] void i2v_commit_kv(const device half* KV [[buffer(0)]], constant i2v_commit_args& a [[buffer(1)]],
    constant i2v_commit_row* rows [[buffer(2)]], constant i2v_commit_slot* slots [[buffer(3)]],
    uint3 p [[thread_position_in_grid]]) {
  const uint i = p.x, row = p.y, slot = p.z / 2u, kind = p.z % 2u;
  if (i >= a.kvw || row >= a.rows) return;
  const i2v_commit_row r = rows[row];
  if (r.keep == 0u) return;
  const i2v_commit_slot s = slots[slot];
  const size_t base = ((size_t)s.locals_before * a.window + (size_t)s.globals_before * r.capacity) * a.kvw;
  const size_t at = base + (size_t)(s.local ? r.pos % a.window : r.pos) * a.kvw + i;
  const half v = KV[(((size_t)slot * 2u + kind) * a.stride + row) * a.kvw + i];
  if (kind == 0u) r.keys[at] = v; else r.values[at] = v;
}

[[kernel]] void i2v_commit_conv(const device float* CAP [[buffer(0)]], constant i2v_commit_args& a [[buffer(1)]],
    constant i2v_commit_row* rows [[buffer(2)]], uint3 p [[thread_position_in_grid]]) {
  const uint c = p.x, row = p.y, slot = p.z / 4u, site = p.z % 4u;
  const uint width = site < 2u ? a.kvw : a.hidden;
  if (c >= width || row >= a.rows) return;
  const i2v_commit_row r = rows[row];
  if (r.keep == 0u || r.conv_keep == 0u) return;
  const uint site_at = site == 0u ? 0u : site == 1u ? a.kvw : site == 2u ? 2u * a.kvw : 2u * a.kvw + a.hidden;
  const size_t src = (size_t)slot * a.stride * (2u * a.kvw + 2u * a.hidden) + (size_t)site_at * a.stride + (size_t)row * width + c;
  r.conv[site][((size_t)slot * a.ring + r.pos % a.ring) * width + c] = CAP[src];
}

// The host scan's pick: the first index of the largest value; a NaN never wins and a NaN at index 0 is kept.
struct i2v_argmax_args { uint vocab; };

[[kernel]] void i2v_argmax(const device float* L [[buffer(0)]], constant i2v_argmax_args& a [[buffer(1)]],
    device uint* OUT [[buffer(2)]], uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tv[32];
  threadgroup uint ti[32];
  const device float* x = L + (size_t)row * a.vocab;
  float v = -INFINITY;
  uint i = 0xffffffffu;
  for (uint k = tid; k < a.vocab; k += nth) {
    const float y = x[k];
    if (y > v || (y == v && k < i)) {
      v = y;
      i = k;
    }
  }
  for (uint off = 16u; off > 0u; off /= 2u) {
    const float ov = simd_shuffle_xor(v, off);
    const uint oi = simd_shuffle_xor(i, off);
    if (ov > v || (ov == v && oi < i)) {
      v = ov;
      i = oi;
    }
  }
  if (lane == 0u) {
    tv[sg] = v;
    ti[sg] = i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint g = 1; g < nth / 32u; ++g) {
      if (tv[g] > v || (tv[g] == v && ti[g] < i)) {
        v = tv[g];
        i = ti[g];
      }
    }
    OUT[row] = (isnan(x[0]) || i == 0xffffffffu) ? 0u : i;
  }
}

[[kernel]] void i2v_argmax_p1(const device float* L [[buffer(0)]], constant i2v_argmax_args& a [[buffer(1)]],
    device float* PV [[buffer(2)]], device uint* PI [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tv[8];
  threadgroup uint ti[8];
  const device float* x = L + (size_t)tg.y * a.vocab;
  float v = -INFINITY;
  uint i = 0xffffffffu;
  for (uint k = tg.x * 256u + tid; k < a.vocab; k += 64u * 256u) {
    const float y = x[k];
    if (y > v || (y == v && k < i)) { v = y; i = k; }
  }
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(v, off);
    const uint oi = simd_shuffle_xor(i, off);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0u) { tv[sg] = v; ti[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    for (uint g = 1u; g < 8u; ++g) {
      if (tv[g] > v || (tv[g] == v && ti[g] < i)) { v = tv[g]; i = ti[g]; }
    }
    PV[tg.y * 64u + tg.x] = v;
    PI[tg.y * 64u + tg.x] = i;
  }
}

[[kernel]] void i2v_argmax_p2(const device float* L [[buffer(0)]], const device float* PV [[buffer(1)]],
    const device uint* PI [[buffer(2)]], constant i2v_argmax_args& a [[buffer(3)]],
    device uint* OUT [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float tv[2];
  threadgroup uint ti[2];
  float v = PV[row * 64u + tid];
  uint i = PI[row * 64u + tid];
  for (ushort off = 16; off > 0; off >>= 1) {
    const float ov = simd_shuffle_xor(v, off);
    const uint oi = simd_shuffle_xor(i, off);
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
  }
  if (lane == 0u) { tv[sg] = v; ti[sg] = i; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0u) {
    if (tv[1] > v || (tv[1] == v && ti[1] < i)) { v = tv[1]; i = ti[1]; }
    OUT[row] = (isnan(L[(size_t)row * a.vocab]) || i == 0xffffffffu) ? 0u : i;
  }
}

struct i2v_stream { uint first, size; };

// ACC: a stream's deepest accepted row (lowest-numbered matching child each step); KEEP: path rows; KEPT: its length.
[[kernel]] void i2v_accept(const device uint* IDS [[buffer(0)]], const device uint* T [[buffer(1)]],
    constant i2v_stream* streams [[buffer(2)]], constant uint& count [[buffer(3)]], device uint* ACC [[buffer(4)]],
    constant uint* PARENT [[buffer(5)]], device uint* KEEP [[buffer(6)]], device uint* KEPT [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const uint first = streams[i].first, end = first + streams[i].size;
  for (uint r = first; r < end; ++r) KEEP[r] = 0u;
  uint at = first, n = 0u;
  KEEP[at] = 1u;
  for (;;) {
    uint next = 0xffffffffu;
    for (uint c = at + 1u; c < end; ++c) {
      if (PARENT[c] == at && IDS[c] == T[at]) {
        next = c;
        break;
      }
    }
    if (next == 0xffffffffu) break;
    at = next;
    ++n;
    KEEP[at] = 1u;
  }
  ACC[i] = at - first;
  KEPT[i] = n + 1u;
}

struct i2v_round_row { device half* keys; device half* values; device float* conv[4]; uint pos, capacity, owner, depth; };

[[kernel]] void i2v_round_kv(const device half* KV [[buffer(0)]], constant i2v_commit_args& a [[buffer(1)]],
    constant i2v_round_row* rows [[buffer(2)]], constant i2v_commit_slot* slots [[buffer(3)]],
    const device uint* KEEP [[buffer(4)]], uint3 p [[thread_position_in_grid]]) {
  const uint i = p.x, row = p.y, slot = p.z / 2u, kind = p.z % 2u;
  if (i >= a.kvw || row >= a.rows) return;
  const i2v_round_row r = rows[row];
  if (KEEP[row] == 0u) return;
  const i2v_commit_slot s = slots[slot];
  const size_t base = ((size_t)s.locals_before * a.window + (size_t)s.globals_before * r.capacity) * a.kvw;
  const size_t at = base + (size_t)(s.local ? r.pos % a.window : r.pos) * a.kvw + i;
  const half v = KV[(((size_t)slot * 2u + kind) * a.stride + row) * a.kvw + i];
  if (kind == 0u) r.keys[at] = v; else r.values[at] = v;
}

[[kernel]] void i2v_round_conv(const device float* CAP [[buffer(0)]], constant i2v_commit_args& a [[buffer(1)]],
    constant i2v_round_row* rows [[buffer(2)]], const device uint* KEEP [[buffer(3)]],
    const device uint* KEPT [[buffer(4)]], uint3 p [[thread_position_in_grid]]) {
  const uint c = p.x, row = p.y, slot = p.z / 4u, site = p.z % 4u;
  const uint width = site < 2u ? a.kvw : a.hidden;
  if (c >= width || row >= a.rows) return;
  const i2v_round_row r = rows[row];
  if (KEEP[row] == 0u || r.depth + a.ring < KEPT[r.owner]) return;
  const uint site_at = site == 0u ? 0u : site == 1u ? a.kvw : site == 2u ? 2u * a.kvw : 2u * a.kvw + a.hidden;
  const size_t src = (size_t)slot * a.stride * (2u * a.kvw + 2u * a.hidden) + (size_t)site_at * a.stride + (size_t)row * width + c;
  r.conv[site][((size_t)slot * a.ring + r.pos % a.ring) * width + c] = CAP[src];
}

struct i2v_qkv_row { device float* k_ring; device float* v_ring; uint pos, start; uchar back[4]; };

[[kernel]] void i2v_qkv(const device float* K [[buffer(0)]], const device float* V [[buffer(1)]],
    const device bfloat* KW [[buffer(2)]], const device bfloat* VW [[buffer(3)]], const device float* KD [[buffer(4)]],
    const device float* VD [[buffer(5)]], device float* Q [[buffer(6)]], const device bfloat* QS [[buffer(7)]],
    const device float* R [[buffer(8)]], const device bfloat* RS [[buffer(9)]], const device bfloat* P [[buffer(10)]],
    constant i2_qkv_args& a [[buffer(11)]], device float* REL [[buffer(12)]], constant i2v_qkv_row* rows [[buffer(13)]],
    device float* KC [[buffer(14)]], device float* VC [[buffer(15)]], device float* KCAP [[buffer(16)]],
    device float* VCAP [[buffer(17)]], device half* KH [[buffer(18)]], device half* VH [[buffer(19)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  threadgroup float part[8];
  threadgroup float rc[2][16];
  const uint row = group.y, pos = rows[row].pos, start = rows[row].start;
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
      if (lag > pos) {
        kh[lag] = 0.0f;
        vh[lag] = 0.0f;
      } else if (pos - lag >= start) {
        const size_t back = (size_t)rows[row].back[lag - 1u] * count + c;
        kh[lag] = K[back];
        vh[lag] = V[back];
      } else {
        const size_t r = ((pos - lag) % 4u) * count + c;
        kh[lag] = rows[row].k_ring[r];
        vh[lag] = rows[row].v_ring[r];
      }
    }
    kc = i2_qkv_conv(kh, pos, KW, a, c, KD + (size_t)row * a.kv * 4u);
    vc = i2_qkv_conv(vh, pos, VW, a, c, VD + (size_t)row * a.kv * 4u);
    KCAP[at] = kh[0];
    VCAP[at] = vh[0];
  }
  const float2 kv = i2_qkv_rows(kc, vc, live, a, part, tid);
  if (!live) return;
  KC[(size_t)row * count + c] = kv.x;
  VC[(size_t)row * count + c] = vc;
  KH[(size_t)row * count + c] = kv.x;
  VH[(size_t)row * count + c] = kv.y;
}
