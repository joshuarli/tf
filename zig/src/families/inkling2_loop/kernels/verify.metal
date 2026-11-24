// Each kernel keeps the bits of the batch kernel it twins; a group's earlier rows come from captures, not the cache.

inline float i2v_bf(const device bfloat* p, size_t i) { return float(p[i]); }

struct i2v_conv_row { device float* ring; uint pos, first; };
struct i2v_conv_args { uint channels, heads, ring, pos, layout; float norm; uint residual; };

[[kernel]] void i2v_conv(const device float* X [[buffer(0)]], const device bfloat* W [[buffer(1)]],
    const device float* D [[buffer(2)]], constant i2v_conv_args& a [[buffer(3)]],
    device float* O [[buffer(4)]], constant i2v_conv_row* rows [[buffer(5)]], device float* CAP [[buffer(6)]],
    uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.channels) return;
  const uint c = p.x, pos = rows[p.y].pos, first = rows[p.y].first;
  const uint start = pos - (p.y - first);
  device float* ring = rows[p.y].ring;
  const device float* x = X + (size_t)p.y * a.channels;
  float sum = x[c];
  for (uint k = 0; k < 4u; ++k) {
    const int at = int(pos) + int(k) - 3;
    if (at < 0) continue;
    const float v = at == int(pos) ? x[c]
      : (uint(at) >= start ? X[(size_t)(first + uint(at) - start) * a.channels + c] : ring[(uint(at) % a.ring) * a.channels + c]);
    const size_t tap = a.layout ? (size_t)(3u - k) * a.channels + c : (size_t)c * 4u + k;
    sum = fma(v, i2v_bf(W, tap), sum);
  }
  if (a.heads > 0u) {
    const uint head = c / (a.channels / a.heads);
    for (uint lag = 0; lag < 4u; ++lag) {
      if (lag > pos) continue;
      const float v = lag == 0u ? x[c]
        : (pos - lag >= start ? X[(size_t)(first + pos - lag - start) * a.channels + c] : ring[((pos - lag) % a.ring) * a.channels + c]);
      sum = fma(v, D[(size_t)p.y * a.heads * 4u + head * 4u + lag], sum);
    }
  }
  CAP[(size_t)p.y * a.channels + c] = x[c];
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

// i2_split_serial with positions from `gpos` on read from the group's captured rows, KH/VH row first + (j - gpos).
inline void i2v_split_serial(const device float* Q, const device half* KEYS, const device half* VALUES,
    const device half* KH, const device half* VH, uint first, uint gpos,
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
    const size_t at = fresh ? ((size_t)(first + j - gpos) * a.kv + kv) * D : ((size_t)rj * a.kv + kv) * D;
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
    const device half* KH, const device half* VH, uint first_row, uint gpos,
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
      const size_t at = fresh ? ((size_t)(first_row + j0 + r - gpos) * a.kv + kv) * D : ((size_t)rj * a.kv + kv) * D;
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

struct i2v_row { device half* keys; device half* values; i2_attn_parts args; uint first, gpos; };

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
  const uint first = rows[tg.y].first, gpos = rows[tg.y].gpos;
  if (i2_blocked(a, nth2.x)) i2v_split_blocks(Q, rows[tg.y].keys, rows[tg.y].values, KH, VH, first, gpos, B, sink, a, PART, tg.x, tid2.x, sg, lane, KS4, VS4);
  else i2v_split_serial(Q, rows[tg.y].keys, rows[tg.y].values, KH, VH, first, gpos, B, sink, a, PART, tg.x, tid2.x, sg, lane, nth2.x, (threadgroup float4*)KS4, (threadgroup float4*)VS4);
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
