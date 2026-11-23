#include <metal_stdlib>
using namespace metal;

// Grouped verification: the batch step's rows, several of them a stream's consecutive positions. Each kernel here is
// the batch kernel it names with the same arithmetic, except that a row's earlier positions in its own group come from
// this forward's rows instead of the stream's cache, and nothing is written to a cache: the accepted rows are committed
// afterwards from the captures. A group of one row computes exactly what the batch step computes for it.

inline float i2v_bf(const device bfloat* p, size_t i) { return float(p[i]); }

// i2p_batch_conv for grouped rows. `first` is the chunk row of the row's group start, at position pos - (row - first);
// CAP receives the row's convolution input for the commit.
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

// i2p_batch_store_rel for grouped rows: the normalized keys and values go to KH/VH [rows][kv][dim] (the slot's
// capture), not the cache.
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

// i2_batch_split for grouped rows: positions from `gpos` on are this forward's rows, KH/VH row first + (j - gpos).
struct i2_attn_parts { uint heads, kv, dim, rel, extent, window, pos, tau, splits, chunk; float scale, floor, alpha; };
struct i2v_row { device half* keys; device half* values; i2_attn_parts args; uint first, gpos; };

[[kernel]] void i2v_split(const device float* Q [[buffer(0)]], const device float* B [[buffer(1)]],
    const device float* sink [[buffer(2)]], constant i2v_row* rows [[buffer(3)]], device float* PART [[buffer(4)]],
    const device half* KH [[buffer(5)]], const device half* VH [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint2 nth2 [[threads_per_threadgroup]]) {
  const uint tid = tid2.x, nth = nth2.x;
  constant i2_attn_parts& a = rows[tg.y].args;
  const uint first = rows[tg.y].first, gpos = rows[tg.y].gpos;
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

// The accepted rows into their streams' caches: every slot's normalized keys and values at the row's position (a
// local slot's ring row), and the convolution inputs of a group's newest ring-length rows. A row with keep 0 is a
// rejected draft; conv_keep 0 marks an accepted row the newer accepted rows of its group push out of the ring. The
// captures hold `stride` rows a slot: KV [slots][2][stride][kvw], CAP as the verification's ConvCapture.
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
