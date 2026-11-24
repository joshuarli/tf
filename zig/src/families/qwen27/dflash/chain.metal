#include <metal_stdlib>
using namespace metal;

// Written once the verify returns: the path's length, the context's rows, ring start and own slot, the path's rows.
struct dfc_params { uint count, context, begin, own; uint path[16]; };
// A dispatch's shape: the first row's position, uint4 words a row, rows, the ring's slots, the block and the keys.
struct dfc_rows { ulong base; uint width, rows, window, block, keys, pad; };

// The path's tap rows in path order; rows past its length repeat its last.
[[kernel]] void dfc_gather(device const uint4* taps [[buffer(0)]], device const dfc_params& p [[buffer(1)]],
    constant dfc_rows& r [[buffer(2)]], device uint4* out [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= r.width || at.y >= r.rows) return;
  out[(size_t)at.y * r.width + at.x] = taps[(size_t)p.path[at.y] * r.width + at.x];
}

// Keys and values below the path's length into the ring at their positions' slots; later rows write nothing.
[[kernel]] void dfc_append(device const uint4* k [[buffer(0)]], device const uint4* v [[buffer(1)]],
    device const dfc_params& p [[buffer(2)]], constant dfc_rows& r [[buffer(3)]], device uint4* cache_k [[buffer(4)]],
    device uint4* cache_v [[buffer(5)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= r.width || at.y >= r.rows || at.y >= p.count) return;
  const size_t slot = (size_t)((r.base + at.y) % r.window);
  cache_k[slot * r.width + at.x] = k[(size_t)at.y * r.width + at.x];
  cache_v[slot * r.width + at.x] = v[(size_t)at.y * r.width + at.x];
}

// One layer's keys or values for the block: the context's rows in position order from the ring, then its own rows.
[[kernel]] void dfc_context(device const uint4* cache [[buffer(0)]], device const uint4* own [[buffer(1)]],
    device const dfc_params& p [[buffer(2)]], constant dfc_rows& r [[buffer(3)]], device uint4* out [[buffer(4)]],
    uint2 at [[thread_position_in_grid]]) {
  const uint j = at.y;
  if (at.x >= r.width || j >= r.keys || j >= p.context + r.block) return;
  out[(size_t)j * r.width + at.x] = j < p.context ? cache[(size_t)((p.begin + j) % r.window) * r.width + at.x]
                                                  : own[(size_t)(j - p.context) * r.width + at.x];
}

// The block's own rows behind an unwrapped context, in the ring buffer itself.
[[kernel]] void dfc_own(device const uint4* own [[buffer(0)]], device const dfc_params& p [[buffer(1)]],
    constant dfc_rows& r [[buffer(2)]], device uint4* cache [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= r.width || at.y >= r.block) return;
  cache[(size_t)(p.own + at.y) * r.width + at.x] = own[(size_t)at.y * r.width + at.x];
}
