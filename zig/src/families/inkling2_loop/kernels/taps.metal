#include <metal_stdlib>
using namespace metal;

struct i2t_tap_args { uint width, rows, stride, column; };

// f32 rows into bf16 columns [column, column + width) of rows `stride` wide, rounded to nearest even.
[[kernel]] void i2t_tap(const device float* X [[buffer(0)]], constant i2t_tap_args& a [[buffer(1)]],
    device ushort* Y [[buffer(2)]], uint2 p [[thread_position_in_grid]]) {
  if (p.x >= a.width || p.y >= a.rows) return;
  const uint u = as_type<uint>(X[(size_t)p.y * a.width + p.x]);
  const bool nan = (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
  Y[(size_t)p.y * a.stride + a.column + p.x] = nan ? ushort((u >> 16) | 0x40u) : ushort((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
