import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma_chain(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* D [[buffer(2)]], constant uint& steps [[buffer(3)]], uint tg [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 a, b, d = simdgroup_float8x8(0.0f);
  for (uint s = 0; s < steps; ++s) {
    simdgroup_load(a, A + tg * 64 * steps + s * 8, 8 * steps);
    simdgroup_load(b, B + tg * 64 * steps + s * 64, 8);
    simdgroup_multiply_accumulate(d, a, b, d);
  }
  simdgroup_store(d, D + tg * 64, 8);
}
kernel void scalar_chain(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* D [[buffer(2)]], constant uint& steps [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
  for (uint e = lane; e < 64; e += 32) {
    const uint i = e / 8, j = e % 8;
    float d = 0.0f;
    for (uint k = 0; k < 8 * steps; ++k) d = fma(A[tg * 64 * steps + i * 8 * steps + k], B[tg * 64 * steps + k * 8 + j], d);
    D[tg * 64 + e] = d;
  }
}
kernel void mma_rate(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  simdgroup_float8x8 a(1.0001f), b(0.9999f), c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
  for (uint i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    simdgroup_multiply_accumulate(c3, a, b, c3);
  }
  threadgroup float t[64];
  simdgroup_store(c0, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c1, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c2, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c3, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
}
kernel void mma_rate_h(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  simdgroup_half8x8 a(half(1.0001f)), b(half(0.9999f));
  simdgroup_float8x8 c0(0.0f), c1(0.0f), c2(0.0f), c3(0.0f);
  for (uint i = 0; i < iters; ++i) {
    simdgroup_multiply_accumulate(c0, a, b, c0);
    simdgroup_multiply_accumulate(c1, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c2);
    simdgroup_multiply_accumulate(c3, a, b, c3);
  }
  threadgroup float t[64];
  simdgroup_store(c0, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c1, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c2, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
  simdgroup_store(c3, t, 8);
  if (gid % 32 == 0 && t[0] == 123.0f) out[0] = t[1];
}
kernel void fma_rate(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  float a = 1.0001f + gid * 1e-9f, b = 0.9999f;
  float c[16];
  for (uint j = 0; j < 16; ++j) c[j] = float(j);
  for (uint i = 0; i < iters; ++i) {
#pragma unroll
    for (uint j = 0; j < 16; ++j) c[j] = fma(a, b, c[j]);
  }
  float s = 0; for (uint j = 0; j < 16; ++j) s += c[j];
  if (s == 123.0f) out[0] = s;
}
"""
let opts = MTLCompileOptions()
opts.mathMode = .safe
let lib = try! dev.makeLibrary(source: src, options: opts)
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

func special(_ rng: inout SystemRandomNumberGenerator) -> Float {
  let r = Int.random(in: 0..<20, using: &rng)
  switch r {
  case 0: return 0.0
  case 1: return -0.0
  case 2: return Float.leastNonzeroMagnitude * Float(Int.random(in: 1..<1000, using: &rng))
  case 3: return -Float.leastNormalMagnitude * Float.random(in: 0.5..<1.5, using: &rng)
  case 4: return Float.random(in: 1..<2, using: &rng) * 1e30
  case 5: return -Float.random(in: 1..<2, using: &rng) * 1e30
  case 6: return Float.infinity
  case 7: return Float.random(in: 1..<2, using: &rng) * 1e-20
  default:
    let m = Float.random(in: 1.0..<2.0, using: &rng)
    let e = Int.random(in: -20...20, using: &rng)
    return (Bool.random(using: &rng) ? 1 : -1) * m * Float(sign: .plus, exponent: e, significand: 1)
  }
}

var rng = SystemRandomNumberGenerator()
let tiles = 4000
for steps in [1, 9] {
  var A = [Float](repeating: 0, count: tiles * 64 * steps), B = A
  for i in 0..<A.count { A[i] = special(&rng); B[i] = special(&rng) }
  let ab = dev.makeBuffer(bytes: A, length: A.count * 4, options: .storageModeShared)!
  let bb = dev.makeBuffer(bytes: B, length: B.count * 4, options: .storageModeShared)!
  let d1 = dev.makeBuffer(length: tiles * 64 * 4, options: .storageModeShared)!
  let d2 = dev.makeBuffer(length: tiles * 64 * 4, options: .storageModeShared)!
  var st = UInt32(steps)
  for (p, d) in [(pipe("mma_chain"), d1), (pipe("scalar_chain"), d2)] {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p); e.setBuffer(ab, offset: 0, index: 0); e.setBuffer(bb, offset: 0, index: 1); e.setBuffer(d, offset: 0, index: 2); e.setBytes(&st, length: 4, index: 3)
    e.dispatchThreadgroups(MTLSize(width: tiles, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  }
  let x = d1.contents().bindMemory(to: UInt32.self, capacity: tiles * 64)
  let y = d2.contents().bindMemory(to: UInt32.self, capacity: tiles * 64)
  var same = 0, bothNaN = 0, diff = 0, zeroSign = 0
  for i in 0..<tiles * 64 {
    let fx = Float(bitPattern: x[i]), fy = Float(bitPattern: y[i])
    if x[i] == y[i] { same += 1 } else if fx.isNaN && fy.isNaN { bothNaN += 1 } else if fx == 0 && fy == 0 { zeroSign += 1 } else { diff += 1; if diff < 5 { print("  diff mma=\(fx) scalar=\(fy)") } }
  }
  print("steps \(steps): same \(same), both NaN (payload differs) \(bothNaN), zero sign differs \(zeroSign), other \(diff)")
}

func rate(_ name: String, _ threads: Int, _ iters: UInt32, _ flopsPerThreadIter: Double) {
  let p = pipe(name)
  let out = dev.makeBuffer(length: 16, options: .storageModeShared)!
  var it = iters
  var best = 1e9
  for _ in 0..<5 {
    let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
    e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0); e.setBytes(&it, length: 4, index: 1)
    e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    best = min(best, cb.gpuEndTime - cb.gpuStartTime)
  }
  let flops = Double(threads) * Double(iters) * flopsPerThreadIter
  print(String(format: "%@: %.2f TFLOP/s (%.3f ms)", name, flops / best / 1e12, best * 1e3))
}
rate("mma_rate", 40 * 1024 * 8, 2000, 4.0 * 1024.0 / 32.0)
rate("mma_rate_h", 40 * 1024 * 8, 2000, 4.0 * 1024.0 / 32.0)
rate("fma_rate", 40 * 1024 * 8, 2000, 16.0 * 2.0)
