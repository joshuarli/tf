import Metal
import Foundation

let dev = MTLCreateSystemDefaultDevice()!
print(dev.name)
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void mma_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const float* C [[buffer(2)]], device float* D [[buffer(3)]], uint tg [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 a, b, c, d;
  simdgroup_load(a, A + tg * 64, 8);
  simdgroup_load(b, B + tg * 64, 8);
  simdgroup_load(c, C + tg * 64, 8);
  simdgroup_multiply_accumulate(d, a, b, c);
  simdgroup_store(d, D + tg * 64, 8);
}
kernel void mma_f32_chain(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device const float* C [[buffer(2)]], device float* D [[buffer(3)]], uint tg [[threadgroup_position_in_grid]]) {
  simdgroup_float8x8 a, b, d;
  simdgroup_load(d, C + tg * 64, 8);
  for (uint s = 0; s < 4; ++s) {
    simdgroup_load(a, A + tg * 256 + s * 8, 32);
    simdgroup_load(b, B + tg * 256 + s * 64, 8);
    simdgroup_multiply_accumulate(d, a, b, d);
  }
  simdgroup_store(d, D + tg * 64, 8);
}
kernel void mma_h(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]], device const float* C [[buffer(2)]], device float* D [[buffer(3)]], uint tg [[threadgroup_position_in_grid]]) {
  simdgroup_half8x8 a, b;
  simdgroup_float8x8 c, d;
  simdgroup_load(a, A + tg * 64, 8);
  simdgroup_load(b, B + tg * 64, 8);
  simdgroup_load(c, C + tg * 64, 8);
  simdgroup_multiply_accumulate(d, a, b, c);
  simdgroup_store(d, D + tg * 64, 8);
}
"""
let opts = MTLCompileOptions()
opts.mathMode = .safe
let lib = try! dev.makeLibrary(source: src, options: opts)
func pipe(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

var rng = SystemRandomNumberGenerator()
func rnd(_ spread: Int) -> Float {
  let m = Float.random(in: 1.0..<2.0, using: &rng)
  let e = Int.random(in: -spread...spread, using: &rng)
  let s: Float = Bool.random(using: &rng) ? 1 : -1
  return s * m * Float(sign: .plus, exponent: e, significand: 1)
}

func run(_ p: MTLComputePipelineState, _ a: MTLBuffer, _ b: MTLBuffer, _ c: MTLBuffer, _ d: MTLBuffer, _ tiles: Int) {
  let cb = q.makeCommandBuffer()!
  let e = cb.makeComputeCommandEncoder()!
  e.setComputePipelineState(p)
  e.setBuffer(a, offset: 0, index: 0); e.setBuffer(b, offset: 0, index: 1); e.setBuffer(c, offset: 0, index: 2); e.setBuffer(d, offset: 0, index: 3)
  e.dispatchThreadgroups(MTLSize(width: tiles, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
  e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}

let tiles = 20000
let spread = Int(CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "12")!
// single MMA, f32
do {
  var A = [Float](repeating: 0, count: tiles * 64), B = A, C = A
  for i in 0..<A.count { A[i] = rnd(spread); B[i] = rnd(spread); C[i] = rnd(spread) }
  let ab = dev.makeBuffer(bytes: A, length: A.count * 4, options: .storageModeShared)!
  let bb = dev.makeBuffer(bytes: B, length: B.count * 4, options: .storageModeShared)!
  let cbuf = dev.makeBuffer(bytes: C, length: C.count * 4, options: .storageModeShared)!
  let db = dev.makeBuffer(length: C.count * 4, options: .storageModeShared)!
  run(pipe("mma_f32"), ab, bb, cbuf, db, tiles)
  let D = Array(UnsafeBufferPointer(start: db.contents().bindMemory(to: Float.self, capacity: C.count), count: C.count))
  var m = [String: Int]()
  let names = ["fma_fwd", "fma_rev", "mul_add_fwd", "exact_once", "fma_fwd_c_last", "pair_tree_fma", "fma_rev_c_last"]
  for n in names { m[n] = 0 }
  for t in 0..<tiles {
    for i in 0..<8 { for j in 0..<8 {
      let c = C[t*64 + i*8 + j]
      let got = D[t*64 + i*8 + j]
      var f = c; for k in 0..<8 { f = f.addingProduct(A[t*64+i*8+k], B[t*64+k*8+j]) }
      var r = c; for k in (0..<8).reversed() { r = r.addingProduct(A[t*64+i*8+k], B[t*64+k*8+j]) }
      var ma = c; for k in 0..<8 { ma = ma + A[t*64+i*8+k] * B[t*64+k*8+j] }
      var ex: Double = Double(c); for k in 0..<8 { ex += Double(A[t*64+i*8+k]) * Double(B[t*64+k*8+j]) }
      var cl: Float = 0; for k in 0..<8 { cl = cl.addingProduct(A[t*64+i*8+k], B[t*64+k*8+j]) }; cl = cl + c
      var rcl: Float = 0; for k in (0..<8).reversed() { rcl = rcl.addingProduct(A[t*64+i*8+k], B[t*64+k*8+j]) }; rcl = rcl + c
      var pr = [Float](repeating: 0, count: 8)
      for k in 0..<8 { pr[k] = A[t*64+i*8+k] * B[t*64+k*8+j] }
      let pt = ((pr[0]+pr[1])+(pr[2]+pr[3])) + ((pr[4]+pr[5])+(pr[6]+pr[7])) + c
      if got.bitPattern == f.bitPattern { m["fma_fwd"]! += 1 }
      if got.bitPattern == r.bitPattern { m["fma_rev"]! += 1 }
      if got.bitPattern == ma.bitPattern { m["mul_add_fwd"]! += 1 }
      if got.bitPattern == Float(ex).bitPattern { m["exact_once"]! += 1 }
      if got.bitPattern == cl.bitPattern { m["fma_fwd_c_last"]! += 1 }
      if got.bitPattern == pt.bitPattern { m["pair_tree_fma"]! += 1 }
      if got.bitPattern == rcl.bitPattern { m["fma_rev_c_last"]! += 1 }
    }}
  }
  print("f32 single MMA, \(tiles*64) elements, spread 2^±\(spread):")
  for n in names { print("  \(n): \(m[n]!)") }
}
