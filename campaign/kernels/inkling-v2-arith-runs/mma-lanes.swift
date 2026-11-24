import Metal
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void lanes(device float* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 m;
  thread vec<float, 2>& e = reinterpret_cast<thread vec<float, 2>&>(m.thread_elements());
  e = float2(float(lane * 2), float(lane * 2 + 1));
  simdgroup_store(m, out, 8);
}
"""
let lib = try! dev.makeLibrary(source: src, options: nil)
let p = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "lanes")!)
let out = dev.makeBuffer(length: 256, options: .storageModeShared)!
let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(p); e.setBuffer(out, offset: 0, index: 0)
e.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
let v = out.contents().bindMemory(to: Float.self, capacity: 64)
var ok = true
for lane in 0..<32 {
  let qid = lane >> 2
  let fm = (qid & 4) | ((lane >> 1) & 3)
  let fn = ((qid & 2) << 1) | ((lane & 1) << 1)
  if Int(v[fm * 8 + fn]) != lane * 2 || Int(v[fm * 8 + fn + 1]) != lane * 2 + 1 { ok = false; print("lane \(lane) mismatch") }
}
print("lane map fm=(qid&4)|((lane>>1)&3), fn=((qid&2)<<1)|((lane&1)<<1): \(ok ? "holds" : "fails")")
for r in 0..<8 { print((0..<8).map { String(format: "%3.0f", v[r * 8 + $0]) }.joined(separator: " ")) }
