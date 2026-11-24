import Metal
import Foundation

let args = CommandLine.arguments
let srcPath = args[1]
let fast = args.count > 2 ? args[2] == "fast" : true
let source = try String(contentsOfFile: srcPath, encoding: .utf8)
let device = MTLCreateSystemDefaultDevice()!
let opts = MTLCompileOptions()
opts.mathMode = .safe
opts.mathFloatingPointFunctions = fast ? .fast : .precise
opts.languageVersion = MTLLanguageVersion(rawValue: (4 << 16) + 1)!
let lib = try device.makeLibrary(source: source, options: opts)
let fn = lib.makeFunction(name: "tf_full_rope")!
let pipe = try device.makeComputePipelineState(function: fn)
let queue = device.makeCommandQueue()!

let dim = 128, pairs = 64, heads = 1
let theta: Float = 10000
let positions: [UInt64] = Array(stride(from: 0, to: 33000, by: 7)).map { UInt64($0) }
let rows = positions.count
// x: left = 1, right = 0 for every pair
var x = [UInt16](repeating: 0, count: rows * dim)
for r in 0..<rows { for c in 0..<pairs { x[r * dim + c] = 0x3f80 } }
let xb = device.makeBuffer(bytes: x, length: x.count * 2, options: .storageModeShared)!
let pb = device.makeBuffer(bytes: positions, length: rows * 8, options: .storageModeShared)!
let yb = device.makeBuffer(length: rows * dim * 2, options: .storageModeShared)!
struct Rope { var rows: UInt32; var heads: UInt32; var dim: UInt32; var x_stride: UInt32; var y_stride: UInt32; var theta: Float }
var a = Rope(rows: UInt32(rows), heads: UInt32(heads), dim: UInt32(dim), x_stride: UInt32(dim), y_stride: UInt32(dim), theta: theta)
let cb = queue.makeCommandBuffer()!
let e = cb.makeComputeCommandEncoder()!
e.setComputePipelineState(pipe)
e.setBuffer(xb, offset: 0, index: 0)
e.setBuffer(pb, offset: 0, index: 1)
e.setBytes(&a, length: MemoryLayout<Rope>.size, index: 2)
e.setBuffer(yb, offset: 0, index: 3)
e.dispatchThreads(MTLSize(width: pairs, height: heads, depth: rows), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
e.endEncoding()
cb.commit()
cb.waitUntilCompleted()
let y = yb.contents().bindMemory(to: UInt16.self, capacity: rows * dim)
func bf(_ v: Double) -> Float { let f = Float(v); let u = f.bitPattern; let r = (u &+ 0x7fff &+ ((u >> 16) & 1)) >> 16; return Float(bitPattern: r << 16) }
func val(_ u: UInt16) -> Float { Float(bitPattern: UInt32(u) << 16) }
// buckets by position
var buckets: [(Int, Int)] = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 33000)]
for (lo, hi) in buckets {
  var maxAngleErr = 0.0, sumAngleErr = 0.0, n = 0, mism = 0, maxAbs: Float = 0
  for (r, p) in positions.enumerated() where Int(p) >= lo && Int(p) < hi {
    for c in 0..<pairs {
      let inv = pow(Double(theta), -Double(c) / Double(pairs))
      let ang = Double(p) * inv
      let cs = cos(ang), sn = sin(ang)
      let gc = Double(val(y[r * dim + c])), gs = Double(val(y[r * dim + c + pairs]))
      var d = atan2(gs, gc) - atan2(sn, cs)
      while d > Double.pi { d -= 2 * Double.pi }
      while d < -Double.pi { d += 2 * Double.pi }
      maxAngleErr = max(maxAngleErr, abs(d)); sumAngleErr += abs(d); n += 1
      if val(y[r * dim + c]) != bf(cs) || val(y[r * dim + c + pairs]) != bf(sn) { mism += 1 }
      maxAbs = max(maxAbs, abs(val(y[r * dim + c]) - bf(cs)), abs(val(y[r * dim + c + pairs]) - bf(sn)))
    }
  }
  print(String(format: "%@ positions [%5d,%5d): pairs %6d, angle err max %.5f mean %.6f rad, bf16 mismatches %5.2f%%, max |diff| %.5f", fast ? "fast   " : "precise", lo, hi, n, maxAngleErr, sumAngleErr / Double(n), 100.0 * Double(mism) / Double(n), maxAbs))
}
