"""Attribute Inkling 2 Mini's native Metal decode step: weight floor, bench medians, per-kernel timeline, trace window.

    python3 campaign/kernels/inkling_profile.py weights --model DIR [--gbs B]
    python3 campaign/kernels/inkling_profile.py timeline --log FILE [--rounds N]
    python3 campaign/kernels/inkling_profile.py trace --trace FILE --steps N [--process tf-inkling2-loop-run]
    python3 campaign/kernels/inkling_profile.py bandwidth [--bytes BYTES]

`weights` reads config.json and the text shard's safetensors header to count the parameters one decode token
actually reads: attention and convolution weights of the 48 executed slots, the top-k routed experts, the
controllers of the core layers, the depth-mix queries, the embedding row and the unembedding rows. `timeline`
reduces the runner's TF_METAL4_TIMELINE receipt (one GPU timestamp per dispatch) to a per-kernel table. `trace`
reduces one xctrace Metal System Trace window to its command-buffer GPU intervals; on macOS the trace carries no
per-kernel intervals (Metal's Shader Timeline is unsupported there), so the timeline is the per-kernel source.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1, "I32": 4, "U32": 4}


def read_config(model: Path) -> dict:
    return json.loads((model / "config.json").read_text())["text_config"]


def text_shard(model: Path) -> Path:
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    names = set(index.values())
    if len(names) != 1:
        raise SystemExit(f"the text stack must be one shard, found {sorted(names)}")
    return model / names.pop()


def read_header(model: Path) -> dict:
    with text_shard(model).open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


def tensor_bytes(header: dict, name: str) -> int:
    entry = header.get(name)
    if entry is None:
        raise SystemExit(f"the checkpoint has no {name}")
    begin, end = entry["data_offsets"]
    return end - begin


def is_full(c: dict, layer: int) -> bool:
    local = c["local_layer_ids"]
    if len(local) == c["num_hidden_layers"] and all(isinstance(v, str) for v in local):
        return local[layer] == "hybrid"
    return layer not in set(local)


def executed_layers(c: dict) -> list[int]:
    loop = c["loop_config"]
    layers = c["num_hidden_layers"]
    core = list(range(loop["core_start"], loop["core_end"]))
    order = list(range(loop["core_start"])) + core * loop["num_passes"] + list(range(loop["core_end"], layers))
    return order


def layer_names(c: dict, layer: int) -> list[str]:
    prefix = f"model.llm.layers.{layer}."
    full = is_full(c, layer)
    names: list[str] = []
    if full:
        names += [prefix + n for n in ("q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight")]
    else:
        names.append(prefix + "attn_norm.weight")
    names += [
        prefix + "attn.wq_du.weight",
        prefix + "attn.wk_dv.weight",
        prefix + "attn.wv_dv.weight",
        prefix + "attn.wo_ud.weight",
        prefix + "attn.wr_du.weight",
        prefix + "attn.gate_attn.weight",
        prefix + "attn.attn_sink_proj.weight",
        prefix + "attn.q_l2_clip.scale",
        prefix + "attn.rel_l2_clip.scale",
        prefix + "attn.rel_logits_proj.proj",
        prefix + ("attn.k_sconv.static_w" if full else "attn.k_sconv.weight"),
        prefix + ("attn.v_sconv.static_w" if full else "attn.v_sconv.weight"),
    ]
    if full:
        names += [prefix + "attn.k_sconv.weight_proj", prefix + "attn.v_sconv.weight_proj"]
    names += [
        prefix + "attn_sconv.weight",
        prefix + "mlp_norm.weight",
        prefix + "mlp_sconv.weight",
        prefix + "mlp.gate.weight",
        prefix + "mlp.gate.bias",
    ]
    names += expert_names(c, layer)
    if c["loop_config"]["core_start"] <= layer < c["loop_config"]["core_end"]:
        for site in ("attn", "k", "v", "mlp"):
            names += [prefix + f"controllers.{site}.{n}" for n in ("down_weight", "gain_weight", "forget_weight")]
    return names


def expert_names(c: dict, layer: int) -> list[str]:
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    names = [prefix + n for n in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")]
    return names


def expert_global_names(layer: int) -> list[str]:
    prefix = f"model.llm.layers.{layer}.mlp.experts."
    return [prefix + n for n in ("w13_weight_amax", "w13_weight_outer_scale_denominator", "w2_weight_amax", "w2_weight_outer_scale_denominator")]


CONTROLLER = re.compile(r"controllers\.")


def bucket_of(suffix: str) -> str:
    if CONTROLLER.search(suffix):
        return "controllers"
    if "sconv" in suffix:
        return "convolutions"
    if suffix.startswith("attn."):
        return "attention projections"
    if suffix.endswith("norm.weight") or suffix.startswith("mlp.gate"):
        return "norms and router"
    raise SystemExit(f"unclassified tensor {suffix}")


def weight_floor(model: Path, expert_read: str = "topk") -> dict:
    c = read_config(model)
    header = read_header(model)
    layers = c["num_hidden_layers"]
    experts = c["n_routed_experts"]
    topk = c["num_experts_per_tok"]
    hidden = c["hidden_size"]
    unpadded = c["unpadded_vocab_size"]
    order = executed_layers(c)
    buckets = {"attention projections": 0, "convolutions": 0, "norms and router": 0, "routed experts (top-k)": 0,
               "controllers": 0, "depth-mix queries": 0, "embedding row + norms": 0, "unembedding": 0, "cross-pass": 0}
    per_layer: list[tuple[int, int]] = []
    for layer in order:
        total = 0
        for name in layer_names(c, layer):
            suffix = name.split(f"layers.{layer}.", 1)[1]
            size = tensor_bytes(header, name)
            if suffix.startswith("mlp.experts."):
                continue
            total += size
            buckets[bucket_of(suffix)] += size
        expert_weight_bytes = sum(tensor_bytes(header, n) for n in expert_names(c, layer))
        per_expert = expert_weight_bytes // experts
        routed = per_expert * (experts if expert_read == "all" else topk)
        routed += sum(tensor_bytes(header, n) for n in expert_global_names(layer) if n in header)
        key = "routed experts (top-k)" if expert_read == "topk" else "routed experts (all)"
        buckets[key] = buckets.get(key, 0) + routed
        total += routed
        query = header.get(f"model.llm.attn_res.attn_queries.{layer - 1}") if layer > 0 else None
        if query is not None:
            size = query["data_offsets"][1] - query["data_offsets"][0]
            buckets["depth-mix queries"] += size
            total += size
        mlp_query = f"model.llm.attn_res.mlp_queries.{layer}"
        size = tensor_bytes(header, mlp_query)
        buckets["depth-mix queries"] += size
        total += size
        per_layer.append((layer, total))
    buckets["embedding row + norms"] = hidden * DTYPE_BYTES["BF16"] + 2 * hidden * DTYPE_BYTES["BF16"]
    buckets["unembedding"] = unpadded * hidden * DTYPE_BYTES["BF16"]
    buckets["cross-pass"] = 2 * (tensor_bytes(header, "model.llm.middle_cycle_cross_pass_queries") +
                                 tensor_bytes(header, "model.llm.middle_cycle_cross_pass_kv_queries")) + \
        tensor_bytes(header, "model.llm.attn_res.output_query")
    total = sum(buckets.values())
    return {"config": c, "order": order, "buckets": buckets, "total": total, "per_layer": per_layer}


def format_bytes(n: int) -> str:
    return f"{n / 1e6:.2f} MB"


def cmd_weights(args: argparse.Namespace) -> int:
    floor = weight_floor(Path(args.model), args.expert_read)
    total = floor["total"]
    print(f"executed slots: {len(floor['order'])} (physical layers {floor['order']})")
    print(f"{'class':<28}{'bytes/token':>16}{'share':>10}")
    for name, size in sorted(floor["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"{name:<28}{size:>16,}{100 * size / total:>9.1f}%")
    print(f"{'total':<28}{total:>16,}{100:>9.1f}%")
    if args.gbs:
        print(f"floor at {args.gbs:.1f} GB/s: {total / (args.gbs * 1e9) * 1e3:.3f} ms a step")
    else:
        print("floor at 400 GB/s: %.3f ms a step" % (total / 400e9 * 1e3))
        print("floor at 500 GB/s: %.3f ms a step" % (total / 500e9 * 1e3))
        print("floor at 546 GB/s: %.3f ms a step" % (total / 546e9 * 1e3))
    return 0


TIMELINE_HEADER = re.compile(r"metal 4 timeline: (\S+), (\d+) rounds after (\d+) warm: (\d+) encoders, (\d+) stages and (\d+) dispatches a round")
TIMELINE_SPAN = re.compile(r"metal 4 timeline: (\S+): ([\d.]+) ms a round from the first stamp to the last, ([\d.]+) in stages, ([\d.]+) between encoders")
TIMELINE_ROW = re.compile(r"^\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s{2,}(\S.*)$")


def parse_timeline(text: str) -> dict:
    out: dict = {"rounds": 0, "warm": 0, "encoders": 0, "stages": 0, "dispatches": 0, "ms": 0.0,
                 "inside_ms": 0.0, "between_ms": 0.0, "kernels": []}
    in_table = False
    for raw in text.splitlines():
        line = re.sub(r"^(?:info|warn|error|debug): ?", "", raw)
        m = TIMELINE_HEADER.search(line)
        if m:
            out.update(rounds=int(m.group(2)), warm=int(m.group(3)), encoders=int(m.group(4)),
                       stages=int(m.group(5)), dispatches=int(m.group(6)))
            continue
        m = TIMELINE_SPAN.search(line)
        if m:
            out.update(ms=float(m.group(2)), inside_ms=float(m.group(3)), between_ms=float(m.group(4)))
            continue
        if " by kernel:" in line:
            in_table = True
            continue
        if in_table:
            m = TIMELINE_ROW.match(line)
            if m:
                out["kernels"].append({"ms": float(m.group(1)), "calls": float(m.group(2)),
                                       "us": float(m.group(3)), "name": m.group(4).strip()})
            elif line.startswith("metal 4 timeline:"):
                in_table = False
    return out


def cmd_timeline(args: argparse.Namespace) -> int:
    parsed = parse_timeline(Path(args.log).read_text(errors="replace"))
    if not parsed["kernels"]:
        print("no metal 4 timeline receipt in the log (set TF_METAL4_TIMELINE and run the bench)", file=sys.stderr)
        return 1
    print(f"{parsed['rounds']} timed rounds after {parsed['warm']} warm: {parsed['encoders']} encoders, "
          f"{parsed['stages']} stages, {parsed['dispatches']} dispatches a round")
    print(f"{parsed['ms']:.3f} ms a round ({parsed['inside_ms']:.3f} in stages, {parsed['between_ms']:.3f} between encoders)")
    spans = [k["ms"] for k in parsed["kernels"]]
    total = sum(spans) or parsed["ms"]
    print(f"{'kernel':<26}{'calls/step':>11}{'us/step':>11}{'us/call':>10}{'share':>8}")
    for k in sorted(parsed["kernels"], key=lambda k: -k["ms"]):
        print(f"{k['name']:<26}{k['calls']:>11.1f}{k['ms'] * 1e3:>11.1f}{k['us']:>10.1f}{100 * k['ms'] / total:>7.1f}%")
    print(f"{'sum of kernels':<26}{'':>11}{total * 1e3:>11.1f}{'':>10}{100:>7.1f}%")
    return 0


XCTRACE_TABLE = "metal-gpu-intervals"


def export_trace(trace: Path, table: str, out: Path) -> None:
    subprocess.run(["xctrace", "export", "--input", str(trace), "--xpath",
                    f'/trace-toc/run[@number="1"]/data/table[@schema="{table}"]'],
                   stdout=out.open("wb"), check=True)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def resolve(elem, ids: dict):
    if elem.get("ref"):
        return ids.get(elem.get("ref"), elem)
    return elem


def label_of(elem, ids: dict) -> str:
    if elem is None:
        return ""
    elem = resolve(elem, ids)
    return elem.get("fmt") or "".join(elem.itertext()).strip()


def number_of(elem, ids: dict) -> int:
    elem = resolve(elem, ids)
    return int("".join(elem.itertext()).strip().replace(",", ""))


def cmd_trace(args: argparse.Namespace) -> int:
    import xml.etree.ElementTree as ET

    with tempfile.TemporaryDirectory() as tmp:
        xml = Path(tmp) / "gpu-intervals.xml"
        export_trace(Path(args.trace), XCTRACE_TABLE, xml)
        tree = ET.parse(xml)
    ids = {e.get("id"): e for e in tree.iter() if e.get("id")}
    rows = []
    for row in tree.iter():
        if local_name(row.tag) != "row":
            continue
        fields: dict = {}
        for child in row:
            fields.setdefault(local_name(child.tag), child)
        label = label_of(fields.get("formatted-label"), ids)
        process = label_of(fields.get("process"), ids)
        if args.process not in label and args.process not in process:
            continue
        duration = fields.get("duration")
        if duration is None:
            continue
        rows.append((number_of(fields["start-time"], ids), number_of(duration, ids), label))
    rows.sort()
    if len(rows) < args.steps:
        print(f"only {len(rows)} command-buffer intervals for {args.process}", file=sys.stderr)
        return 1
    window = rows[-args.steps:]
    ms = [ns / 1e6 for _, ns, _ in window]
    print(f"{len(window)} command buffers from one trace, {ms[0]:.3f}...{ms[-1]:.3f} ms")
    print(f"per decode step: median {statistics.median(ms):.3f} ms, mean {statistics.fmean(ms):.3f}, "
          f"min {min(ms):.3f}, max {max(ms):.3f}; sum {sum(ms):.3f} ms")
    print("the trace's per-command-buffer data only; per-kernel GPU time comes from the runner's Metal 4 timeline")
    return 0


SWIFT = r'''
import Metal
import Foundation

let device = MTLCreateSystemDefaultDevice()!
let bytes = __BYTES__
let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
memset(buffer.contents(), 1, bytes)
let source = """
#include <metal_stdlib>
using namespace metal;
kernel void read_bw(device const float4* x [[buffer(0)]], device float* out [[buffer(1)]],
                    constant uint& per [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float4 s = float4(0.0f);
    uint base = gid * per;
    for (uint i = 0; i < per; i++) { s += x[base + i]; }
    if (s.x == -1.0f) { out[gid] = s.x; }
}
"""
let library = try! device.makeLibrary(source: source, options: nil)
let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "read_bw")!)
let out = device.makeBuffer(length: 1 << 16, options: .storageModeShared)!
let threads = 1 << 16
let chunks = bytes / 16
let per = chunks / threads
let queue = device.makeCommandQueue()!
var samples: [Double] = []
for trial in 0..<7 {
    let cb = queue.makeCommandBuffer()!
    let encoder = cb.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    encoder.setBuffer(out, offset: 0, index: 1)
    var n = UInt32(per)
    encoder.setBytes(&n, length: 4, index: 2)
    encoder.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    if trial >= 2 {
        samples.append(cb.gpuEndTime - cb.gpuStartTime)
    }
}
samples.sort()
let median = samples[samples.count / 2]
print("bytes=\(bytes) median_seconds=\(median) gbs=\(Double(bytes) / median / 1e9)")
'''


def cmd_bandwidth(args: argparse.Namespace) -> int:
    script = SWIFT.replace("__BYTES__", str(args.bytes))
    with tempfile.NamedTemporaryFile("w", suffix=".swift", delete=False) as f:
        f.write(script)
        path = f.name
    try:
        result = subprocess.run(["swift", path], capture_output=True, text=True)
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode != 0:
        print(result.stderr.strip()[-2000:], file=sys.stderr)
        return result.returncode
    print(result.stdout.strip())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    w = sub.add_parser("weights", help="count the weights one decode token reads")
    w.add_argument("--model", required=True)
    w.add_argument("--gbs", type=float, default=0.0)
    w.add_argument("--expert-read", choices=("topk", "all"), default="topk")
    w.set_defaults(func=cmd_weights)
    t = sub.add_parser("timeline", help="reduce the runner's Metal 4 timeline receipt")
    t.add_argument("--log", required=True)
    t.set_defaults(func=cmd_timeline)
    r = sub.add_parser("trace", help="reduce one Metal System Trace window")
    r.add_argument("--trace", required=True)
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--process", default="tf-inkling2-loop-run")
    r.set_defaults(func=cmd_trace)
    b = sub.add_parser("bandwidth", help="measure Metal streaming read bandwidth with Swift")
    b.add_argument("--bytes", type=lambda v: int(v, 0), default=1 << 30)
    b.set_defaults(func=cmd_bandwidth)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
