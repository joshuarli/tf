"""Trace every parity outlier to its first divergent tensor, stage by stage.

TensorFold's per-row logit error is compared with the oracle's own two-path error on the same row. Every row where
TensorFold's is more than twice the oracle's is traced down TensorFold's own 16-row logit path and down both
oracle paths: embedding, each layer's attention output, router logits and picks, per-pick expert outputs, the
dense MLP output, the residual after the layer, the final norm and the logits. The first stage where TensorFold
departs by more than the oracle's own two-path spread at that stage carries the verdict.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

VOCAB = 150144
THRESHOLD = 2.0
NAMED = {("binary-search-edges", 3), ("arrow-etl-pipeline", 2), ("json-invoice-records", 32), ("zig-slice-bounds", 200)}


def bits(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def bf16(x: np.ndarray) -> np.ndarray:
    u = np.asarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def maxabs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())


def margins(logits: np.ndarray, topk: int) -> tuple[float, float, float]:
    """The k-th and (k+1)-th router logits and the gap between them, where a pick flips."""
    ranked = np.sort(logits.astype(np.float64))[::-1]
    return float(ranked[topk - 1]), float(ranked[topk]), float(ranked[topk - 1] - ranked[topk])


class TfRow:
    def __init__(self, path: Path, layers: int, hidden: int, experts: int, topk: int):
        self.raw = np.fromfile(path, dtype=np.uint8)
        self.layers, self.hidden, self.experts, self.topk = layers, hidden, experts, topk
        self.size0 = 4 * hidden + 2 * hidden
        self.sizel = 4 * hidden + 2 * hidden * 3 + 4 * experts + 4 * topk + 4 * topk + 2 * topk * hidden
        want = self.size0 + layers * self.sizel
        if self.raw.size != want:
            raise SystemExit(f"{path}: {self.raw.size} bytes, expected {want}")

    def rec(self, l: int) -> dict:
        off = 0 if l == 0 else self.size0 + (l - 1) * self.sizel
        h, e, k = self.hidden, self.experts, self.topk
        r = {
            "resid": np.frombuffer(self.raw[off : off + 4 * h].tobytes(), np.float32),
        }
        off += 4 * h
        r["x"] = np.frombuffer(self.raw[off : off + 2 * h].tobytes(), np.uint16)
        off += 2 * h
        if l == 0:
            return r
        r["y"] = np.frombuffer(self.raw[off : off + 2 * h].tobytes(), np.uint16)
        off += 2 * h
        r["shared"] = np.frombuffer(self.raw[off : off + 2 * h].tobytes(), np.uint16)
        off += 2 * h
        r["router"] = np.frombuffer(self.raw[off : off + 4 * e].tobytes(), np.float32)
        off += 4 * e
        r["pick"] = np.frombuffer(self.raw[off : off + 4 * k].tobytes(), np.int32)
        off += 4 * k
        r["wts"] = np.frombuffer(self.raw[off : off + 4 * k].tobytes(), np.float32)
        off += 4 * k
        r["ey"] = np.frombuffer(self.raw[off : off + 2 * k * h].tobytes(), np.uint16).reshape(k, h)
        return r


def stage(name: str, layer: int, tensor: str, tf: float, self_: float, extra: dict | None = None) -> dict:
    out = {"stage": name, "layer": layer, "tensor": tensor, "tf": tf, "self": self_, "ratio": tf / max(self_, 1e-12)}
    if extra:
        out.update(extra)
    return out


def trace_row(tf: TfRow, final: np.ndarray, dec: dict, chk: dict, slot: int, dense: set[int], layers: int) -> dict:
    stages = []
    stages.append(stage("embedding", 0, "embedding", maxabs(tf.rec(0)["resid"], dec["embed"][slot]), maxabs(chk["embed"][slot], dec["embed"][slot])))
    for i in range(layers):
        r = tf.rec(i + 1)
        resid_tf = maxabs(r["resid"], dec["resid"][slot, i])
        resid_self = maxabs(chk["resid"][slot, i], dec["resid"][slot, i])
        if i in dense:
            stages.append(stage("dense mlp output", i, "mlp output", maxabs(bits(r["y"]), dec["mlp"][slot, i]), maxabs(chk["mlp"][slot, i], dec["mlp"][slot, i])))
        else:
            attn_tf = maxabs(bits(r["y"]), dec["attn"][slot, i])
            attn_self = maxabs(chk["attn"][slot, i], dec["attn"][slot, i])
            stages.append(stage("attention output", i, "attention output", attn_tf, attn_self))
            router_tf = maxabs(r["router"], dec["router"][slot, i])
            router_self = maxabs(chk["router"][slot, i], dec["router"][slot, i])
            tf_picks = set(int(x) for x in r["pick"])
            oracle_picks = set(int(x) for x in dec["picks"][slot, i])
            same = tf_picks == oracle_picks
            sixth, seventh, gap = margins(dec["router"][slot, i], tf.topk)
            stages.append(
                stage(
                    "router logits",
                    i,
                    "router logits",
                    router_tf,
                    router_self,
                    {"picks_equal": same, "tf_picks": sorted(tf_picks), "oracle_picks": sorted(oracle_picks), "boundary_scores": [sixth, seventh], "boundary_gap": gap},
                )
            )
            if not same:
                stages.append(
                    stage(
                        "selected experts",
                        i,
                        "selected experts",
                        router_tf,
                        router_self,
                        {"tf_picks": sorted(tf_picks), "oracle_picks": sorted(oracle_picks), "boundary_scores": [sixth, seventh], "boundary_gap": gap},
                    )
                )
            chunk_picks = set(int(x) for x in chk["picks"][slot, i])
            common = sorted(tf_picks & oracle_picks & chunk_picks)
            if common:
                expert_tf = max(
                    maxabs(bits(r["ey"][int(np.where(r["pick"] == e)[0][0])]), dec["slots"][slot, i][int(np.where(dec["picks"][slot, i] == e)[0][0])])
                    for e in common
                )
                expert_self = max(
                    maxabs(chk["slots"][slot, i][int(np.where(chk["picks"][slot, i] == e)[0][0])], dec["slots"][slot, i][int(np.where(dec["picks"][slot, i] == e)[0][0])])
                    for e in common
                )
                stages.append(stage("expert output", i, "expert output", expert_tf, expert_self, {"common_picks": len(common)}))
            shared_tf = maxabs(bits(r["shared"]), dec["shared"][slot, i])
            shared_self = maxabs(chk["shared"][slot, i], dec["shared"][slot, i])
            stages.append(stage("shared expert output", i, "shared expert output", shared_tf, shared_self))
        stages.append(stage("residual", i, "residual", resid_tf, resid_self))
    stages.append(stage("final norm", layers, "final norm", maxabs(bits(tf.rec(layers)["x"]), dec["final_norm"][slot]), maxabs(chk["final_norm"][slot], dec["final_norm"][slot])))
    stages.append(stage("logits", layers, "logits", maxabs(bits(final), bits(dec["logits"][slot])), maxabs(bits(chk["logits"][slot]), bits(dec["logits"][slot]))))
    return {"stages": stages}


def verdict(stages: list[dict]) -> tuple[str, str, dict]:
    """The brief's rules: a clean embedding or norm divergence, or a selection flip clear of the router noise,
    is a defect; a flip inside the router logits' own path difference is accumulation order."""
    material = next((x for x in stages if x["tf"] > max(4 * x["self"], 0.01)), None)
    fails = [x for x in stages if x["tf"] > x["self"]]
    first = fails[0] if fails else stages[-1]
    flips = [x for x in stages if x["stage"] == "selected experts"]
    clear = [x for x in flips if x["boundary_gap"] > 2 * max(x["tf"], x["self"])]
    residual_max = max((x for x in stages if x["stage"] == "residual"), key=lambda x: x["ratio"])
    info = {
        "first": first["stage"],
        "first_layer": first["layer"],
        "flips": len(flips),
        "first_flip": None,
        "max_residual": None,
        "clear_flips": [{"layer": x["layer"], "gap": x["boundary_gap"], "router_tf": x["tf"], "router_self": x["self"]} for x in clear],
        "material": None if material is None else {"stage": material["stage"], "layer": material["layer"], "tf": material["tf"], "self": material["self"], "ratio": material["ratio"]},
        "max_residual": {"layer": residual_max["layer"], "tf": residual_max["tf"], "self": residual_max["self"], "ratio": residual_max["ratio"]},
        "embedding": {"tf": stages[0]["tf"], "self": stages[0]["self"]},
        "final_norm": {"tf": stages[-2]["tf"], "self": stages[-2]["self"]},
        "final_residual": {"tf": stages[-3]["tf"], "self": stages[-3]["self"]},
    }
    if flips:
        f = flips[0]
        info["first_flip"] = {
            "layer": f["layer"],
            "gap": f["boundary_gap"],
            "router_tf": f["tf"],
            "router_self": f["self"],
            "gained": sorted(set(f["oracle_picks"]) - set(f["tf_picks"])),
            "lost": sorted(set(f["tf_picks"]) - set(f["oracle_picks"])),
        }
    embed = next(x for x in stages if x["stage"] == "embedding")
    if embed["tf"] > embed["self"]:
        return "defect", f"the embedding differs by {embed['tf']:.4g} where the oracle's own paths differ by {embed['self']:.4g}", info
    if clear:
        x = clear[0]
        return "defect", f"layer {x['layer']} selects a different expert across a boundary logit gap of {x['boundary_gap']:.4g}, clear of the router logits' own path difference of {max(x['tf'], x['self']):.4g}", info
    norm = next(x for x in stages if x["stage"] == "final norm")
    resid = [x for x in stages if x["stage"] == "residual"][-1]
    if first["stage"] == "final norm" and norm["tf"] > norm["self"]:
        return "defect", f"the final norm differs by {norm['tf']:.4g} on a residual that matches ({resid['tf']:.4g} vs {resid['self']:.4g})", info
    if flips:
        last = flips[0]
        return "accumulation order", (
            f"the first departure is {first['tensor']} at layer {first['layer']} ({first['tf']:.4g} vs the oracle's own {first['self']:.4g}); "
            f"the earliest of {len(flips)} expert selection changes is at layer {last['layer']} across a boundary logit gap of "
            f"{last['boundary_gap']:.4g}, not clear of the router logits' own path difference of {max(last['tf'], last['self']):.4g}"
            + ("" if material is None else f"; the state becomes materially different at layer {material['layer']} ({material['tf']:.4g} vs {material['self']:.4g})")
        ), info
    return "accumulation order", f"the first departure is {first['tensor']} at layer {first['layer']} ({first['tf']:.4g} vs the oracle's own {first['self']:.4g}) and no expert selection changes", info


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--logits", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--oracle-script", type=Path, required=True)
    p.add_argument("--heavy", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exact-histories", default="debug-session-nan-loss,json-invoice-records")
    p.add_argument("--compare-only", action="store_true")
    p.add_argument("--keep", type=Path, default=None, help="history -> rows to trace instead of the 2x selection")
    args = p.parse_args()

    corpus = json.loads(args.corpus.read_text())
    config = json.loads((args.model_dir / "config.json").read_text())
    layers = int(config["num_hidden_layers"])
    hidden = int(config["hidden_size"])
    experts = int(config["num_experts"])
    topk = int(config["num_experts_per_tok"])
    dense = set(int(x) for x in config.get("mlp_only_layers", []))
    args.out.mkdir(parents=True, exist_ok=True)
    trace_dir = args.out / "trace"
    trace_dir.mkdir(exist_ok=True)

    rows = {}
    if args.keep is not None:
        rows = {k: sorted(v) for k, v in json.loads(args.keep.read_text()).items() if v}
    for entry in corpus["histories"] if args.keep is None else []:
        name, tokens = entry["name"], entry["tokens"]
        tf = np.fromfile(args.logits / "tf" / f"{name}.bin", np.uint16).reshape(-1, VOCAB)
        ref = np.fromfile(args.logits / "oracle" / f"{name}.bin", np.uint16).reshape(-1, VOCAB)
        other = np.fromfile(args.logits / "oracle-chunked" / f"{name}.bin", np.uint16).reshape(-1, VOCAB)
        a = np.abs(bits(tf) - bits(ref)).max(axis=1)
        s = np.abs(bits(other) - bits(ref)).max(axis=1)
        keep = [r for r in range(tokens) if a[r] > THRESHOLD * s[r] or (name, r) in NAMED]
        if keep:
            rows[name] = keep

    keep_path = trace_dir / "keep.json"
    keep_path.write_text(json.dumps(rows, indent=1) + "\n")

    if not args.compare_only:
        for entry in corpus["histories"]:
            name = entry["name"]
            if name not in rows:
                continue
            spec = trace_dir / f"{name}.rows.json"
            spec.write_text(json.dumps(rows[name]))
            dump_dir = trace_dir / "tf" / name
            dump_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["python3", str(args.heavy), "--", str(args.binary), str(args.model_dir), str(args.logits / "tf" / "ids" / f"{name}.json"), "dump", f"{spec},{dump_dir}"],
                check=True,
            )
        subprocess.run(
            [
                "python3",
                str(args.heavy),
                "--",
                str(args.python),
                str(args.oracle_script),
                str(args.corpus),
                str(args.model_dir),
                str(trace_dir / "oracle"),
                str(keep_path),
                str(args.logits),
            ],
            check=True,
        )
    exact = {}
    for name in [x for x in args.exact_histories.split(",") if x]:
        entry = [h for h in corpus["histories"] if h["name"] == name][0]
        ids = entry["ids"]
        split = min(len(ids) - 1, max(1, len(ids) - 64))
        prompt_path = trace_dir / f"{name}.exact-prompt.json"
        cont_path = trace_dir / f"{name}.exact-cont.json"
        prompt_path.write_text(json.dumps(ids[:split]))
        cont_path.write_text(json.dumps(ids[split:]))
        got = subprocess.run(
            ["python3", str(args.heavy), "--", str(args.binary), str(args.model_dir), str(prompt_path), "exact", str(cont_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        exact[name] = {"prompt": split, "continuation": len(ids) - split, "output": got.stderr.strip()}

    results = []
    for entry in corpus["histories"]:
        name = entry["name"]
        if name not in rows:
            continue
        dec = np.load(trace_dir / "oracle" / f"{name}.decode.npz")
        chk = np.load(trace_dir / "oracle" / f"{name}.chunk.npz")
        for slot, row in enumerate(rows[name]):
            tfrow = TfRow(trace_dir / "tf" / name / f"{row}.layers.bin", layers, hidden, experts, topk)
            final = np.fromfile(trace_dir / "tf" / name / f"{row}.final.bin", np.uint16)
            traced = trace_row(tfrow, final, dec, chk, slot, dense, layers)
            excess = [s for s in traced["stages"] if s["tf"] > s["self"]]
            v, reason, info = verdict(traced["stages"])
            first = excess[0] if excess else traced["stages"][-1]
            tf_max = maxabs(bits(final), bits(dec["logits"][slot]))
            self_max = maxabs(bits(chk["logits"][slot]), bits(dec["logits"][slot]))
            results.append(
                {
                    "history": name,
                    "row": row,
                    "tokens": entry["tokens"],
                    "tf_logit_max_abs": tf_max,
                    "oracle_self_logit_max_abs": self_max,
                    "ratio": tf_max / max(self_max, 1e-12),
                    "first": first,
                    "verdict": v,
                    "reason": reason,
                    "summary": info,
                    "stages_exceeding": excess[:6],
                    "verified": {
                        "tf_logits_equal_compared_bits": bool(np.array_equal(final, np.fromfile(args.logits / "tf" / f"{name}.bin", np.uint16).reshape(-1, VOCAB)[row])),
                        "oracle_decode_equal_compared_bits": bool(dec["logits_verified"][slot]),
                        "oracle_chunk_equal_compared_bits": bool(chk["logits_verified"][slot]),
                    },
                }
            )
    verdicts = {}
    for r in results:
        verdicts.setdefault(r["verdict"], 0)
        verdicts[r["verdict"]] += 1
    out = {
        "threshold": THRESHOLD,
        "rows": len(results),
        "verdicts": verdicts,
        "histories": rows,
        "traced": results,
        "exact": exact,
    }
    (args.out / "trace.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"rows": len(results), "verdicts": verdicts, "exact": exact}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
