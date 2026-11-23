"""Streamed teacher-forced logit comparison, and the gate that holds TensorFold to the oMLX oracle.

Both subcommands map one history at a time and read `--block` rows at a time, so no capture is ever held in
memory whole. A capture is `<dir>/<name>.npy`, float32 [tokens, vocab] like the oMLX captures, or
`<dir>/<name>.bin`, raw float32 rows like the runner's. For every row the comparison takes both rows' maximal
tokens, the reference's top-2 margin, and KL(reference || candidate) in nats from float64 log-softmax. Two rows
agree on top-1 when their sets of maximal tokens intersect: the oMLX logits are bfloat16 before the softcap, so
1.7% of the oracle's rows hold an exact tie for the maximum, which argmax would break by token index.

    compare   any two capture directories: prints the metrics, and records them with --json and --report.
    gate      TensorFold's runner captures against the oMLX oracle. The oracle must be the complete capture
              set that the gate file pins (manifest digest, histories and every file's sha256). Every candidate
              must carry a receipt naming the current runner, checkpoint, tokens and arithmetic policy. The
              pooled results must meet the thresholds derived from the gate file's recorded baseline.

Pooled statistics: top-1 agreement over every position; mean KL as the mean of the per-history means, the
campaign's convention, and the row-weighted mean; max KL over every position. The gate bounds top-1 and both KL
means. Max KL is reported, not gated: across the engine pairings in campaign/kernels/inkling-omlx-oracle.md it
ranges from 8 to 17 nats, each time at a different position. A non-finite logit on either side fails the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import provenance  # noqa: E402

DEFAULT_VOCAB = 200058
DEFAULT_MARGIN = 0.1
DEFAULT_BLOCK = 32
ANY_FORMAT = (".npy", ".bin")


def open_capture(folder: Path, name: str, vocab: int,
                 formats: tuple[str, ...] = ANY_FORMAT) -> tuple[np.ndarray | None, str | None]:
    """One history's capture memory-mapped as float32 [rows, vocab], or why it cannot be read. ``formats`` lists
    the accepted suffixes in order of preference."""
    npy, raw = folder / f"{name}.npy", folder / f"{name}.bin"
    if ".npy" in formats and npy.is_file():
        try:
            data = np.load(npy, mmap_mode="r")
        except (OSError, ValueError) as exc:
            return None, f"{npy}: unreadable: {exc}"
        if data.dtype != np.float32 or data.ndim != 2 or data.shape[1] != vocab:
            return None, f"{npy}: holds {data.dtype} {data.shape}, not float32 [rows, {vocab}]"
        return data, None
    if ".bin" in formats and raw.is_file():
        size = raw.stat().st_size
        if size == 0 or size % (4 * vocab):
            return None, f"{raw}: {size} bytes is not whole float32 rows of {vocab}"
        return np.memmap(raw, dtype=np.float32, mode="r", shape=(size // (4 * vocab), vocab)), None
    return None, f"no capture of {name} in {folder}"


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def history_metrics(ref: np.ndarray, cand: np.ndarray, margin: float, block: int) -> dict:
    """Agreement and divergence of two [rows, vocab] captures of the same history."""
    rows = ref.shape[0]
    out = {"rows": rows, "top1": 0, "confident": 0, "confident_top1": 0, "reference_ties": 0, "kl_sum": 0.0,
           "max_kl": 0.0, "max_kl_position": 0, "max_abs": 0.0, "nonfinite_reference": 0,
           "nonfinite_candidate": 0, "nonfinite_kl_rows": 0, "disagreements": []}
    for start in range(0, rows, block):
        a = np.asarray(ref[start:start + block], dtype=np.float64)
        b = np.asarray(cand[start:start + block], dtype=np.float64)
        out["nonfinite_reference"] += int(np.count_nonzero(~np.isfinite(a)))
        out["nonfinite_candidate"] += int(np.count_nonzero(~np.isfinite(b)))
        top_a = a == a.max(axis=1, keepdims=True)
        hit = np.any(top_a & (b == b.max(axis=1, keepdims=True)), axis=1)
        out["reference_ties"] += int(np.count_nonzero(top_a.sum(axis=1) > 1))
        top2 = np.partition(a, -2, axis=1)[:, -2:]
        sure = top2[:, 1] - top2[:, 0] >= margin
        out["top1"] += int(hit.sum())
        out["confident"] += int(sure.sum())
        out["confident_top1"] += int((hit & sure).sum())
        out["disagreements"] += [start + int(i) for i in np.nonzero(~hit)[0]]
        out["max_abs"] = max(out["max_abs"], float(np.nanmax(np.abs(a - b))))
        la, lb = log_softmax(a), log_softmax(b)
        kl = (np.exp(la) * (la - lb)).sum(axis=1)
        finite = np.isfinite(kl)
        out["nonfinite_kl_rows"] += int(np.count_nonzero(~finite))
        out["kl_sum"] += float(kl[finite].sum())
        if finite.any():
            worst = int(np.where(finite, kl, -np.inf).argmax())
            if kl[worst] > out["max_kl"]:
                out["max_kl"], out["max_kl_position"] = float(kl[worst]), start + worst
    out["mean_kl"] = out["kl_sum"] / rows
    return out


def compare_dirs(histories: list[dict], ref_dir: Path, cand_dir: Path, vocab: int, margin: float, block: int,
                 ref_formats: tuple[str, ...] = ANY_FORMAT,
                 cand_formats: tuple[str, ...] = ANY_FORMAT) -> tuple[list[dict], list[str]]:
    results, problems = [], []
    for entry in histories:
        name, expected = entry["name"], len(entry["token_ids"])
        ref, why_ref = open_capture(ref_dir, name, vocab, ref_formats)
        cand, why_cand = open_capture(cand_dir, name, vocab, cand_formats)
        bad = [why for why in (why_ref, why_cand) if why]
        bad += [f"{name}: {side} holds {data.shape[0]} rows, histories.json names {expected} tokens"
                for side, data in (("reference", ref), ("candidate", cand))
                if data is not None and data.shape[0] != expected]
        if bad:
            problems += bad
            continue
        results.append({"name": name} | history_metrics(ref, cand, margin, block))
    return results, problems


def summarize(results: list[dict]) -> dict:
    if not results:
        return {"histories": 0, "positions": 0}
    positions = sum(r["rows"] for r in results)
    confident = sum(r["confident"] for r in results)
    worst = max(results, key=lambda r: r["max_kl"])
    summary = {
        "histories": len(results),
        "positions": positions,
        "top1": sum(r["top1"] for r in results),
        "confident": confident,
        "confident_top1": sum(r["confident_top1"] for r in results),
        "reference_ties": sum(r["reference_ties"] for r in results),
        "mean_kl": float(np.mean([r["mean_kl"] for r in results])),
        "row_mean_kl": sum(r["kl_sum"] for r in results) / positions,
        "max_kl": worst["max_kl"],
        "max_kl_at": {"history": worst["name"], "position": worst["max_kl_position"]},
        "max_abs": max(r["max_abs"] for r in results),
        "nonfinite_reference": sum(r["nonfinite_reference"] for r in results),
        "nonfinite_candidate": sum(r["nonfinite_candidate"] for r in results),
        "nonfinite_kl_rows": sum(r["nonfinite_kl_rows"] for r in results),
    }
    summary["top1_percent"] = 100.0 * summary["top1"] / positions
    summary["confident_top1_percent"] = (100.0 * summary["confident_top1"] / confident) if confident else None
    return summary


def thresholds(gate: dict) -> dict:
    """The pass bounds: the recorded baseline less the gate file's margins. Both KL means are bounded: the mean
    of history means weights the short histories, the row-weighted mean the two past the sliding window."""
    base, margin = gate["baseline"], gate["margin"]
    return {"min_top1_percent": 100.0 * base["top1"] / base["positions"] - margin["top1_points"],
            "max_mean_kl": base["mean_kl"] * margin["mean_kl_factor"],
            "max_row_mean_kl": base["row_mean_kl"] * margin["mean_kl_factor"],
            "positions": base["positions"]}


def judge(summary: dict, bounds: dict) -> list[str]:
    """Threshold failures of a pooled summary; empty when it passes."""
    if summary.get("positions") != bounds["positions"]:
        return [f"compared {summary.get('positions', 0)} positions, the baseline covers {bounds['positions']}"]
    failures = []
    if summary["nonfinite_reference"] or summary["nonfinite_candidate"]:
        failures.append(f"non-finite logits: {summary['nonfinite_reference']} in the oracle, "
                        f"{summary['nonfinite_candidate']} in the candidate")
    if summary["top1_percent"] < bounds["min_top1_percent"]:
        failures.append(f"top-1 agreement {summary['top1_percent']:.3f}% is below {bounds['min_top1_percent']:.3f}%")
    if not summary["mean_kl"] <= bounds["max_mean_kl"]:
        failures.append(f"mean KL {summary['mean_kl']:.6g} is above {bounds['max_mean_kl']:.6g}")
    if not summary["row_mean_kl"] <= bounds["max_row_mean_kl"]:
        failures.append(f"row-weighted mean KL {summary['row_mean_kl']:.6g} is above {bounds['max_row_mean_kl']:.6g}")
    return failures


def verify_oracle(folder: Path, histories: list[dict], histories_path: Path, gate: dict, vocab: int) -> list[str]:
    """Problems with the oracle captures: they must be the complete set the gate file pins, byte for byte."""
    path = folder / "manifest.json"
    if not path.is_file():
        return [f"no oracle manifest at {path}; capture the oracle with `make -C campaign omlx-oracle-capture`"]
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{path}: unreadable manifest: {exc}"]
    outputs = manifest.get("outputs", {})
    problems = []
    if manifest.get("complete") is not True:
        problems.append(f"{path}: the oracle capture is incomplete")
    if manifest.get("histories", {}).get("sha256") != provenance.sha256_file(histories_path):
        problems.append(f"{path}: the oracle was captured from other histories than {histories_path}")
    if manifest.get("layout", {}).get("vocab") != vocab:
        problems.append(f"{path}: the oracle's vocabulary is not {vocab}")
    digest = provenance.capture_digest({n: o.get("output_sha256") for n, o in outputs.items()})
    if manifest.get("digest") != digest:
        problems.append(f"{path}: the manifest digest does not name its own outputs")
    if digest != gate["oracle"]["digest"]:
        problems.append(f"oracle digest {digest[:16]} is not the {gate['oracle']['digest'][:16]} the gate's "
                        "baseline was measured against; re-measure the baseline before gating on this oracle")
    for entry in histories:
        name, tokens = entry["name"], entry["token_ids"]
        held, npy = outputs.get(name), folder / f"{name}.npy"
        if held is None or not npy.is_file():
            problems.append(f"{name}: no oracle capture")
            continue
        if held.get("token_sha256") != provenance.token_history_sha(tokens) or held.get("rows") != len(tokens):
            problems.append(f"{name}: the oracle capture names other tokens")
        elif provenance.sha256_file(npy) != held.get("output_sha256"):
            problems.append(f"{name}: {npy} is not the file the manifest records")
    return problems


def verify_candidates(folder: Path, histories: list[dict], runner_sha256: str, model_identity: str,
                      policy: str, vocab: int) -> list[str]:
    problems = []
    for entry in histories:
        tokens = entry["token_ids"]
        problems += provenance.check_receipt(
            folder / f"{entry['name']}.bin", runner_sha256=runner_sha256, model_identity_digest=model_identity,
            token_sha256=provenance.token_history_sha(tokens), policy=policy, vocab=vocab, rows=len(tokens))
    return problems


def fmt_kl(value: float) -> str:
    return f"{value:.4g}"


def report_lines(title: str, inputs: dict, results: list[dict], summary: dict, margin: float) -> list[str]:
    lines = [f"# {title}", ""]
    lines += [f"- {key}: {value}" for key, value in inputs.items()]
    if summary.get("positions"):
        s = summary
        confident = (f"{s['confident_top1']}/{s['confident']} ({s['confident_top1_percent']:.2f}%)"
                     if s["confident"] else "none")
        lines += [
            f"- positions: {s['positions']} over {s['histories']} histories",
            f"- top-1 agreement: {s['top1']}/{s['positions']} ({s['top1_percent']:.2f}%); "
            f"{s['reference_ties']} reference rows tie for the maximum",
            f"- top-1 agreement where the reference's top-2 margin is at least {margin:g}: {confident}",
            f"- mean KL, mean of history means: {s['mean_kl']:.6g}; row-weighted: {s['row_mean_kl']:.6g}",
            f"- max KL: {s['max_kl']:.6g} at {s['max_kl_at']['history']} position {s['max_kl_at']['position']}",
            f"- max absolute logit difference: {s['max_abs']:.4g}",
            f"- non-finite logits: reference {s['nonfinite_reference']}, candidate {s['nonfinite_candidate']}",
            "",
            "| history | rows | top-1 | top-1 % | confident top-1 % | mean KL | max KL | max KL position | max abs |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in results:
            sure = f"{100.0 * r['confident_top1'] / r['confident']:.2f}" if r["confident"] else "n/a"
            lines.append(f"| {r['name']} | {r['rows']} | {r['top1']} | {100.0 * r['top1'] / r['rows']:.2f} "
                         f"| {sure} | {fmt_kl(r['mean_kl'])} | {fmt_kl(r['max_kl'])} | {r['max_kl_position']} "
                         f"| {r['max_abs']:.3g} |")
        by_top1 = sorted(results, key=lambda r: r["top1"] / r["rows"])[:3]
        by_kl = sorted(results, key=lambda r: -r["mean_kl"])[:3]
        lines += ["",
                  "Lowest top-1: " + ", ".join(f"{r['name']} ({100.0 * r['top1'] / r['rows']:.2f}%)"
                                                for r in by_top1) + ".",
                  "Highest mean KL: " + ", ".join(f"{r['name']} ({fmt_kl(r['mean_kl'])})" for r in by_kl) + "."]
    return lines


def record(args, payload: dict, lines: list[str]) -> None:
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def public(results: list[dict]) -> list[dict]:
    """Per-history results for the record: the first disagreeing positions, not every one."""
    return [{k: v for k, v in r.items() if k != "disagreements"} | {"first_disagreements": r["disagreements"][:16]}
            for r in results]


def run_compare(args) -> int:
    histories = json.loads(Path(args.histories).read_text())
    began = time.monotonic()
    results, problems = compare_dirs(histories, Path(args.ref), Path(args.cand), args.vocab, args.margin, args.block)
    summary = summarize(results)
    inputs = {"reference": args.ref, "candidate": args.cand}
    lines = report_lines(args.title, inputs, results, summary, args.margin)
    lines += [""] + [f"problem: {p}" for p in problems]
    record(args, {"inputs": inputs, "summary": summary, "histories": public(results), "problems": problems,
                  "seconds": round(time.monotonic() - began, 1)}, lines)
    return 1 if problems else 0


def run_gate(args) -> int:
    histories_path = Path(args.histories)
    histories = json.loads(histories_path.read_text())
    gate = json.loads(Path(args.gate).read_text())
    oracle, cand = Path(args.oracle), Path(args.cand)
    began = time.monotonic()
    policy = gate["candidate"]["arithmetic_policy"]
    problems = []
    if args.runner and not Path(args.runner).is_file():
        problems.append(f"no runner at {args.runner}")
    runner_sha256 = args.runner_sha256 or (provenance.sha256_file(Path(args.runner)) if not problems else "")
    model_identity = args.model_identity or provenance.model_identity(Path(args.model))
    problems += verify_oracle(oracle, histories, histories_path, gate, args.vocab)
    problems += verify_candidates(cand, histories, runner_sha256, model_identity, policy, args.vocab)
    results, unreadable = compare_dirs(histories, oracle, cand, args.vocab, args.margin, args.block,
                                       ref_formats=(".npy",), cand_formats=(".bin",))
    problems += unreadable
    summary = summarize(results)
    bounds = thresholds(gate)
    failures = problems + judge(summary, bounds)
    verdict = "pass" if not failures else "FAIL"
    inputs = {"oracle": f"{oracle} (digest {gate['oracle']['digest'][:16]}, {gate['oracle']['engine']})",
              "candidate": str(cand), "runner sha256": runner_sha256, "model identity": model_identity,
              "arithmetic policy": policy}
    lines = report_lines("Inkling 2 Mini: TensorFold against the oMLX oracle", inputs, results, summary, args.margin)
    lines += ["",
              f"Baseline {gate['baseline']['revision']}: top-1 {gate['baseline']['top1']}/"
              f"{gate['baseline']['positions']}, mean KL {gate['baseline']['mean_kl']:.6g}, "
              f"row-weighted {gate['baseline']['row_mean_kl']:.6g}, max KL {gate['baseline']['max_kl']:.6g} "
              "(reported, not gated).",
              f"Thresholds: top-1 at least {bounds['min_top1_percent']:.3f}%, mean KL at most "
              f"{bounds['max_mean_kl']:.6g}, row-weighted mean KL at most {bounds['max_row_mean_kl']:.6g}, "
              "no non-finite logits.",
              "", f"omlx-oracle gate: {verdict}"]
    lines += [f"- {f}" for f in failures]
    record(args, {"verdict": verdict, "failures": failures, "inputs": inputs, "thresholds": bounds,
                  "baseline": gate["baseline"], "summary": summary, "histories": public(results),
                  "seconds": round(time.monotonic() - began, 1)}, lines)
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("compare", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--histories", default=str(HERE / "histories.json"))
        p.add_argument("--cand", required=True, help="candidate capture directory")
        p.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
        p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                       help="reference top-2 logit margin of a confident position")
        p.add_argument("--block", type=int, default=DEFAULT_BLOCK, help="rows read at a time")
        p.add_argument("--json", default="")
        p.add_argument("--report", default="")
    compare = sub.choices["compare"]
    compare.add_argument("--ref", required=True, help="reference capture directory")
    compare.add_argument("--title", default="Inkling 2 Mini teacher-forced logits")
    gate = sub.choices["gate"]
    gate.add_argument("--oracle", required=True, help="the oMLX oracle capture directory")
    gate.add_argument("--gate", default=str(HERE / "omlx_gate.json"))
    who = gate.add_mutually_exclusive_group(required=True)
    who.add_argument("--runner")
    who.add_argument("--runner-sha256")
    what = gate.add_mutually_exclusive_group(required=True)
    what.add_argument("--model")
    what.add_argument("--model-identity")
    args = parser.parse_args(argv)
    if args.block < 1:
        parser.error("--block must be positive")
    return run_compare(args) if args.action == "compare" else run_gate(args)


if __name__ == "__main__":
    sys.exit(main())
