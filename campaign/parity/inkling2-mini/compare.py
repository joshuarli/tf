"""Compare the Metal teacher-forced pass with the float32 reference on the parity histories.

Every position's logits are compared: shape, finiteness, absolute and relative error, the KL between the two
next-token distributions, top-1 agreement overall and on positions the reference is confident about (its top-2
margin at least ``--min-margin``), and, for each history whose error crosses ``--tensor-tol``, the first divergent
tensor at the onset position.

The first divergent tensor is found by tracing the onset position's prefix on both sides: the reference's
``--trace-row`` writes one summary line per tensor for that position's row alone in execution order, the runner's
``trace <out> <at>`` folds only that position, and the first overlapping name (the slot-indexed mixin, normin,
attn_out, ffn_inp and l_out tensors, in execution order) whose summary differs by more than ``--tensor-tol`` is the
first tensor to diverge. The onset-1 prefix is traced the same way as the clean control. Both traces run inside the
campaign's memory ceiling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

VOCAB = 200058


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def percentiles(values: np.ndarray) -> str:
    if values.size == 0:
        return "n/a"
    q = np.percentile(values, [50, 90, 99, 100])
    return f"p50 {q[0]:.3e}, p90 {q[1]:.3e}, p99 {q[2]:.3e}, max {q[3]:.3e}"


OVERLAP_PREFIXES = ("mixin-", "normin-", "attn_out-", "ffn_inp-", "l_out-")


def read_trace(path: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 5:
            out[fields[0]] = [float(x) for x in fields[1:5]]
    return out


def fmt4(v: list[float]) -> str:
    return "(mean {:.6g}, std {:.6g}, min {:.6g}, max {:.6g})".format(*v)


def first_divergence(ref_path: Path, metal_path: Path, pos: int, tol: float) -> str:
    if not ref_path.exists() or not metal_path.exists():
        missing = ref_path if not ref_path.exists() else metal_path
        return f"no trace at {missing}"
    mine = read_trace(metal_path)
    suffix = f"-row{pos}"
    checked = 0
    for line in ref_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 5 or not fields[0].endswith(suffix):
            continue
        base = fields[0][: -len(suffix)]
        if not base.startswith(OVERLAP_PREFIXES):
            continue
        want = [float(x) for x in fields[1:5]]
        checked += 1
        got = mine.get(base)
        if got is None:
            return f"{base}: absent from the Metal trace"
        delta = max(abs(a - b) for a, b in zip(got, want))
        if delta > tol:
            return (f"{base}: reference {fmt4(want)} against Metal {fmt4(got)}, "
                    f"largest difference {delta:.3e} over {checked} overlapping tensors")
    if checked == 0:
        return f"no position-{pos} rows for the overlapping tensors in {ref_path}"
    return f"every one of {checked} overlapping tensors agrees within {tol:g}"


def trace_prefix(name: str, tokens: list[int], pos: int, args) -> tuple[Path, Path]:
    """Write both sides' traces for the prefix ending at one position, each inside the memory ceiling."""
    metal_dir, ref_dir = Path(args.metal_traces), Path(args.ref_traces)
    metal_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ids = metal_dir / f"{name}.p{pos}.tokens.json"
    ids.write_text(json.dumps(tokens[:pos + 1]))
    metal = metal_dir / f"{name}.p{pos}.trace"
    if not metal.exists():
        done = subprocess.run(["python3", args.heavy, "--", args.runner, args.model, str(ids),
                               "trace", str(metal), str(pos)], capture_output=True, text=True)
        if done.returncode != 0:
            return Path(f"the runner's trace failed: {done.stderr.strip()[-200:]}"), metal
    ref = ref_dir / f"{name}.p{pos}.trace"
    if not ref.exists():
        venv = Path(args.ref_venv) / "bin" / "python"
        cmd = ["python3", args.heavy, "--", str(venv), args.reference, "--model", args.model,
               "--tokens", " ".join(str(t) for t in tokens[:pos + 1]),
               "--out", str(ref_dir / f"{name}.p{pos}.npy"),
               "--trace", str(ref), "--trace-row", str(pos),
               "--router-report", str(ref_dir / f"{name}.p{pos}.routers.json"),
               "--expert-cache-gib", "10"]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            return metal, Path(f"the reference's trace failed: {done.stderr.strip()[-200:]}")
    return ref, metal


def tightest_routers(ref_dir: Path, name: str, pos: int, count: int = 3) -> str:
    path = ref_dir / f"{name}.p{pos}.routers.json"
    if not path.exists():
        return f"no router report at {path}"
    rows = json.loads(path.read_text())
    ranked = sorted(((abs(v["gap_8_9"]), slot) for slot, v in rows.items()))
    bits = []
    for gap, slot in ranked[:count]:
        v = rows[slot]
        bits.append(f"slot {slot} (layer {v['layer']}): picks {v['picks']} against 9th {v['ninth']}, "
                    f"8th/9th selection gap {v['gap_8_9']:.3g} (prob gap {v['gap_8_9_prob']:.3g})")
    return "; ".join(bits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--metal-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--agreement", type=float, default=99.5)
    parser.add_argument("--tensor-tol", type=float, default=1e-3)
    parser.add_argument("--runner")
    parser.add_argument("--model")
    parser.add_argument("--heavy")
    parser.add_argument("--ref-venv")
    parser.add_argument("--reference")
    parser.add_argument("--metal-traces", default="")
    parser.add_argument("--ref-traces", default="")
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    ref_dir, metal_dir = Path(args.ref_dir), Path(args.metal_dir)
    report = Path(args.report)
    args.metal_traces = args.metal_traces or str(metal_dir / "trace")
    args.ref_traces = args.ref_traces or str(ref_dir.parent / "trace-ref")
    can_trace = all([args.runner, args.model, args.heavy, args.ref_venv, args.reference,
                     Path(args.runner).exists(), Path(args.model).exists(), Path(args.heavy).exists()])

    total = confident_total = agree_total = agree_confident = 0
    kls: list[float] = []
    abs_all: list[np.ndarray] = []
    rel_all: list[np.ndarray] = []
    rows: list[str] = []
    divergences: list[str] = []
    problems: list[str] = []
    long_histories = 0

    for entry in histories:
        name, tokens = entry["name"], entry["token_ids"]
        if len(tokens) > 1024:
            long_histories += 1
        ref_path, got_path = ref_dir / f"{name}.npy", metal_dir / f"{name}.bin"
        if not ref_path.exists() or not got_path.exists():
            problems.append(f"{name}: missing {'reference' if not ref_path.exists() else 'Metal'} logits")
            continue
        reference = np.load(ref_path)
        mine = np.fromfile(got_path, dtype=np.float32)
        if mine.size % VOCAB != 0:
            problems.append(f"{name}: Metal logits are {mine.size} values, not whole rows of {VOCAB}")
            continue
        mine = mine.reshape(-1, VOCAB)
        if mine.shape != reference.shape:
            problems.append(f"{name}: Metal {mine.shape} against reference {reference.shape}")
            continue
        if not np.isfinite(mine).all():
            problems.append(f"{name}: Metal logits are not all finite")
            continue
        if not np.isfinite(reference).all():
            problems.append(f"{name}: reference logits are not all finite")
            continue

        err = np.abs(mine.astype(np.float64) - reference.astype(np.float64))
        scale = np.maximum(np.abs(reference).max(axis=1, keepdims=True), 1e-12)
        rel = err / scale
        kl = (np.exp(log_softmax(reference)) * (log_softmax(reference) - log_softmax(mine))).sum(axis=1)
        sorted_ref = np.sort(reference, axis=1)
        margin = sorted_ref[:, -1] - sorted_ref[:, -2]
        confident = margin >= args.min_margin
        agree = mine.argmax(axis=1) == reference.argmax(axis=1)

        n = reference.shape[0]
        total += n
        confident_total += int(confident.sum())
        agree_total += int(agree.sum())
        agree_confident += int((agree & confident).sum())
        kls.append(float(kl.mean()))
        abs_all.append(err.max(axis=1))
        rel_all.append(rel.max(axis=1))
        conf_pct = agree[confident].mean() * 100 if confident.any() else float("nan")
        rows.append(f"| {name} | {len(tokens)} | {err.max():.3e} | {kl.mean():.3e} "
                    f"| {agree.mean() * 100:.2f}% | {conf_pct:.2f}% | {int((~agree & confident).sum())} |")

        disagreed = np.nonzero(~agree)[0]
        for i in disagreed:
            divergences.append(
                f"\n### {name} position {int(i)}: reference top-1 {int(reference[i].argmax())}, "
                f"Metal top-1 {int(mine[i].argmax())}\n"
                f"- reference top-2 margin {margin[i]:.4g}, largest logit error {err[i].max():.4g}, "
                f"KL {kl[i]:.4g}")
        over = np.nonzero(err.max(axis=1) > args.tensor_tol)[0]
        if over.size:
            onset = int(over[0])
            if not can_trace:
                divergences.append(
                    f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}\n"
                    "- first divergent tensor: not traced: no runner, model or memory ceiling was available")
            else:
                ref_tr, metal_tr = trace_prefix(name, tokens, onset, args)
                if ref_tr.exists() and metal_tr.exists():
                    where = first_divergence(ref_tr, metal_tr, onset, args.tensor_tol)
                    ctrl = "no earlier position"
                    if onset > 0:
                        ref_c, metal_c = trace_prefix(name, tokens, onset - 1, args)
                        if ref_c.exists() and metal_c.exists():
                            ctrl = first_divergence(ref_c, metal_c, onset - 1, args.tensor_tol)
                        else:
                            ctrl = str(ref_c) if not ref_c.exists() else str(metal_c)
                    routers = tightest_routers(Path(args.ref_traces), name, onset)
                    divergences.append(
                        f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}, "
                        f"KL {kl[onset]:.4g}\n"
                        f"- first divergent tensor: {where}\n"
                        f"- clean control at {onset - 1}: {ctrl}\n"
                        f"- tightest reference router 8th/9th gaps at {onset}: {routers}")
                else:
                    bad = str(ref_tr) if not ref_tr.exists() else str(metal_tr)
                    divergences.append(
                        f"\n### {name} onset position {onset}: largest logit error {err[onset].max():.4g}\n"
                        f"- first divergent tensor: {bad}")

    abs_flat = np.concatenate(abs_all) if abs_all else np.array([])
    rel_flat = np.concatenate(rel_all) if rel_all else np.array([])
    overall = 100.0 * agree_total / total if total else 0.0
    confident_pct = 100.0 * agree_confident / confident_total if confident_total else float("nan")

    lines = [
        "# Inkling 2 Mini parity: Metal against the float32 reference",
        "",
        "Arithmetic: NVFP4 and MXFP8 weights decoded from the checkpoint's own bytes, float32 activations. This is",
        "not the official activation-quantized path and no equivalence with it is claimed.",
        "",
        "## Summary",
        "",
        f"- positions compared: {total} over {len(histories)} histories "
        f"({long_histories} past the 1,024-token sliding window)",
        f"- mean KL: {float(np.mean(kls)):.6g}" if kls else "- mean KL: n/a",
        f"- top-1 agreement overall: {overall:.2f}%",
        f"- top-1 agreement on positions with reference top-2 margin >= {args.min_margin}: "
        f"{confident_pct:.2f}% over {confident_total} positions (required >= {args.agreement}%)",
        f"- absolute logit error: {percentiles(abs_flat)}",
        f"- relative logit error: {percentiles(rel_flat)}",
        "",
        "## Per history",
        "",
        "| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident disagreements |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Divergences",
    ]
    lines += divergences if divergences else ["", "Every position agrees on top-1."]
    report.write_text("\n".join(lines) + "\n")

    for line in lines[:16]:
        print(line)
    print(f"report: {report}")
    for p in problems:
        print(f"problem: {p}", file=sys.stderr)
    if long_histories < 2:
        print(f"problem: {long_histories} histories run past the sliding window, at least two must", file=sys.stderr)
        problems.append("short corpus")
    failed = bool(problems) or not (confident_pct >= args.agreement) or total == 0
    if failed:
        print("parity: not met", file=sys.stderr)
        return 1
    print("parity: met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
