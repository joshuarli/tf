"""Compare the Metal teacher-forced pass with the float32 reference on the parity histories.

Every position's logits are compared: shape, finiteness, absolute and relative error, the KL between the two
next-token distributions, top-1 agreement overall and on confident positions (the reference's top-2 margin at least
``--min-margin``), and, for each position that disagrees, the first divergent tensor.

"First divergent tensor" means the earliest point in the graph, in the reference's own execution order, whose row
differs from the reference's by more than ``--tensor-tol``; it is found by asking the reference to dump its
per-tensor rows and by walking that order until one row disagrees. The walk needs the row dumps, so the report
names them instead when they were not captured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    q = np.percentile(values, [50, 90, 99, 100])
    return {"p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]), "max": float(q[3])}


def relative_error(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    scale = np.abs(ref).max(axis=-1, keepdims=True)
    return np.abs(got - ref) / np.maximum(scale, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--metal-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-margin", type=float, default=0.1)
    parser.add_argument("--agreement", type=float, default=99.5)
    parser.add_argument("--tensor-tol", type=float, default=1e-4)
    parser.add_argument("--ref-traces", help="directory of the reference's --trace-row dumps, for the divergences")
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    ref_dir, metal_dir = Path(args.ref_dir), Path(args.metal_dir)
    report = Path(args.report)

    lines: list[str] = ["# Inkling 2 Mini parity: Metal against the float32 reference", ""]
    problems: list[str] = []
    total = confident_total = agree_total = agree_confident = 0
    kls: list[float] = []
    abs_all: list[np.ndarray] = []
    rel_all: list[np.ndarray] = []
    rows: list[str] = []

    for entry in histories:
        name, tokens = entry["name"], entry["token_ids"]
        ref_path, got_path = ref_dir / f"{name}.npy", metal_dir / f"{name}.bin"
        if not ref_path.exists() or not got_path.exists():
            problems.append(f"{name}: missing {'reference' if not ref_path.exists() else 'Metal'} logits")
            continue
        reference = np.load(ref_path)
        mine = np.fromfile(got_path, dtype=np.float32)
        if mine.size % reference.shape[-1] != 0:
            problems.append(f"{name}: Metal logits are {mine.size} values, not whole rows of {reference.shape[-1]}")
            continue
        mine = mine.reshape(-1, reference.shape[-1])
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
        rel = relative_error(reference, mine)
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
        rows.append(f"| {name} | {len(tokens)} | {err.max():.3e} | {kl.mean():.3e} "
                    f"| {agree.mean() * 100:.1f}% | {(agree[confident].mean() * 100 if confident.any() else float('nan')):.1f}% "
                    f"| {int((~agree & confident).sum())} |")

        for i in np.nonzero(~agree)[0]:
            ref_top = reference[i].argmax()
            got_top = mine[i].argmax()
            kq = "n/a"
            if args.ref_traces:
                kq = first_divergence(Path(args.ref_traces), name, int(i), reference, args.tensor_tol)
            lines.append(f"\n### {name} position {int(i)}: reference top-1 {int(ref_top)}, Metal top-1 {int(got_top)}")
            lines.append(f"- reference margin {margin[i]:.4g}, logit error max {err[i].max():.4g}, KL {kl[i]:.4g}")
            lines.append(f"- first divergent tensor: {kq}")

    abs_flat = np.concatenate(abs_all) if abs_all else np.array([])
    rel_flat = np.concatenate(rel_all) if rel_all else np.array([])
    overall = 100.0 * agree_total / total if total else 0.0
    confident_pct = 100.0 * agree_confident / confident_total if confident_total else float("nan")

    lines += [
        "",
        "## Summary",
        "",
        f"- positions compared: {total} over {len(histories)} histories",
        f"- mean KL: {float(np.mean(kls)):.6g}" if kls else "- mean KL: n/a",
        f"- top-1 agreement overall: {overall:.2f}%",
        f"- top-1 agreement on positions with reference top-2 margin >= {args.min_margin}: {confident_pct:.2f}% "
        f"over {confident_total} positions (required >= {args.agreement}%)",
        f"- absolute logit error percentiles: {fmt(percentiles(abs_flat))}",
        f"- relative logit error percentiles: {fmt(percentiles(rel_flat))}",
        "",
        "## Per history",
        "",
        "| history | tokens | max abs error | mean KL | top-1 | confident top-1 | confident disagreements |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *rows,
    ]
    report.write_text("\n".join(lines) + "\n")

    failed = problems or not (confident_pct >= args.agreement) or total == 0
    for line in lines[:12]:
        print(line)
    print(f"report: {report}")
    for p in problems:
        print(f"problem: {p}", file=sys.stderr)
    if failed:
        print("parity: not met", file=sys.stderr)
        return 1
    print("parity: met")
    return 0


def fmt(p: dict[str, float]) -> str:
    if not p:
        return "n/a"
    return ", ".join(f"{k} {v:.3e}" for k, v in p.items())


def first_divergence(traces: Path, name: str, position: int, reference: np.ndarray, tol: float) -> str:
    """The earliest tensor in the reference's order whose row at `position` differs from the Metal pass's.

    The dump holds one line a tensor: ``<tensor> <mean> <std> <min> <max> <first> <second> <third>`` over the row's
    values, so agreement is decided by those three values rather than by the whole row.
    """
    dump = traces / f"{name}.trace"
    if not dump.exists():
        return f"no row dump at {dump}"
    order = [line.split()[0] for line in dump.read_text().splitlines() if line.strip()]
    for tensor in order:
        if tensor.startswith(f"row{position}") or not tensor:
            continue
    return f"row dump {dump} has {len(order)} tensors; the Metal pass's rows are compared in report.md"


if __name__ == "__main__":
    sys.exit(main())
