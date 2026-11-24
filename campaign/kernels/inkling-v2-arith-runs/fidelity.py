"""Decode-path fidelity (V2): teacher-forced decode captures against a reference and against each other.

usage: fidelity.py --histories H.json --ref REF_DIR [--cand NAME=DIR ...] [--same A=B ...] [--json OUT]

Every capture is `<dir>/<name>.npy` or `<dir>/<name>.bin`, f32 [tokens, vocab], as omlx_gate.py reads them. For each
candidate against the reference it reports, pooled over the histories: top-1 agreement (the two rows' sets of maximal
tokens intersect), the mean of per-history mean KL(reference || candidate) in nats and the row-weighted mean, the
largest KL, and the distribution of each row's largest |logit difference|. `--same A=B` counts the rows whose bytes
are equal in two candidates.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "parity" / "inkling2-mini"))

from omlx_gate import log_softmax, open_capture  # noqa: E402

VOCAB = 200058


def against(histories: list[dict], ref_dir: Path, cand_dir: Path) -> dict:
    top1 = rows = 0
    kl_means, kl_rows, max_abs, max_kl = [], [], [], 0.0
    for h in histories:
        ref, why = open_capture(ref_dir, h["name"], VOCAB)
        cand, why_c = open_capture(cand_dir, h["name"], VOCAB)
        if ref is None or cand is None:
            raise SystemExit(why or why_c)
        if ref.shape != cand.shape:
            raise SystemExit(f"{h['name']}: reference {ref.shape} against candidate {cand.shape}")
        kls = []
        for start in range(0, ref.shape[0], 32):
            a = np.asarray(ref[start:start + 32], dtype=np.float64)
            b = np.asarray(cand[start:start + 32], dtype=np.float64)
            top1 += int(np.any((a == a.max(axis=1, keepdims=True)) & (b == b.max(axis=1, keepdims=True)), axis=1).sum())
            max_abs += np.abs(a - b).max(axis=1).tolist()
            la, lb = log_softmax(a), log_softmax(b)
            kls += (np.exp(la) * (la - lb)).sum(axis=1).tolist()
        rows += ref.shape[0]
        kl_means.append(float(np.mean(kls)))
        kl_rows += kls
        max_kl = max(max_kl, max(kls))
    q = np.quantile(np.array(max_abs), [0.5, 0.9, 0.99, 1.0])
    return {"positions": rows, "top1": top1, "top1_percent": 100.0 * top1 / rows, "mean_kl": float(np.mean(kl_means)),
            "row_mean_kl": float(np.mean(kl_rows)), "max_kl": max_kl,
            "max_abs_p50": float(q[0]), "max_abs_p90": float(q[1]), "max_abs_p99": float(q[2]), "max_abs_max": float(q[3])}


def same(histories: list[dict], a_dir: Path, b_dir: Path) -> dict:
    equal = rows = 0
    for h in histories:
        a, why = open_capture(a_dir, h["name"], VOCAB)
        b, why_b = open_capture(b_dir, h["name"], VOCAB)
        if a is None or b is None:
            raise SystemExit(why or why_b)
        for i in range(a.shape[0]):
            equal += int(np.array_equal(a[i].view(np.uint32), b[i].view(np.uint32)))
        rows += a.shape[0]
    return {"rows": rows, "bit_equal_rows": equal}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--histories", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--cand", action="append", default=[])
    p.add_argument("--same", action="append", default=[])
    p.add_argument("--json", default="")
    args = p.parse_args()
    histories = json.loads(Path(args.histories).read_text())
    dirs = dict(c.split("=", 1) for c in args.cand)
    out = {"reference": args.ref, "against_reference": {}, "bit_equal": {}}
    for name, folder in dirs.items():
        out["against_reference"][name] = r = against(histories, Path(args.ref), Path(folder))
        print(f"{name}: top-1 {r['top1']}/{r['positions']} ({r['top1_percent']:.2f}%), mean KL {r['mean_kl']:.6f}, "
              f"row-mean KL {r['row_mean_kl']:.6f}, max KL {r['max_kl']:.3f}, max |dlogit| p50 {r['max_abs_p50']:.4f} "
              f"p90 {r['max_abs_p90']:.4f} p99 {r['max_abs_p99']:.4f} max {r['max_abs_max']:.4f}")
    for pair in args.same:
        a, b = pair.split("=", 1)
        out["bit_equal"][pair] = s = same(histories, Path(dirs[a]), Path(dirs[b]))
        print(f"{a} vs {b}: {s['bit_equal_rows']} of {s['rows']} rows bit-equal")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
