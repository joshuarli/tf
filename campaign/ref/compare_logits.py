"""Compare teacher-forced logits from the CPU reference with llama.cpp.

The reference is the float32 oracle; llama.cpp is the teacher. For every
position this prints top-1 agreement, the absolute logit error and the KL
between the two next-token distributions. A position counts as confident when
the teacher's top-2 margin is at least --min-margin; any top-1 disagreement
there fails the check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def read_llama(path: Path) -> np.ndarray:
    tokens, vocab = np.fromfile(path, dtype=np.int32, count=2)
    logits = np.fromfile(path, dtype=np.float32, offset=8)
    return logits.reshape(int(tokens), int(vocab))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument("--min-margin", type=float, default=0.1)
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    failures = 0
    print(f"positions are teacher-forced; confident = llama.cpp top-2 margin >= {args.min_margin}")
    for entry in histories:
        name = entry["name"]
        teacher = read_llama(Path(args.llama_dir) / f"{name}.bin")
        reference = np.load(Path(args.ref_dir) / f"{name}.npy")
        if teacher.shape != reference.shape:
            print(f"{name}: shape mismatch llama {teacher.shape} reference {reference.shape}")
            return 1
        top_teacher = teacher.argmax(axis=1)
        top_ref = reference.argmax(axis=1)
        sorted_teacher = np.sort(teacher, axis=1)
        margin = sorted_teacher[:, -1] - sorted_teacher[:, -2]
        confident = margin >= args.min_margin
        agree = top_teacher == top_ref
        error = np.abs(teacher.astype(np.float64) - reference.astype(np.float64))
        kl = (np.exp(log_softmax(teacher)) * (log_softmax(teacher) - log_softmax(reference))).sum(axis=1)

        print(f"\n{name}: {len(entry['token_ids'])} tokens, "
              f"top-1 agreement {agree.mean() * 100:.1f}% overall, "
              f"{(f'{agree[confident].mean() * 100:.1f}%' if confident.any() else 'n/a')} "
              f"on {int(confident.sum())} confident positions")
        print(f"  max |logit error| {error.max():.6g}, mean |logit error| {error.mean():.6g}, mean KL {kl.mean():.6g}")
        print("  position  agree  margin    max|err|   KL       teacher/ref top-1")
        for i in range(teacher.shape[0]):
            mark = "ok" if agree[i] else ("FAIL" if confident[i] else "diff")
            print(f"  {i:8d}  {mark:6s} {margin[i]:8.4f}  {error[i].max():9.4g}  {kl[i]:8.4g}  "
                  f"{top_teacher[i]} / {top_ref[i]}")
        if not agree[confident].all():
            failures += 1
            print(f"  {int((~agree & confident).sum())} confident positions disagree")

    print()
    if failures:
        print(f"reference disagrees with llama.cpp on confident positions in {failures} histories")
        return 1
    print("reference reproduces llama.cpp top-1 on every confident position")
    return 0


if __name__ == "__main__":
    sys.exit(main())
