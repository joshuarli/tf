"""Compare teacher-forced logits from the CPU reference with llama.cpp under a declared arithmetic contract.

Exits non-zero unless every history has the expected shape, finite values, numerical error within the
contract's bounds and full top-1 agreement on confident positions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from logit_gate import check, load_contract


def read_llama(path: Path) -> np.ndarray:
    tokens, vocab = np.fromfile(path, dtype=np.int32, count=2)
    logits = np.fromfile(path, dtype=np.float32, offset=8)
    if logits.size != int(tokens) * int(vocab):
        return np.empty((0, 0), dtype=np.float32)
    return logits.reshape(int(tokens), int(vocab))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument("--contract", default="f32-vs-f32")
    args = parser.parse_args()

    contract = load_contract(args.contract)
    histories = json.loads(Path(args.histories).read_text())
    if not histories:
        print("no histories to compare")
        return 1
    failed = []
    print(f"contract {contract.name}: {contract.bounds}")
    for entry in histories:
        name = entry["name"]
        teacher = read_llama(Path(args.llama_dir) / f"{name}.bin")
        reference = np.load(Path(args.ref_dir) / f"{name}.npy")
        result = check(teacher, reference, len(entry["token_ids"]), contract)
        print(f"\n{name}: {result.positions} positions, top-1 {result.top1:.2%}, "
              f"confident top-1 {result.confident_top1:.2%} on {result.confident}")
        print(f"  max |err| {result.max_abs:.4g} (position {result.worst_position}), mean |err| {result.mean_abs:.4g}, "
              f"max KL {result.max_position_kl:.4g}, mean KL {result.mean_kl:.4g}")
        for problem in result.problems:
            print(f"  FAIL {problem}")
        if result.problems:
            failed.append(name)

    print()
    if failed:
        print(f"{len(failed)} of {len(histories)} histories fail the {contract.name} contract: {', '.join(failed)}")
        return 1
    print(f"all {len(histories)} histories meet the {contract.name} contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
