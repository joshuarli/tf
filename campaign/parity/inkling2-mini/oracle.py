"""Check that the saved reference logits are the corpus's, without recomputing them.

The reference is the campaign's expensive step, so the parity check compares against the outputs it already wrote.
This verifies those files instead: each history's file exists, is named by the sha256 the corpus records, and holds
one finite row a token over the unpadded vocabulary. A missing or stale file is named, so the caller knows the oracle
has to be produced again, not that the Metal pass is wrong.

    python3 campaign/parity/inkling2-mini/oracle.py --histories ... --dir .work/parity/inkling2-mini/ref
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

VOCAB = 200058


def digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    folder = Path(args.dir)
    problems: list[str] = []
    long_histories = 0
    for entry in histories:
        name, tokens = entry["name"], entry["token_ids"]
        if entry.get("sha256") is not None and entry["sha256"] != digest(tokens):
            problems.append(f"{name}: histories.json's sha256 does not name its token ids")
        if len(tokens) > 1024:
            long_histories += 1
        held = folder / f"{name}.npy"
        if not held.exists():
            problems.append(f"{name}: no saved logits at {held}")
            continue
        logits = np.load(held, mmap_mode="r")
        if logits.shape != (len(tokens), VOCAB):
            problems.append(f"{name}: saved logits are {logits.shape}, expected {(len(tokens), VOCAB)}")
            continue
        if not np.isfinite(np.asarray(logits[0])).all() or not np.isfinite(np.asarray(logits[-1])).all():
            problems.append(f"{name}: saved logits are not finite")
    if long_histories < 2:
        problems.append(f"only {long_histories} histories run past the 1,024-token sliding window; at least two must")
    for p in problems:
        print(f"oracle: {p}", file=sys.stderr)
    if problems:
        return 1
    print(f"oracle: {len(histories)} saved histories verified, {long_histories} past the sliding window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
