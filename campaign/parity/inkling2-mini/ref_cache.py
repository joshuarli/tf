"""Run the float32 reference over the parity histories once, caching its logits by history sha256.

The reference is the slow side of the comparison and the Metal runner is fast, so the logits of each history are
written once and a rerun with unchanged histories recomputes nothing. A history is identified by the sha256 its
entry carries, which is also the identity the parity report names.

The reference streams one layer at a time under the GPU lock; a rerun of the whole corpus is one model at a time,
never two. Scenarios other than logits are not cached here: they are the runner's own fixtures.

    python3 campaign/parity/inkling2-mini/ref_cache.py --model ~/models/Inkling-2-Mini@... \
        --histories campaign/parity/inkling2-mini/histories.json --out .work/parity/inkling2-mini/ref \
        --reference campaign/ref/inkling2_loop_ref.py --venv .work/venv-ref --lock .work/gpu.lock \
        --expert-cache-gib 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--expert-cache-gib", type=float, default=10.0)
    args = parser.parse_args()

    histories = json.loads(Path(args.histories).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    python = str(Path(args.venv) / "bin" / "python")
    ran = 0
    for entry in histories:
        name = entry["name"]
        tokens = entry["token_ids"]
        if len(tokens) < 2:
            print(f"{name}: {len(tokens)} tokens, too short to teacher-force", file=sys.stderr)
            return 1
        if entry.get("sha256") is not None and entry["sha256"] != digest(tokens):
            print(f"{name}: sha256 in histories.json does not name these token ids", file=sys.stderr)
            return 1
        held = out / f"{name}.npy"
        stamp = out / f"{name}.sha256"
        if held.exists() and stamp.exists() and stamp.read_text().strip() == digest(tokens):
            continue
        ids = " ".join(str(t) for t in tokens)
        cmd = ["lockf", args.lock, python, args.reference, "--model", args.model,
               "--tokens", ids, "--out", str(held), "--expert-cache-gib", str(args.expert_cache_gib)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout, done.stderr, file=sys.stderr)
            return done.returncode
        stamp.write_text(digest(tokens) + "\n")
        ran += 1
        logits = np.load(held)
        print(f"{name}: {logits.shape[0]} rows, {len(tokens)} tokens")
    print(f"reference logits: {len(histories)} histories, {ran} computed, {len(histories) - ran} cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
