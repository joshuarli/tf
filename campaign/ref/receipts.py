"""Immutable receipts for the expensive Inkling 2 Mini oracle outputs.

`record` hashes the oracle's semantic inputs and its outputs; `verify` exits 0 only when every input still hashes
the same and every recorded output is present and unchanged, so the comparison can reuse them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
RECEIPTS = HERE / "oracle_receipts.json"
SOURCES = ["histories.json", "inkling2_loop_ref.py", "llama_logits.cpp", "convert/uv.lock", "uv.lock"]
PINNED = ["inkling-2-mini/model-00001-of-00001.safetensors", "inkling-2-mini/config.json",
          "inkling-2-mini/tokenizer.json", "inkling-2-mini/chat_template.jinja"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def inputs(llama_commit: str) -> dict:
    lock = {i["name"]: i for i in json.loads((CAMPAIGN / "sources.lock.json").read_text())["inputs"]}
    return {
        "sources": {name: sha256(HERE / name) for name in SOURCES},
        "pinned": {name: lock[name]["sha256"] for name in PINNED},
        "llama_commit": llama_commit,
        "oracle_policy": "reference: CPU float32 over exact NVFP4/MXFP8 decode; teacher: llama.cpp CPU, op_offload off, F32 KV cache, f32 GGUF",
    }


def outputs(out_dir: Path) -> dict:
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.suffix in (".npy", ".bin"))
    return {str(p.relative_to(out_dir)): sha256(p) for p in files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "verify"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--llama-commit", required=True)
    args = parser.parse_args()

    now = inputs(args.llama_commit)
    if args.action == "record":
        produced = outputs(args.out_dir)
        if not produced:
            print(f"no oracle outputs under {args.out_dir}", file=sys.stderr)
            return 1
        RECEIPTS.write_text(json.dumps({"inputs": now, "outputs": produced}, indent=2, sort_keys=True) + "\n")
        print(f"recorded {len(produced)} oracle outputs in {RECEIPTS.name}")
        return 0

    if not RECEIPTS.exists():
        print("no oracle receipts recorded")
        return 1
    recorded = json.loads(RECEIPTS.read_text())
    if recorded["inputs"] != now:
        changed = [k for k in now if recorded["inputs"].get(k) != now[k]]
        print(f"oracle inputs changed ({', '.join(changed)}); rerun the oracles")
        return 1
    present = outputs(args.out_dir)
    stale = [name for name, digest in recorded["outputs"].items() if present.get(name) != digest]
    if stale:
        print(f"{len(stale)} recorded oracle outputs missing or changed, e.g. {stale[0]}; rerun the oracles")
        return 1
    print(f"reusing {len(recorded['outputs'])} oracle outputs whose inputs are unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
