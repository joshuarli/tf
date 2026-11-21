"""nanoattention (Rach's engine / codegraff's) as an F3 comparator.

It has no HTTP server. `flint session CHECKPOINT --context N` reads JSONL
{"tokens":[..],"max_tokens":N,"reset":bool} on stdin and keeps weights and KV
resident, answering each line with one JSON object carrying prefill_seconds,
decode_seconds and the greedy token ids; every generated token, including EOS,
is committed. Two differences from the HTTP engines are recorded in `omitted`:
it stops at EOS (no ignore_eos), and it has no prefix-cache size setting.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import harness

WORK = harness.WORK
BIN = WORK / "build" / "nanoattention@6f782a8dace8" / "zig-out" / "bin" / "flint"


def argv(model_dir: Path, port: int, cache_gib: int, max_output: int) -> list[str]:
    return ["lockf", str(WORK / "gpu.lock"), str(BIN), "session", str(model_dir), "--context", "262144"]


def env(model_dir: Path, cache_gib: int) -> dict:
    return {"TMPDIR": str(WORK / "tmp")}


def wait_ready(base: str, timeout: float) -> None:
    pass


def model_id(base: str) -> str:
    return "flint"


def metadata(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "flint-int4.json").read_bytes())
    weights = []
    for name, digest in sorted(manifest["checkpoint_files"].items()):
        path = model_dir / name
        weights.append({"name": name, "sha256": digest,
                        "size": path.stat().st_size if path.is_file() else None})
    aggregate = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"model_key": "flint-int4", "model_dir": str(model_dir),
            "binary": {"role": "comparator", "path": str(BIN), "sha256": _sha256(BIN),
                       "version": "nanoattention 6f782a8dace846e4e64a5a12cbdf5c0252166199"},
            "hashes": {"revision": manifest.get("revision"), "weights": weights,
                       "aggregate_sha256": aggregate},
            "omitted": {"ignore_eos": "nanoattention stops at EOS; a reply short of 512 tokens is recorded invalid",
                        "prompt_cache_gib": "nanoattention has no prefix-cache size setting"},
            "footprint": None}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def footprint(base: str, process) -> dict:
    return None


def stop(process) -> None:
    if process.poll() is not None:
        return
    try:
        process.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 15)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
