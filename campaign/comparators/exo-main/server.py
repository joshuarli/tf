"""Exo (main or PR #2533 head) as an F3 comparator.

Exo runs as a single master node over Zenoh with an API on 127.0.0.1:52415.
It lists RecursiveSI/flint-int4, so a POST /place_instance loads the model and
/v1/chat/completions serves it. It has no ignore_eos (a reply short of 512
tokens is recorded invalid) and its cache knob is a boolean use_prefix_cache,
not a GiB size; both are recorded in `omitted`.
"""

import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import harness

WORK = harness.WORK
MODEL_ID = "RecursiveSI/flint-int4"


def repo(commit: str) -> Path:
    return WORK / "build" / f"exo-{commit[:6].removeprefix('a60b95')}"


def argv(model_dir: Path, port: int, cache_gib: int, max_output: int) -> list[str]:
    return [str(WORK / "tmp" / "exo-serve.sh")]


def env(model_dir: Path, cache_gib: int) -> dict:
    return {}


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=3) as response:
                if response.status == 200:
                    break
        except OSError:
            time.sleep(0.5)
            continue
        time.sleep(0.5)
    else:
        raise TimeoutError("exo did not list models before the startup timeout")
    body = json.dumps({"model_id": MODEL_ID, "sharding": "Pipeline", "instance_meta": "MlxRing",
                       "min_nodes": 1}).encode()
    request = urllib.request.Request(f"{base}/place_instance", data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(request, timeout=60).read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"exo refused the placement: {error}") from error


def model_id(base: str) -> str:
    return MODEL_ID


def metadata(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "flint-int4.json").read_bytes())
    weights = [{"name": name, "sha256": digest, "size": (model_dir / name).stat().st_size}
               for name, digest in sorted(manifest["checkpoint_files"].items()) if (model_dir / name).is_file()]
    aggregate = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    commit = json.loads((WORK / "exo-commit.json").read_text())["commit"] if (WORK / "exo-commit.json").is_file() else None
    return {"model_key": "flint-int4", "model_dir": str(model_dir),
            "binary": {"role": "comparator", "path": str(WORK / "tmp" / "exo-serve.sh"), "sha256": None,
                       "version": f"exo {commit or 'main'}"},
            "hashes": {"revision": manifest.get("revision"), "weights": weights,
                       "aggregate_sha256": aggregate},
            "omitted": {"ignore_eos": "exo stops at EOS; a reply short of 512 tokens is recorded invalid",
                        "prompt_cache_gib": "exo has a boolean use_prefix_cache, not a GiB size"},
            "footprint": None}


def footprint(base: str, process) -> dict:
    return None


def stop(process) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), 15)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
