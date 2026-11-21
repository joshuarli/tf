"""rec v0.1.0 (rec_mlx) as an F3 comparator: argv, readiness, identity and stop.

rec always stops on EOS, so `ignore_eos` has no equivalent: the fixed-length protocol
uses `max_tokens` only and any early stop shows up as a short reply that the driver
records invalid. It has no prompt-cache size setting and no `/memory` endpoint, so
`prompt_cache_gib` is 0 in argv, prefix re-use is whatever rec does, and footprint is
recorded unavailable.
"""

import hashlib
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]


def shim() -> Path:
    return WORK / "build" / "rec@e53663b262af" / "engine"


def argv(model_dir: Path, port: int, cache_gib: int, max_output: int) -> list[str]:
    return ["lockf", str(WORK / "gpu.lock"), sys.executable, "-m", "rec_mlx.cli", "serve", "flint",
            "--model-dir", str(model_dir), "--manifest", str(model_dir / "flint-int4.json"),
            "--context-limit", "262144", "--grammar", "none", "--port", str(port)]


def env(model_dir: Path, cache_gib: int) -> dict:
    return {"PYTHONPATH": str(shim()), "MLX_METAL_FAST_SYNCH": "1"}


def wait_ready(base: str, timeout: float) -> None:
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError("rec did not report healthy before the startup timeout")


def model_id(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
        data = json.loads(response.read())
    return (data.get("data") or [{}])[0].get("id") or "flint"


def metadata(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "flint-int4.json").read_bytes())
    weights = []
    for name, digest in sorted(manifest["checkpoint_files"].items()):
        path = model_dir / name
        weights.append({"name": name, "sha256": digest,
                        "size": path.stat().st_size if path.is_file() else None})
    aggregate = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"model_key": "flint-int4", "model_dir": str(model_dir),
            "binary": {"role": "comparator", "path": str(shim()), "sha256": None,
                       "version": "rec v0.1.0 e53663b262af4cb1ca498e596ca8b1a2d79e0800"},
            "hashes": {"revision": manifest.get("revision"), "weights": weights,
                       "aggregate_sha256": aggregate},
            "omitted": {"prompt_cache_gib": "rec has no prompt-cache size setting; it caches prefixes itself",
                        "ignore_eos": "rec always stops on EOS; a short reply is recorded invalid"},
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
    except Exception:  # noqa: BLE001 - a stubborn server is force-killed
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
