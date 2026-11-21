"""mlx-lm at the Exo-pinned rev as an F3 comparator.

mlx-lm resolves `model_type: flint` by importing `mlx_lm.models.flint`, which exists
only at the rev Exo pins (6b572ab1). Its server has `max_tokens` but no `ignore_eos`
equivalent and no prompt-cache size setting, so both are recorded omitted and a reply
short of 512 tokens is recorded invalid.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]


def binary() -> Path:
    return WORK / "venv-mlx-lm" / "bin" / "mlx_lm.server"


def argv(model_dir: Path, port: int, cache_gib: int, max_output: int) -> list[str]:
    return ["lockf", str(WORK / "gpu.lock"), str(binary()), "--model", str(model_dir),
            "--host", "127.0.0.1", "--port", str(port), "--temp", "0", "--top-p", "1", "--top-k", "0",
            "--max-tokens", str(max_output), "--log-level", "WARNING"]


def env(model_dir: Path, cache_gib: int) -> dict:
    return {}


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError("mlx-lm did not report healthy before the startup timeout")


def model_id(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
        data = json.loads(response.read())
    return (data.get("data") or [{}])[0].get("id") or "flint"


def metadata(model_dir: Path) -> dict:
    names = sorted(p.name for p in model_dir.glob("model-*.safetensors"))
    weights = [{"name": name, "sha256": _sha256(model_dir / name), "size": (model_dir / name).stat().st_size}
               for name in names]
    aggregate = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"model_key": "flint-int4", "model_dir": str(model_dir),
            "binary": {"role": "comparator", "path": str(binary()), "sha256": None,
                       "version": "mlx-lm 6b572ab1b531dd8fd2d13f6f36a89c2eccda18ea"},
            "hashes": {"revision": "6b572ab1b531dd8fd2d13f6f36a89c2eccda18ea", "weights": weights,
                       "aggregate_sha256": aggregate},
            "omitted": {"ignore_eos": "mlx-lm has no ignore_eos; a reply short of 512 tokens is recorded invalid",
                        "prompt_cache_gib": "mlx-lm has no prompt-cache size setting"}},
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
