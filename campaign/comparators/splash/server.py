"""Splash 1.7.2 as the same-model Inkling 2 Mini reference (plan Decision 17).

Splash always decodes with its DFlash2 draft; it has no autoregressive mode. It runs from the
pinned release build with HOME pointed at .work/splash-home, so its prepared-model links, decode
cost calibration and caches stay inside the campaign scratch. The SSD prefix cache is off
(`--max-cache-disk 0`) and weights stay resident (`--idle-release off`).

SPLASH_PROFILE selects the arithmetic:
  controlled  bf16 KV and bf16 activations, the closest match to TensorFold's f16 KV path;
  default     Splash's own defaults (int8 KV, int8 prefill activations on M5).
SPLASH_VERIFY_ROWS is `auto` (Splash's default, outputs vary run to run) or `all`.
"""

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

import harness

WORK = harness.WORK
HEAVY = harness.HEAVY
ROOT = WORK / "src" / "splash@2afc7337aacc" / "x" / "splash-1.7.2-arm64-macos26"
HOME = WORK / "splash-home"
DRAFT = Path.home() / "models" / "Inkling-2-Mini-DFlash2@4f9088ea66c2"
PROFILES = {"controlled": ["--kv-format", "bf16", "--activations", "bf16"], "default": []}


def profile() -> str:
    name = os.environ.get("SPLASH_PROFILE", "controlled")
    if name not in PROFILES:
        raise SystemExit(f"SPLASH_PROFILE must be one of {sorted(PROFILES)}, not {name}")
    return name


def verify_rows() -> str:
    rows = os.environ.get("SPLASH_VERIFY_ROWS", "auto")
    if rows not in ("auto", "all"):
        raise SystemExit(f"SPLASH_VERIFY_ROWS must be auto or all, not {rows}")
    return rows


def binary() -> Path:
    return ROOT / "engine" / "splash"


def target(model_dir: Path) -> Path:
    ggufs = sorted(model_dir.glob("*.gguf"))
    return ggufs[0] if len(ggufs) == 1 else model_dir


def argv(model_dir: Path, port: int, cache_gib: int, max_output: int) -> list[str]:
    if not binary().is_file():
        raise SystemExit(f"{binary()} is missing; run `make -C campaign splash-setup`")
    return ["python3", str(HEAVY), "--", str(ROOT / "python" / "bin" / "python3"), "-u",
            str(ROOT / "install" / "launcher.py"), "serve", "--model", "thinkingmachines/Inkling-2-Mini",
            "--model-dir", str(target(model_dir)), "--draft-model", str(DRAFT), "--language-only", "--offline",
            "--host", "127.0.0.1", "--port", str(port), "--no-webui", "--max-memory", "26G",
            "--max-context", "32K", "--max-cache-disk", "0", "--idle-release", "off",
            "--max-concurrent-requests", "1", "--verify-rows", verify_rows(), *PROFILES[profile()]]


def env(model_dir: Path, cache_gib: int) -> dict:
    return {"HOME": str(HOME), "TMPDIR": str(WORK / "tmp"), "HF_HUB_OFFLINE": "1"}


def wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/ready", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.25)
    raise TimeoutError("Splash did not report ready before the startup timeout")


def model_id(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
        return json.loads(response.read())["data"][0]["id"]


def metadata(model_dir: Path) -> dict:
    files = sorted(p for p in model_dir.iterdir() if p.suffix in (".gguf", ".safetensors"))
    weights = [{"name": p.name, "size": p.stat().st_size} for p in files]
    return {"model_key": model_dir.name, "model_dir": str(model_dir),
            "binary": {"role": "comparator", "path": str(binary()), "sha256": _sha256(binary()),
                       "version": "splash 1.7.2 2afc7337aacc044722fa95e4cb3da75c281a19a8"},
            "hashes": {"revision": model_dir.name.rsplit("@", 1)[-1], "weights": weights,
                       "draft": DRAFT.name, "profile": profile(), "verify_rows": verify_rows()},
            "omitted": {"autoregressive": "Splash always decodes with its DFlash2 draft",
                        "prompt_cache_gib": "Splash sizes its prefix cache itself; the SSD cache is off"},
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


def expected_prompt_tokens(prompt: dict) -> int | None:
    lengths = json.loads((harness.CAMPAIGN / "prompts" / "lengths-inkling-2-mini.json").read_bytes())["prompts"]
    entry = lengths.get(prompt["name"])
    return entry["prompt_tokens"] if entry else None
