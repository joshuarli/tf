"""Teacher-forced oMLX logits on the oQ4e checkpoint: the campaign's correctness oracle.

oMLX serves `Inkling-2-Mini-oQ4e-mtp` with its own Inkling 2 model code on MLX. This script starts oMLX's
`VLMBatchedEngine` in process with Lightning MTP off, as the oMLX comparator adapter does, so every patch oMLX
installs when it loads the checkpoint is in place. It then teacher-forces each saved history through the engine's
model adapter, the object the scheduler calls, with a fresh cache and at most `--chunk` tokens a forward (default:
the scheduler's prefill step). The forwards run on oMLX's MLX executor thread. Nothing goes through the scheduler
and nothing is sampled.

Each history's logits over the unpadded vocabulary stream to `<out>/<name>.npy` as float32 [tokens, 200058], the
layout of the saved float32 captures, so the same comparators read both. A file takes its final name only when
complete, beside `<name>.npy.receipt.json` (token-history sha256, rows, chunk, engine, output sha256). A history
whose receipt already names the current engine, checkpoint, chunk and tokens, and whose file still has the
recorded digest, is kept. `manifest.json` is rebuilt from those receipts on every run. It records the oMLX, MLX
and package versions, the checkpoint revision and its file checks against `campaign/sources.lock.json`, the
device, the settings, the command, every output digest and one digest over all of them.

`make -C campaign omlx-oracle-capture` runs it inside the pinned oMLX environment and the campaign's memory
ceiling. By hand, from the repository root (`-B` keeps bytecode out of the pinned environment):

    HF_HUB_OFFLINE=1 INKLING_LOAD_LIMIT_GB=26 python3 campaign/heavy.py -- \\
        .work/venv-omlx/bin/python -I -B campaign/parity/inkling2-mini/omlx_capture.py \\
        --out .work/parity/inkling2-mini/omlx-ref

It exits 0 only when every history in `--histories` has a finite capture under the current inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(HERE))

import provenance  # noqa: E402

VOCAB = 200058
SCHEMA = 1
LOCK_MODEL = "inkling-2-mini-oq4e-mtp"
CHECKPOINT = Path.home() / "models" / "Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
PACKAGES = ("omlx", "mlx", "mlx-metal", "mlx-lm", "mlx-vlm", "numpy", "transformers")
ENV_KEYS = ("HF_HUB_OFFLINE", "INKLING_LOAD_LIMIT_GB", "OMLX_WITH_CUSTOM_KERNEL", "HF_HOME", "TMPDIR")
RECEIPT_KEYS = ("rows", "token_sha256", "output_sha256", "nonfinite", "chunk", "seconds", "mlx_peak_bytes")
# Rows copied from MLX to the output file at a time.
COPY_ROWS = 128


def receipt_path(output: Path) -> Path:
    return output.with_name(output.name + ".receipt.json")


def engine_identity() -> dict:
    """Package versions and the oMLX source commit; a capture is kept only under the same engine."""
    direct = json.loads(metadata.distribution("omlx").read_text("direct_url.json") or "{}")
    return {"packages": {name: metadata.version(name) for name in PACKAGES},
            "omlx_commit": direct.get("vcs_info", {}).get("commit_id"),
            "python": platform.python_version()}


def lock_entries(model: Path) -> list[dict]:
    """The checkpoint's pinned files from sources.lock.json; the directory name must name their revision."""
    entries = [e for e in json.loads((CAMPAIGN / "sources.lock.json").read_text())["inputs"]
               if e.get("model") == LOCK_MODEL]
    revisions = {e["version"] for e in entries}
    if not entries or len(revisions) != 1:
        raise SystemExit(f"sources.lock.json pins no single revision for {LOCK_MODEL}")
    revision = next(iter(revisions))
    if not model.name.endswith("@" + revision[:12]):
        raise SystemExit(f"{model} is not the pinned revision {revision[:12]}")
    return entries


def check_checkpoint(model: Path, entries: list[dict], hash_weights: bool) -> list[dict]:
    """Each pinned file's size, and its sha256 unless only sizes are asked for, against the lock."""
    rows, bad = [], []
    for entry in sorted(entries, key=lambda e: e["name"]):
        path = model / entry["name"].split("/", 1)[1]
        row = {"file": path.name, "size": entry["size"], "lock_sha256": entry["sha256"]}
        if not path.is_file() or path.stat().st_size != entry["size"]:
            bad.append(f"{path.name}: missing or not {entry['size']} bytes")
        elif hash_weights or entry["size"] <= provenance.SMALL_FILE_LIMIT:
            row["sha256"] = provenance.sha256_file(path)
            if row["sha256"] != entry["sha256"]:
                bad.append(f"{path.name}: sha256 {row['sha256'][:12]} is not the pinned {entry['sha256'][:12]}")
        rows.append(row)
    if bad:
        raise SystemExit("checkpoint does not match sources.lock.json: " + "; ".join(bad))
    return rows


def wanted_receipt(name: str, tokens: list[int], chunk: int, engine: dict, model_identity: str) -> dict:
    return {"schema": SCHEMA, "name": name, "token_sha256": provenance.token_history_sha(tokens),
            "rows": len(tokens), "vocab": VOCAB, "dtype": "float32", "chunk": chunk, "mtp": False,
            "engine": engine, "model_identity": model_identity}


def held_receipt(output: Path, want: dict) -> dict | None:
    """The receipt of a finite output that names exactly the wanted inputs and still has its digest."""
    path = receipt_path(output)
    if not output.is_file() or not path.is_file():
        return None
    try:
        held = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if any(held.get(key) != value for key, value in want.items()) or held.get("nonfinite") != 0:
        return None
    return held if held.get("output_sha256") == provenance.sha256_file(output) else None


def teacher_force(mx, adapter, tokens: list[int], chunk: int, rows: np.ndarray) -> int:
    """Fill ``rows`` with the logits of every position; returns the count of non-finite logits."""
    cache = adapter.make_cache()
    nonfinite = 0
    for start in range(0, len(tokens), chunk):
        piece = tokens[start:start + chunk]
        logits = adapter(mx.array([piece]), cache=cache)
        if logits.dtype != mx.float32 or logits.shape[:2] != (1, len(piece)) or logits.shape[2] < VOCAB:
            raise RuntimeError(f"oMLX returned logits {logits.shape} {logits.dtype} for {len(piece)} tokens")
        padded = logits[0, :, VOCAB:]
        block = logits[0, :, :VOCAB]
        mx.eval(block, padded, [c.state for c in cache])
        if padded.size and not bool(mx.all(padded == -mx.inf).item()):
            raise RuntimeError("oMLX logits past the unpadded vocabulary are not -inf")
        for at in range(0, len(piece), COPY_ROWS):
            part = np.asarray(block[at:at + COPY_ROWS])
            rows[start + at:start + at + part.shape[0]] = part
            nonfinite += int(np.count_nonzero(~np.isfinite(part)))
        del logits, padded, block
    return nonfinite


async def capture(model: Path, out: Path, pending: list[dict], chunk: int, settings, config,
                  engine: dict, model_identity: str) -> dict:
    """Load the engine once and write each pending history's capture and receipt; returns run facts."""
    import mlx.core as mx
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.engine_core import get_mlx_executor

    vlm = VLMBatchedEngine(str(model), model_settings=settings, scheduler_config=config)
    began = time.monotonic()
    await vlm.start()
    run = {"load_seconds": round(time.monotonic() - began, 3)}
    try:
        if vlm.dflash_drafter is not None:
            raise RuntimeError("a Lightning MTP drafter is attached; the oracle runs with MTP off")
        run["model_class"] = f"{type(vlm._vlm_model).__module__}.{type(vlm._vlm_model).__qualname__}"
        executor = get_mlx_executor()
        loop = asyncio.get_running_loop()
        run["device"] = await loop.run_in_executor(executor, mx.device_info)
        for history in pending:
            name, tokens = history["name"], history["token_ids"]
            final = out / f"{name}.npy"
            partial = out / f".{name}.npy.partial"
            rows = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(tokens), VOCAB))
            began = time.monotonic()
            await loop.run_in_executor(executor, mx.reset_peak_memory)
            nonfinite = await loop.run_in_executor(executor, teacher_force, mx, vlm._adapter, tokens, chunk, rows)
            peak = await loop.run_in_executor(executor, mx.get_peak_memory)
            await loop.run_in_executor(executor, mx.clear_cache)
            rows.flush()
            del rows
            receipt = wanted_receipt(name, tokens, chunk, engine, model_identity) | {
                "output_sha256": provenance.sha256_file(partial), "nonfinite": nonfinite,
                "seconds": round(time.monotonic() - began, 3), "mlx_peak_bytes": int(peak)}
            os.replace(partial, final)
            receipt_path(final).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(f"omlx_capture: {name}: {len(tokens)} rows in {receipt['seconds']:.1f} s, "
                  f"{nonfinite} non-finite, MLX peak {peak / 1e9:.1f} GB", flush=True)
    finally:
        await vlm.stop()
    return run


def repo_path(path: Path) -> str:
    """``path`` relative to the repository root when it lies inside it, else absolute."""
    path = path.resolve()
    return str(path.relative_to(CAMPAIGN.parent)) if path.is_relative_to(CAMPAIGN.parent) else str(path)


def campaign_revision() -> dict:
    head = subprocess.run(["git", "-C", str(CAMPAIGN), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(CAMPAIGN), "status", "--porcelain", "--", "."],
                           capture_output=True, text=True)
    return {"head": head.stdout.strip() or None, "campaign_tree_dirty": bool(dirty.stdout.strip())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--histories", type=Path, default=HERE / "histories.json")
    parser.add_argument("--model", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--chunk", type=int, default=0,
                        help="tokens per forward; 0 takes the oMLX scheduler's prefill step")
    parser.add_argument("--names", default="", help="comma-separated histories to capture (default: all)")
    parser.add_argument("--force", action="store_true", help="recapture histories whose receipts still match")
    parser.add_argument("--skip-weight-hashes", action="store_true",
                        help="check the large weight files against the lock by size only")
    args = parser.parse_args()
    if args.chunk < 0:
        parser.error("--chunk must be positive, or 0 for the scheduler's prefill step")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("set HF_HUB_OFFLINE=1: the oracle loads the pinned local checkpoint only")

    histories = json.loads(args.histories.read_text())
    names = [n for n in args.names.split(",") if n]
    unknown = sorted(set(names) - {h["name"] for h in histories})
    if unknown:
        parser.error(f"unknown histories: {', '.join(unknown)}")
    for history in histories:
        tokens = history["token_ids"]
        if history.get("sha256") not in (None, provenance.token_history_sha(tokens)):
            raise SystemExit(f"{history['name']}: histories.json's sha256 does not name its token ids")
        if not tokens or any(type(t) is not int or not 0 <= t < VOCAB for t in tokens):
            raise SystemExit(f"{history['name']}: token ids must lie in the unpadded vocabulary")

    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig

    config = SchedulerConfig()
    settings = ModelSettings(mtp_enabled=False)
    chunk = args.chunk or int(config.prefill_step_size)
    model = args.model.expanduser().resolve()
    entries = lock_entries(model)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    files = check_checkpoint(model, entries, hash_weights=not args.skip_weight_hashes)
    model_identity = provenance.model_identity(model)
    engine = engine_identity()
    args.out.mkdir(parents=True, exist_ok=True)

    def want(history: dict) -> dict:
        return wanted_receipt(history["name"], history["token_ids"], chunk, engine, model_identity)

    selected = [h for h in histories if not names or h["name"] in names]
    pending = [h for h in selected if args.force or held_receipt(args.out / f"{h['name']}.npy", want(h)) is None]
    run = {"chunk": chunk, "prefill_step_size": int(config.prefill_step_size),
           "mtp_enabled": settings.mtp_enabled, "turboquant_kv_enabled": settings.turboquant_kv_enabled,
           "engine": "omlx.engine.vlm.VLMBatchedEngine",
           "call": "engine._adapter(token_ids[1, n], cache=engine._adapter.make_cache())",
           "captured": [h["name"] for h in pending]}
    if pending:
        run |= asyncio.run(capture(model, args.out, pending, chunk, settings, config, engine, model_identity))
    outputs = {}
    for history in histories:
        held = held_receipt(args.out / f"{history['name']}.npy", want(history))
        if held is not None:
            outputs[history["name"]] = {key: held.get(key) for key in RECEIPT_KEYS}
    complete = len(outputs) == len(histories)
    manifest = {
        "schema": SCHEMA,
        "oracle": "oMLX teacher-forced logits, Lightning MTP off",
        "complete": complete,
        "histories": {"path": repo_path(args.histories),
                      "sha256": provenance.sha256_file(args.histories), "count": len(histories),
                      "positions": sum(len(h["token_ids"]) for h in histories)},
        "layout": {"format": "npy", "dtype": "float32", "shape": "[tokens, vocab]", "vocab": VOCAB,
                   "row": "logits after reading token i of the history"},
        "engine": engine,
        "model": {"path": str(model), "revision": entries[0]["version"], "identity": model_identity,
                  "weights_hashed": not args.skip_weight_hashes, "files": files},
        "run": run,
        "command": [sys.executable, *sys.argv],
        "python_flags": {"isolated": bool(sys.flags.isolated),
                         "dont_write_bytecode": bool(sys.flags.dont_write_bytecode)},
        "cwd": os.getcwd(),
        "env": {key: os.environ.get(key) for key in ENV_KEYS},
        "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0]},
        "campaign": campaign_revision(),
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "outputs": outputs,
        "digest": provenance.capture_digest({n: o["output_sha256"] for n, o in outputs.items()}),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [h["name"] for h in histories if h["name"] not in outputs]
    print(f"omlx_capture: {len(outputs)}/{len(histories)} histories held, digest {manifest['digest'][:16]}"
          + (f", missing or non-finite: {', '.join(missing)}" if missing else ""))
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
