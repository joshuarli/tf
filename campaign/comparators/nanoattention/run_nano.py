"""Run nanoattention over the campaign prompts through its JSONL session protocol.

nanoattention has no HTTP server: `flint session` reads one JSON request per line
and keeps weights and KV resident. A request is a fresh cold prompt (reset true)
followed by an identical greedy repeat; both are timed on the client clock from
write to the answered line. Each prompt's ids come from prompt_ids.py, the
checkpoint's own chat-templated 1024/4096/32768-token prompt.
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))
sys.path.insert(0, str(HERE))

import harness  # noqa: E402
from prompt_ids import prompt_ids  # noqa: E402

OUTPUT_TOKENS = 512


def load_prompts() -> dict:
    return json.loads(harness.PROMPT_MANIFEST.read_bytes())


def select(manifest: dict, cells: list[int]) -> dict[int, list[dict]]:
    by_cell: dict[int, list[dict]] = {}
    for prompt in manifest["prompts"]:
        if prompt["set"] != "dev" or prompt["cell"] not in cells:
            continue
        by_cell.setdefault(prompt["cell"], []).append(prompt)
    for prompts in by_cell.values():
        prompts.sort(key=lambda p: p["name"])
    return by_cell


def read_answer(process, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"nanoattention exited: {process.stderr.read()[-2000:]}")
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        return json.loads(line)
    raise TimeoutError("nanoattention did not answer before the timeout")


def request(process, ids: list[int], *, reset: bool, timeout: float) -> dict:
    body = {"tokens": ids, "max_tokens": OUTPUT_TOKENS, "reset": reset}
    sent = time.perf_counter()
    process.stdin.write(json.dumps(body) + "\n")
    process.stdin.flush()
    answer = read_answer(process, timeout)
    completed = time.perf_counter()
    tokens = answer.get("decode_forward_tokens")
    decode = answer.get("decode_seconds")
    first = completed - sent - (decode or 0)
    return {"sent": sent, "completed": completed, "first": sent + max(first, 0),
            "pieces": [], "ttft_s": first if first > 0 else None,
            "decode_s": decode, "total_s": completed - sent, "complete": True,
            "tokens": tokens, "completion": "truncated" if not answer.get("eos") else "finished",
            "finish_reasons": ["eos" if answer.get("eos") else "length"],
            "prompt_tokens": answer.get("prefill_tokens"), "cached_tokens": None,
            "cache_state": "unverified", "prefill_seconds": answer.get("prefill_seconds"),
            "decode_tps": (tokens - 1) / decode if tokens and decode else None}


def trial(path, *, run_id, metadata, mode, cell, cache_gib, pair, prompt, ids, cold, warm, natural):
    record = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "run_id": run_id, "engine": "nanoattention", "binary": metadata["binary"], "mode": mode,
              "mode_flags": metadata.get("argv"), "cell": cell, "prompt_cache_gib": cache_gib, "pair": pair,
              "prompt": {"name": prompt["name"], "set": prompt["set"], "topic": prompt["topic"],
                         "path": prompt["path"], "sha256": prompt["sha256"],
                         "tokens": prompt["measured_prompt_tokens"],
                         "local_content_tokens": prompt["local_content_tokens"]},
              "model": {"key": metadata.get("model_key"), "dir": metadata.get("model_dir"),
                        "served": "flint", "hashes": metadata.get("hashes")},
              "server": {"argv": metadata.get("argv"), "port": None},
              "host": harness.host_summary(), "requests": {"cold": cold, "warm": warm},
              "natural_termination": natural, "footprint": None, "valid": False, "invalid_reasons": [],
              "omitted": metadata.get("omitted") or {}, "error": None}
    before = harness.conditions()
    reasons = list(before["pressure_reasons"])
    for name, run in (("cold", cold), ("warm", warm)):
        if run.get("tokens") != OUTPUT_TOKENS:
            reasons.append(f"{name} returned {run.get('tokens')} tokens, not {OUTPUT_TOKENS}")
        if run.get("prompt_tokens") != cell:
            reasons.append(f"{name} prompt_tokens {run.get('prompt_tokens')} != cell {cell}")
    record["invalid_reasons"] = reasons
    record["valid"] = not reasons
    after = harness.conditions()
    record["pressure_before"] = before["pressure"]
    record["pressure_after"] = after["pressure"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    rate = cold.get("decode_tps")
    print(f"{prompt['name']} {mode} {cell} pair {pair}: cold {None if rate is None else round(rate, 1)} tok/s, "
          f"valid={record['valid']}", flush=True)


def run(args) -> int:
    key = harness.model_key(args.model)
    model_dir = harness.model_dir(key)
    manifest = load_prompts()
    cells = [int(token.lstrip("~")) for token in args.cells.split(",") if token.strip()]
    by_cell = select(manifest, cells)
    for cell in cells:
        if problem := harness.context_problem(cell, OUTPUT_TOKENS):
            raise SystemExit(problem)
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"nanoattention-{args.mode}-{'-'.join(map(str, cells))}-{stamp.strftime('%H%M%S')}"
    out_dir = harness.RUNS_DIR / stamp.astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    path.touch()
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("nano_server", HERE / "server.py")
    server = module_from_spec(spec)
    spec.loader.exec_module(server)
    metadata = server.metadata(model_dir)
    metadata["argv"] = server.argv(model_dir, 0, 0, OUTPUT_TOKENS)
    failures = 0
    for cell in cells:
        for pair in range(1, args.pairs + 1):
            log_path = harness.TMP / f"nano-{run_id}-{cell}-{pair}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("wb")
            proc = subprocess.Popen(metadata["argv"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=log, text=True, bufsize=1, start_new_session=True,
                                    env=dict(os.environ, **server.env(model_dir, 0)))
            try:
                ready = proc.stdout.readline()
                if not ready:
                    raise RuntimeError(f"nanoattention exited during load: {log_path.read_text()[-2000:]}")
                for prompt in by_cell[cell]:
                    text = (harness.REPO / prompt["path"]).read_text(encoding="utf-8")
                    ids = prompt_ids(model_dir, text)
                    if problem := harness.context_problem(len(ids), OUTPUT_TOKENS):
                        raise ValueError(f"{prompt['name']}: {problem}")
                    cold = request(proc, ids, reset=True, timeout=args.ready_timeout)
                    warm = request(proc, ids, reset=True, timeout=args.ready_timeout)
                    natural = request(proc, ids, reset=True, timeout=args.ready_timeout)
                    trial(path, run_id=run_id, metadata=metadata, mode=args.mode, cell=cell, cache_gib=0,
                          pair=pair, prompt=prompt, ids=ids, cold=cold, warm=warm, natural=natural)
            except Exception as error:  # noqa: BLE001 - a failed trial is a recorded result
                failures += 1
                print(f"cell {cell} pair {pair}: {type(error).__name__}: {error}", file=sys.stderr)
            finally:
                server.stop(proc)
                log.close()
    print(f"appended trials to {path}" if path.stat().st_size else "no trials recorded", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--mode", default="greedy")
    parser.add_argument("--cells", default="1024,4096")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
