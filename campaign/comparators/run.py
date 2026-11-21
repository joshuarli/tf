"""Run one comparator engine over the campaign prompts and append campaign-schema trials.

The record schema is campaign/bench/bench.py's: scorecard.py reads `kind: trial`
records (and `kind: unavailable`). A comparator with no equivalent of a setting
records it in `omitted` rather than dropping the record silently.

Each comparator supplies `server.py` with:
  argv(model_dir, port, cache_gib, max_output) -> list[str]
  wait_ready(base, timeout) -> None
  metadata(model_dir) -> dict          # model_key, model_dir, hashes, omitted, argv, port, timing
  stop(process) -> None
"""

import argparse
import datetime as dt
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN / "bench"))
sys.path.insert(0, str(HERE))

import client  # noqa: E402
import harness  # noqa: E402

OUTPUT_TOKENS = 512
TEMPERATURE = 0
SEED = 1234


def load_prompts() -> dict:
    if not harness.PROMPT_MANIFEST.is_file():
        raise SystemExit(f"{harness.PROMPT_MANIFEST} is missing; run `make -C campaign prompts`")
    return json.loads(harness.PROMPT_MANIFEST.read_bytes())


def select(manifest: dict, cells: list[int], wanted: str) -> dict[int, list[dict]]:
    by_cell: dict[int, list[dict]] = {}
    names = None if wanted.strip() in ("", "all") else {n.strip() for n in wanted.split(",") if n.strip()}
    for prompt in manifest["prompts"]:
        if prompt["cell"] not in cells or prompt["set"] != "dev":
            continue
        if names is not None and prompt["name"] not in names:
            continue
        by_cell.setdefault(prompt["cell"], []).append(prompt)
    for prompts in by_cell.values():
        prompts.sort(key=lambda p: p["name"])
    return by_cell


def parse_cells(value: str) -> list[tuple[int, int]]:
    cells = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        cold = token.startswith("~") or token.startswith("cold:")
        cells.append((int(token.lstrip("~").removeprefix("cold:")), 0 if cold else 2))
    if not cells:
        raise SystemExit("CELLS must name at least one length")
    return cells


def load_server(name: str):
    path = HERE / name / "server.py"
    if not path.is_file():
        raise SystemExit(f"comparator {name} has no server.py at {path}")
    spec = importlib.util.spec_from_file_location(f"comparator_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trial(path, *, base, model_id, metadata, run_id, engine, mode, flags, cell, cache_gib, pair, order, prompt):
    record = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "run_id": run_id, "engine": engine, "binary": metadata["binary"], "mode": mode, "mode_flags": flags,
              "cell": cell, "prompt_cache_gib": cache_gib, "pair": pair, "pair_order": order,
              "prompt": {"name": prompt["name"], "set": prompt["set"], "topic": prompt["topic"],
                         "path": prompt["path"], "sha256": prompt["sha256"],
                         "tokens": prompt["measured_prompt_tokens"],
                         "local_content_tokens": prompt["local_content_tokens"]},
              "model": {"key": metadata.get("model_key"), "dir": metadata.get("model_dir"),
                        "served": model_id, "hashes": metadata.get("hashes")},
              "server": {"argv": metadata.get("argv"), "port": metadata.get("port"), **metadata.get("timing", {})},
              "host": harness.host_summary(), "requests": {}, "natural_termination": None,
              "footprint": metadata.get("footprint"), "valid": False, "invalid_reasons": [],
              "omitted": metadata.get("omitted") or {}, "error": None}
    before = harness.conditions()
    record["pressure_before"] = before["pressure"]
    text = (harness.REPO / prompt["path"]).read_text(encoding="utf-8")
    try:
        cold = client.chat_request(base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE, seed=SEED)
        record["requests"]["cold"] = harness.request_record(cold, "cold")
        warm = client.chat_request(base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE, seed=SEED)
        record["requests"]["warm"] = harness.request_record(warm, "warm", cold)
        natural = client.chat_request(base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE,
                                      ignore_eos=False, seed=SEED)
        record["natural_termination"] = harness.request_record(natural, "natural")
        after = harness.conditions()
        record["pressure_after"] = after["pressure"]
        record["invalid_reasons"] = before["pressure_reasons"] + after["pressure_reasons"]
        for name, run in (("cold", cold), ("warm", warm)):
            if not run.get("complete"):
                record["invalid_reasons"].append(f"{name} request did not complete")
            if run.get("tokens") != OUTPUT_TOKENS:
                record["invalid_reasons"].append(f"{name} returned {run.get('tokens')} tokens, not {OUTPUT_TOKENS}")
            if run.get("prompt_tokens") != cell:
                record["invalid_reasons"].append(f"{name} prompt_tokens {run.get('prompt_tokens')} != cell {cell}")
        record["valid"] = not record["invalid_reasons"]
    except Exception as error:  # noqa: BLE001 - a failed trial is a recorded result, not a crash
        record["error"] = f"{type(error).__name__}: {error}"
        record["pressure_after"] = harness.conditions()["pressure"]
        record["invalid_reasons"] = [record["error"]]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    rate = (record["requests"].get("cold") or {}).get("decode_tps")
    print(f"{prompt['name']} {mode} {cell} pair {pair}: cold {rate if rate is None else round(rate, 1)} tok/s, "
          f"valid={record['valid']}", flush=True)


def unavailable(path, *, run_id, engine, metadata, mode, cell, reason):
    entry = {"schema": 1, "kind": "unavailable", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": metadata.get("binary"), "mode": mode, "cell": cell,
             "reason": reason, "model": metadata.get("model_key"), "host": harness.host_summary()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def failure(path, *, run_id, engine, metadata, mode, cell, cache_gib, pair, error):
    entry = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": metadata.get("binary"), "mode": mode, "mode_flags": [],
             "cell": cell, "prompt_cache_gib": cache_gib, "pair": pair,
             "model": {"key": metadata.get("model_key"), "dir": metadata.get("model_dir")},
             "host": harness.host_summary(), "requests": {}, "natural_termination": None, "footprint": None,
             "valid": False, "invalid_reasons": [error], "error": error}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def run(args) -> int:
    server = load_server(args.name)
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_dir():
        raise SystemExit(f"model directory {model_dir} is missing")
    manifest = load_prompts()
    cells = parse_cells(args.cells)
    by_cell = select(manifest, [cell for cell, _ in cells], args.prompts)
    missing = [cell for cell, _ in cells if not by_cell.get(cell)]
    if missing:
        raise SystemExit(f"no prompts for cells {missing}")
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"{args.name}-{args.mode}-{'-'.join(str(c) for c, _ in cells)}-{stamp.strftime('%H%M%S')}"
    out_dir = harness.RUNS_DIR / stamp.astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    path.touch()
    engine = args.name
    failures = 0
    meta_base = server.metadata(model_dir)
    for cell, cache_gib in cells:
        for pair in range(1, args.pairs + 1):
            port = harness.free_port()
            base = f"http://127.0.0.1:{port}"
            argv = server.argv(model_dir, port, cache_gib, OUTPUT_TOKENS)
            log_path = harness.TMP / f"cmp-{run_id}-{cell}-{pair}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("wb")
            launch = time.perf_counter()
            env = dict(__import__("os").environ, **server.env(model_dir, cache_gib))
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            metadata = dict(meta_base, argv=argv, port=port)
            try:
                server.wait_ready(base, args.ready_timeout)
                metadata["timing"] = {"spawn_to_ready_s": round(time.perf_counter() - launch, 3)}
                metadata["footprint"] = server.footprint(base, process)
                model_id = server.model_id(base)
                for prompt in by_cell[cell]:
                    trial(path, base=base, model_id=model_id, metadata=metadata, run_id=run_id, engine=engine,
                          mode=args.mode, flags=list(argv), cell=cell, cache_gib=cache_gib, pair=pair,
                          order=0, prompt=prompt)
            except Exception as error:  # noqa: BLE001 - a dead server is a recorded failure
                failures += 1
                failure(path, run_id=run_id, engine=engine, metadata=metadata, mode=args.mode, cell=cell,
                        cache_gib=cache_gib, pair=pair, error=f"{type(error).__name__}: {error}")
                print(f"cell {cell} pair {pair}: {type(error).__name__}: {error}", file=sys.stderr)
            finally:
                server.stop(process)
                log.close()
    if path.exists() and path.stat().st_size == 0:
        path.unlink()
    print(f"appended trials to {path}" if path.exists() else "no trials recorded", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="comparator directory under campaign/comparators")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", default="greedy")
    parser.add_argument("--cells", default="1024,4096,32768")
    parser.add_argument("--prompts", default="all")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=1800)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
