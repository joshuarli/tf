"""Run the fixed-prompt Flint benchmark: one model server per (cell, pair, binary), raw trials appended to runs/."""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

OUTPUT_TOKENS = 512
TEMPERATURE = 0
SEED = 1234
MODES = {
    "ar": {"flags": ["--no-drafts"], "drafter": False},
    "copy": {"flags": [], "drafter": False},
    "drafter": {"flags": [], "drafter": True},
    "combined": {"flags": [], "drafter": True},
}


def parse_cells(value: str) -> list[tuple[int, int]]:
    cells = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        cold = token.startswith("~") or token.startswith("cold:")
        number = token.lstrip("~").removeprefix("cold:")
        cells.append((int(number), 0 if cold else 2))
    if not cells:
        raise SystemExit("CELLS must name at least one length")
    return cells


def load_prompts() -> dict:
    if not harness.PROMPT_MANIFEST.is_file():
        raise SystemExit(f"{harness.PROMPT_MANIFEST} is missing; run `make -C campaign prompts`")
    return json.loads(harness.PROMPT_MANIFEST.read_bytes())


def select(manifest: dict, cells: list[int], wanted: str) -> dict[int, list[dict]]:
    by_cell: dict[int, list[dict]] = {}
    names = None if wanted.strip() in ("", "all") else {n.strip() for n in wanted.split(",") if n.strip()}
    heldout = wanted.strip() in ("heldout", "holdout")
    for prompt in manifest["prompts"]:
        if prompt["cell"] not in cells:
            continue
        if names is not None and prompt["name"] not in names:
            continue
        if names is None and not heldout and prompt["set"] != "dev":
            continue
        if heldout and prompt["set"] != "holdout":
            continue
        by_cell.setdefault(prompt["cell"], []).append(prompt)
    for prompts in by_cell.values():
        prompts.sort(key=lambda p: p["name"])
    return by_cell


def mode_args(mode: str) -> tuple[list[str], bool]:
    spec = MODES[mode]
    return list(spec["flags"]), spec["drafter"]


def record_unavailable(path: Path, *, run_id: str, engine: str, binary: dict, mode: str, cell: int,
                       flags: list[str]) -> None:
    entry = {"schema": 1, "kind": "unavailable", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags, "cell": cell,
             "reason": f"drafter checkpoint unavailable at {harness.drafter_path() or '~/models/Flint-DFlash2@39b0a7f7a12a'}; not fetched",
             "host": harness.host_summary()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def record_failure(path: Path, *, run_id: str, engine: str, binary: dict, model_key: str, model: Path, mode: str,
                   flags: list[str], cell: int, cache_gib: int, pair: int, error: str) -> None:
    entry = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags, "cell": cell,
             "prompt_cache_gib": cache_gib, "pair": pair, "model": {"key": model_key, "dir": str(model)},
             "host": harness.host_summary(), "requests": {}, "natural_termination": None, "footprint": None,
             "valid": False, "invalid_reasons": [error], "error": error}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def trial(path: Path, *, run_id: str, engine: str, binary: dict, model_key: str, mode: str, flags: list[str],
          cell: int, cache_gib: int, pair: int, order: int, prompt: dict, server: harness.Server, model_id: str,
          hashes: dict) -> None:
    record = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags,
              "cell": cell, "prompt_cache_gib": cache_gib, "pair": pair, "pair_order": order,
              "prompt": {"name": prompt["name"], "set": prompt["set"], "topic": prompt["topic"],
                         "path": prompt["path"], "sha256": prompt["sha256"],
                         "tokens": prompt["measured_prompt_tokens"],
                         "local_content_tokens": prompt["local_content_tokens"]},
              "model": {"key": model_key, "dir": str(server.model), "served": model_id, "hashes": hashes},
              "server": {"argv": server.argv, "port": server.port, **server.timing()},
              "host": harness.host_summary(), "requests": {}, "natural_termination": None,
              "footprint": None, "valid": False, "invalid_reasons": [], "error": None}
    before = harness.conditions()
    record["pressure_before"] = before["pressure"]
    mark = server.sample_mark()
    text = (harness.REPO / prompt["path"]).read_text(encoding="utf-8")
    try:
        cold = harness.chat_request(server.base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE,
                                    seed=SEED)
        record["requests"]["cold"] = harness.request_record(cold, "cold")
        warm = harness.chat_request(server.base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE,
                                    seed=SEED)
        record["requests"]["warm"] = harness.request_record(warm, "warm", cold)
        natural = harness.chat_request(server.base, model_id, text, max_tokens=OUTPUT_TOKENS, temperature=TEMPERATURE,
                                       ignore_eos=False, seed=SEED)
        record["natural_termination"] = harness.request_record(natural, "natural")
        record["footprint"] = server.stats(mark)
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
        record["footprint"] = server.stats(mark)
        record["pressure_after"] = harness.conditions()["pressure"]
        record["invalid_reasons"] = [record["error"]]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    cold_rate = (record["requests"].get("cold") or {}).get("decode_tps")
    print(f"{prompt['name']} {mode} {cell} pair {pair}: cold {cold_rate if cold_rate is None else round(cold_rate, 1)} tok/s, "
          f"valid={record['valid']}", flush=True)


def run(args) -> int:
    key = harness.model_key(args.model)
    model = harness.model_dir(key)
    if not model.is_dir():
        raise SystemExit(f"model directory {model} is missing; run `make -C campaign fetch NAMES={key}`")
    manifest = load_prompts()
    cells = parse_cells(args.cells)
    by_cell = select(manifest, [cell for cell, _ in cells], args.prompts)
    missing = [cell for cell, _ in cells if not by_cell.get(cell)]
    if missing:
        raise SystemExit(f"no prompts for cells {missing}")
    flags, wants_drafter = mode_args(args.mode)
    drafter = harness.drafter_path() if wants_drafter else None
    if wants_drafter and drafter is None:
        print(f"{args.mode}: drafter checkpoint absent; recording the cells unavailable", file=sys.stderr)
    baseline = Path(args.bin).expanduser().resolve()
    candidate = Path(args.candidate_bin).expanduser().resolve() if args.candidate_bin else None
    for binary in [baseline, *([candidate] if candidate else [])]:
        if not binary.is_file():
            raise SystemExit(f"binary {binary} is missing")
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"{args.model}-{args.mode}-{'-'.join(str(c) for c, _ in cells)}-{stamp.strftime('%H%M%S')}"
    out_dir = harness.RUNS_DIR / stamp.astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    path.touch()
    hashes = harness.model_hashes(key)
    binaries = [("baseline", baseline)] + ([("candidate", candidate)] if candidate else [])
    failures = 0
    for role, binary in binaries:
        print(f"{role} {binary} sha256 {harness.sha256_file(binary)}", file=sys.stderr)
    for cell, cache_gib in cells:
        for pair in range(1, args.pairs + 1):
            ordered = binaries if pair % 2 else list(reversed(binaries))
            for order, (role, binary) in enumerate(ordered):
                identity = {"role": role, "path": str(binary), "sha256": harness.sha256_file(binary)}
                engine = f"tensorfold-native@{identity['sha256'][:12]}"
                intended = list(flags) + (["--drafter", "~/models/Flint-DFlash2@39b0a7f7a12a"] if wants_drafter else [])
                if wants_drafter and drafter is None:
                    record_unavailable(path, run_id=run_id, engine=engine, binary=identity, mode=args.mode,
                                       cell=cell, flags=intended)
                    continue
                server_args = list(flags) + (["--drafter", str(drafter)] if drafter else [])
                log_path = harness.TMP / f"bench-{run_id}-{cell}-{args.mode}-{pair}-{role}.log"
                try:
                    with harness.Server(binary, model, server_args, cache_gib, log_path) as server:
                        served = server.models()
                        if not served:
                            raise RuntimeError("server listed no model id")
                        for prompt in by_cell[cell]:
                            trial(path, run_id=run_id, engine=engine, binary=identity, model_key=key,
                                  mode=args.mode, flags=server_args, cell=cell, cache_gib=cache_gib, pair=pair,
                                  order=order, prompt=prompt, server=server, model_id=served[0], hashes=hashes)
                except Exception as error:  # noqa: BLE001 - a dead server is a recorded failure
                    failures += 1
                    record_failure(path, run_id=run_id, engine=engine, binary=identity, model_key=key, model=model,
                                   mode=args.mode, flags=server_args, cell=cell, cache_gib=cache_gib, pair=pair,
                                   error=f"{type(error).__name__}: {error}")
                    print(f"cell {cell} pair {pair} {role}: {type(error).__name__}: {error}", file=sys.stderr)
    if path.exists() and path.stat().st_size == 0:
        path.unlink()
    print(f"appended trials to {path}" if path.exists() else "no trials recorded", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--mode", default="ar", choices=sorted(MODES))
    parser.add_argument("--cells", default=",".join(str(n) for n in (1024, 4096, 32768)))
    parser.add_argument("--prompts", default="all", help="all (dev), heldout, or comma-separated prompt names")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN), help="baseline process binary")
    parser.add_argument("--candidate-bin", default=None, help="optional candidate binary, alternated per pair")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
