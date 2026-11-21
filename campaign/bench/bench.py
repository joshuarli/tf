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
PARALLEL = 1
MODES = {
    "ar": {"flags": ["--no-drafts"], "drafter": False, "copy_drafts": False},
    "copy": {"flags": [], "drafter": False, "copy_drafts": True},
    "drafter": {"flags": [], "drafter": True, "copy_drafts": True},
}
UNSUPPORTED_MODES = {
    "combined": "the runtime has no flag that combines copy drafts with a named drafter distinct from `drafter`, "
                "which already runs both; the label is refused until distinct counters prove it",
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


def requested_set(wanted: str) -> str:
    value = wanted.strip()
    if value in ("", "all", "dev"):
        return "dev"
    if value in ("heldout", "holdout"):
        return "holdout"
    return "names"


def select(manifest: dict, cells: list[int], wanted: str) -> dict[int, list[dict]]:
    by_cell: dict[int, list[dict]] = {}
    requested = requested_set(wanted)
    names = {n.strip() for n in wanted.split(",") if n.strip()} if requested == "names" else None
    for prompt in manifest["prompts"]:
        if prompt["cell"] not in cells:
            continue
        if requested == "dev" and prompt["set"] != "dev":
            continue
        if requested == "holdout" and prompt["set"] != "holdout":
            continue
        if requested == "names" and prompt["name"] not in names:
            continue
        by_cell.setdefault(prompt["cell"], []).append(prompt)
    for prompts in by_cell.values():
        prompts.sort(key=lambda p: p["name"])
    return by_cell


def verify_prompts(by_cell: dict[int, list[dict]]) -> None:
    """Refuse a selection whose files are missing or no longer hash to the manifest."""
    for cell, prompts in sorted(by_cell.items()):
        for prompt in prompts:
            path = harness.REPO / prompt["path"]
            if not path.is_file():
                raise SystemExit(f"prompt {prompt['name']} is missing at {path}")
            digest = harness.sha256_file(path)
            if digest != prompt["sha256"]:
                raise SystemExit(f"prompt {prompt['name']} hashes {digest}, not the manifest's {prompt['sha256']}")


def mode_args(mode: str) -> tuple[list[str], bool, bool]:
    if mode in UNSUPPORTED_MODES:
        raise SystemExit(f"mode {mode} is unsupported: {UNSUPPORTED_MODES[mode]}")
    spec = MODES[mode]
    return list(spec["flags"]), spec["drafter"], spec["copy_drafts"]


def acceleration_problems(mode: str, cold: dict) -> list[str]:
    """Problems where the counters do not prove the acceleration the mode label claims."""
    cold = cold or {}
    drafted = cold.get("drafted_tokens") or 0
    rounds = cold.get("verification_rounds")
    problems = []
    if mode == "ar" and drafted > 0:
        problems.append(f"ar mode drafted {drafted} tokens")
    if mode == "copy" and not rounds:
        problems.append("copy mode reported no verification rounds")
    if mode == "drafter":
        if not cold.get("speculative"):
            problems.append("drafter mode reported no speculative counters")
        if drafted == 0:
            problems.append("drafter mode drafted no tokens, so the drafter did not run")
    return problems


def cache_problems(cache_gib: int, cold: dict) -> list[str]:
    cold = cold or {}
    cached = cold.get("cached_tokens")
    problems = []
    if cache_gib == 0 and isinstance(cached, int) and cached > 0:
        problems.append(f"cold-prefill cell reported {cached} cached tokens")
    if cache_gib == 0 and cold.get("cache_state") not in (None, "cold"):
        problems.append(f"cold-prefill cell reported cache state {cold.get('cache_state')!r}")
    return problems


def quantization_label(key: str, hashes: dict) -> str:
    for row in hashes.get("config") or []:
        if row.get("name", "").rsplit("/", 1)[-1] == "hf_quant_config.json":
            return f"{key}:{str(row.get('sha256', ''))[:12]}"
    return key.rsplit("-", 1)[-1] if "-" in key else key


def contract(*, key: str, hashes: dict, host: dict, mode: str, flags: list[str], cell: int, cache_gib: int,
             prompts: list[dict], copy_drafts: bool) -> dict:
    return {"model_key": key, "model_revision": hashes.get("revision"),
            "model_aggregate_sha256": hashes.get("aggregate_sha256"),
            "quantization": quantization_label(key, hashes),
            "host_signature": {field: host.get(field) for field in (
                "product", "os", "memory_bytes", "recommended_max_working_set_bytes",
                "performance_cores", "efficiency_cores", "gpu_cores", "metal")},
            "context_policy": {"parallel": PARALLEL, "cell_tokens": cell,
                               "served_model": None},
            "output_policy": {"max_tokens": OUTPUT_TOKENS, "temperature": TEMPERATURE, "seed": SEED,
                              "ignore_eos": True, "fixed_length": True},
            "cache_policy": {"prompt_cache_gib": cache_gib, "cold_prefill": cache_gib == 0},
            "acceleration": {"mode": mode, "flags": list(flags), "copy_drafts": copy_drafts},
            "cell": cell,
            "prompts": [{"name": prompt["name"], "sha256": prompt["sha256"]} for prompt in prompts]}


def pairing(experiment: str, session: str, cell: int, mode: str, pair: int, role: str,
            contract_value: dict) -> dict:
    return {"experiment": experiment, "session": session, "process_pair": f"{session}:{mode}:{cell}:{pair}",
            "role": role, "contract": contract_value}


def record_unavailable(path: Path, *, run_id: str, experiment: str, session: str, engine: str, binary: dict,
                       mode: str, cell: int, cache_gib: int, flags: list[str], identity: dict) -> None:
    entry = {"schema": 1, "kind": "unavailable", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags, "cell": cell,
             "prompt_cache_gib": cache_gib,
             "pairing": pairing(experiment, session, cell, mode, 0, binary.get("role"), identity),
             "reason": f"drafter checkpoint unavailable at {harness.drafter_path() or '~/models/Flint-DFlash2@39b0a7f7a12a'}; not fetched",
             "host": harness.host_summary()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def record_failure(path: Path, *, run_id: str, experiment: str, session: str, engine: str, binary: dict,
                   model_key: str, model: Path, mode: str, flags: list[str], cell: int, cache_gib: int,
                   pair: int, identity: dict, error: str) -> None:
    entry = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags, "cell": cell,
             "prompt_cache_gib": cache_gib, "pair": pair,
             "pairing": pairing(experiment, session, cell, mode, pair, binary.get("role"), identity),
             "model": {"key": model_key, "dir": str(model)},
             "host": harness.host_summary(), "requests": {}, "natural_termination": None, "footprint": None,
             "valid": False, "invalid_reasons": [error], "error": error}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")
        stream.flush()


def trial(path: Path, *, run_id: str, experiment: str, session: str, engine: str, binary: dict, model_key: str,
          mode: str, flags: list[str], cell: int, cache_gib: int, pair: int, order: int, prompt: dict,
          server: harness.Server, model_id: str, hashes: dict, identity: dict) -> None:
    record = {"schema": 1, "kind": "trial", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "run_id": run_id, "engine": engine, "binary": binary, "mode": mode, "mode_flags": flags,
              "cell": cell, "prompt_cache_gib": cache_gib, "pair": pair, "pair_order": order,
              "pairing": pairing(experiment, session, cell, mode, pair, binary.get("role"), identity),
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
        cold_cached = record["requests"]["cold"].get("cached_tokens")
        warm_cached = record["requests"]["warm"].get("cached_tokens")
        record["cache_observed"] = {"prompt_cache_gib": cache_gib, "cold_cached_tokens": cold_cached,
                                    "warm_cached_tokens": warm_cached}
        record["invalid_reasons"] += cache_problems(cache_gib, record["requests"]["cold"])
        speculative = record["requests"]["cold"].get("speculative") or {}
        record["acceleration_observed"] = {
            "declared": mode, "drafted_tokens": record["requests"]["cold"].get("drafted_tokens"),
            "accepted_tokens": record["requests"]["cold"].get("accepted_tokens"),
            "verification_rounds": record["requests"]["cold"].get("verification_rounds"),
            "tokens_per_round": record["requests"]["cold"].get("tokens_per_round"),
            "reported": bool(speculative)}
        record["invalid_reasons"] += acceleration_problems(mode, record["requests"]["cold"])
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
    if args.mode in UNSUPPORTED_MODES:
        raise SystemExit(f"mode {args.mode} is unsupported: {UNSUPPORTED_MODES[args.mode]}")
    key = harness.model_key(args.model)
    model = harness.model_dir(key)
    if not model.is_dir():
        raise SystemExit(f"model directory {model} is missing; run `make -C campaign fetch NAMES={key}`")
    manifest = load_prompts()
    cells = parse_cells(args.cells)
    by_cell = select(manifest, [cell for cell, _ in cells], args.prompts)
    missing = [cell for cell, _ in cells if not by_cell.get(cell)]
    if missing:
        raise SystemExit(f"no prompts for cells {missing} from selection {args.prompts!r}")
    verify_prompts(by_cell)
    flags, wants_drafter, copy_drafts = mode_args(args.mode)
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
    experiment = args.experiment or run_id
    session = args.session or run_id
    out_dir = harness.RUNS_DIR / stamp.astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    path.touch()
    hashes = harness.model_hashes(key)
    host = harness.host_summary()
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
                contract_value = contract(key=key, hashes=hashes, host=host, mode=args.mode,
                                          flags=list(flags), cell=cell, cache_gib=cache_gib,
                                          prompts=by_cell[cell], copy_drafts=copy_drafts)
                intended = list(flags) + (["--drafter", "~/models/Flint-DFlash2@39b0a7f7a12a"] if wants_drafter else [])
                if wants_drafter and drafter is None:
                    record_unavailable(path, run_id=run_id, experiment=experiment, session=session,
                                       engine=engine, binary=identity,
                                       mode=args.mode, cell=cell, cache_gib=cache_gib, flags=intended,
                                       identity=contract_value)
                    continue
                server_args = list(flags) + (["--drafter", str(drafter)] if drafter else [])
                log_path = harness.TMP / f"bench-{run_id}-{cell}-{args.mode}-{pair}-{role}.log"
                try:
                    with harness.Server(binary, model, server_args, cache_gib, log_path) as server:
                        served = server.models()
                        if not served:
                            raise RuntimeError("server listed no model id")
                        contract_value["context_policy"]["served_model"] = served[0]
                        for prompt in by_cell[cell]:
                            trial(path, run_id=run_id, experiment=experiment, session=session,
                                  engine=engine, binary=identity,
                                  model_key=key, mode=args.mode, flags=server_args, cell=cell,
                                  cache_gib=cache_gib, pair=pair, order=order, prompt=prompt, server=server,
                                  model_id=served[0], hashes=hashes, identity=contract_value)
                except Exception as error:  # noqa: BLE001 - a dead server is a recorded failure
                    failures += 1
                    record_failure(path, run_id=run_id, experiment=experiment, session=session,
                                   engine=engine, binary=identity,
                                   model_key=key, model=model, mode=args.mode, flags=server_args, cell=cell,
                                   cache_gib=cache_gib, pair=pair, identity=contract_value,
                                   error=f"{type(error).__name__}: {error}")
                    print(f"cell {cell} pair {pair} {role}: {type(error).__name__}: {error}", file=sys.stderr)
    if path.exists() and path.stat().st_size == 0:
        path.unlink()
    print(f"appended trials to {path}" if path.exists() else "no trials recorded", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--mode", default="ar", choices=sorted([*MODES, *UNSUPPORTED_MODES]))
    parser.add_argument("--cells", default=",".join(str(n) for n in (1024, 4096, 32768)))
    parser.add_argument("--prompts", default="all", help="all (dev), heldout/holdout, or comma-separated prompt names")
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN), help="baseline process binary")
    parser.add_argument("--candidate-bin", default=None, help="optional candidate binary, alternated per pair")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment", default=None,
                        help="experiment identity shared by the sessions that may be combined")
    parser.add_argument("--session", default=None, help="session identity for this run's pairing records")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
