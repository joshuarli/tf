"""Run the campaign bench protocol with an explicit pair range.

campaign/bench/bench.py always numbers its pairs 1..N, but Decision 11 asks for six
balanced process pairs across two sessions: pairs 1-3, then 4-6. This drives the
same code path (bench.trial and the same server lifecycle) and only renumbers the
pair, so scorecard.py sees six distinct pairs without a bench change.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))

import bench  # noqa: E402
import harness  # noqa: E402


def run(args) -> int:
    key = harness.model_key(args.model)
    model = harness.model_dir(key)
    if not model.is_dir():
        raise SystemExit(f"model directory {model} is missing; run `make -C campaign fetch NAMES={key}`")
    manifest = bench.load_prompts()
    cells = bench.parse_cells(args.cells)
    by_cell = bench.select(manifest, [cell for cell, _ in cells], args.prompts)
    missing = [cell for cell, _ in cells if not by_cell.get(cell)]
    if missing:
        raise SystemExit(f"no prompts for cells {missing}")
    flags, wants_drafter = bench.mode_args(args.mode)
    drafter = harness.drafter_path() if wants_drafter else None
    if wants_drafter and drafter is None:
        raise SystemExit(f"{args.mode} needs the drafter checkpoint")
    binary = Path(args.bin).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"binary {binary} is missing")
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or (f"{args.model}-{args.mode}-{'-'.join(str(c) for c, _ in cells)}-"
                             f"s{args.session}-{stamp.strftime('%H%M%S')}")
    out_dir = harness.RUNS_DIR / stamp.astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.jsonl"
    path.touch()
    hashes = harness.model_hashes(key)
    identity = {"role": "baseline", "path": str(binary), "sha256": harness.sha256_file(binary)}
    engine = f"tensorfold-native@{identity['sha256'][:12]}"
    server_args = list(flags) + (["--drafter", str(drafter)] if drafter else [])
    failures = 0
    print(f"baseline {binary} sha256 {identity['sha256']}", file=sys.stderr)
    for cell, cache_gib in cells:
        for pair in args.pair_list:
            log_path = harness.TMP / f"bench-{run_id}-{cell}-{args.mode}-{pair}-baseline.log"
            try:
                with harness.Server(binary, model, server_args, cache_gib, log_path) as server:
                    served = server.models()
                    if not served:
                        raise RuntimeError("server listed no model id")
                    for prompt in by_cell[cell]:
                        bench.trial(path, run_id=run_id, engine=engine, binary=identity, model_key=key,
                                    mode=args.mode, flags=server_args, cell=cell, cache_gib=cache_gib, pair=pair,
                                    order=0, prompt=prompt, server=server, model_id=served[0], hashes=hashes)
            except Exception as error:  # noqa: BLE001 - a dead server is a recorded failure
                failures += 1
                bench.record_failure(path, run_id=run_id, engine=engine, binary=identity, model_key=key, model=model,
                                     mode=args.mode, flags=server_args, cell=cell, cache_gib=cache_gib, pair=pair,
                                     error=f"{type(error).__name__}: {error}")
                print(f"cell {cell} pair {pair}: {type(error).__name__}: {error}", file=sys.stderr)
    print(f"appended trials to {path}" if path.stat().st_size else "no trials recorded", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--mode", default="ar", choices=sorted(bench.MODES))
    parser.add_argument("--cells", default="1024,4096,32768")
    parser.add_argument("--prompts", default="all")
    parser.add_argument("--session", default="B")
    parser.add_argument("--pairs-start", type=int, default=4)
    parser.add_argument("--pairs-stop", type=int, default=6)
    parser.add_argument("--pairs", default=None, help="explicit comma list, overriding pairs-start/stop")
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.pairs is not None:
        args.pair_list = [int(token) for token in args.pairs.split(",") if token.strip()]
        if not args.pair_list:
            parser.error("--pairs named no pair")
    else:
        if not 1 <= args.pairs_start <= args.pairs_stop:
            parser.error("need 1 <= pairs-start <= pairs-stop")
        args.pair_list = list(range(args.pairs_start, args.pairs_stop + 1))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
