"""One entrypoint for the campaign's checks and benchmarks: list, preflight, and run a tier or named workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent.parent
REPO = CAMPAIGN.parent
MAIN = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=CAMPAIGN,
                           capture_output=True, text=True, check=True).stdout.strip()).parent
WORK = MAIN / ".work"
GPU_LOCK = WORK / "gpu.lock"
BASELINE_BIN = WORK / "wt/baseline/zig-out/native/bin/tensorfold-native"
BASELINE_SHA = "8c2c8e6f4fd9f2c111d6fba7d89251a32a5afb309cc950ee3ff2d6a50816ac87"
FLINT = Path.home() / "models/flint-int4@3b93f9053b53"
INKLING = Path.home() / "models/Inkling-2-Mini@ed40e7da625f"
FLINT_RUN = REPO / "zig-out/bin/tf-flint-run"
FIXTURES = WORK / "suite/fixtures"
TIERS = ["edit", "change", "scenario", "full"]


@dataclass(frozen=True)
class Workload:
    name: str
    tier: str
    area: str
    purpose: str
    cost: str
    needs: tuple[str, ...]
    command: tuple[str, ...]


def make(*args: str) -> tuple[str, ...]:
    return ("make", "-C", str(CAMPAIGN), *args)


def zig(*args: str) -> tuple[str, ...]:
    return ("zig", "build", *args)


def flint_run(mode: str, *extra: str) -> tuple[str, ...]:
    return ("lockf", str(GPU_LOCK), str(FLINT_RUN), str(FLINT), str(FIXTURES / "code-1000.json"), mode, *extra)


WORKLOADS = [
    Workload("gate-controls", "edit", "gate", "logit gate negative controls", "observed 1 s", ("target:ref-gate-test",),
             make("ref-gate-test")),
    Workload("evidence-controls", "edit", "harness", "pairing, uncertainty, selection and qualification fixtures",
             "estimated <60 s", ("target:evidence-tests",), make("evidence-tests")),
    Workload("quality-cases", "edit", "quality", "quality case set matches its manifest", "observed 2 s",
             ("target:quality-cases-check",), make("quality-cases-check")),
    Workload("prompt-corpus", "edit", "bench", "prompt hashes and token lengths", "observed 5 s",
             ("target:prompts-check", "flint-weights"), make("prompts-check")),
    Workload("flint-host", "edit", "flint", "Flint draft config and the no-drafts switch", "observed 22 s warm",
             ("zig",), zig("test-flint-draft-host")),
    Workload("inkling-host", "edit", "inkling", "Inkling 2 Mini config, tensor map, template, refusals",
             "observed 4 s warm", ("zig",), zig("test-inkling2-loop-host")),
    Workload("flint-steps", "change", "flint", "decode step time without a server: wall, encode and GPU ms",
             "observed 1 min + lock wait", ("zig", "flint-weights", "flint-run-fresh", "flint-fixture"),
             flint_run("bench", "128")),
    Workload("flint-exact", "change", "flint", "one-row steps equal 16-row windows bit for bit in this build",
             "observed 7 s", ("zig", "flint-weights", "flint-run-fresh", "flint-fixture"),
             flint_run("exact", str(FIXTURES / "code-1000-cont.json"))),
    Workload("flint-smoke", "change", "flint", "served greedy 512 tokens with zero drafts", "observed 30 s",
             ("target:smoke", "baseline", "flint-weights"), make("smoke", "MODEL=flint", "MODE=ar")),
    Workload("flint-bench-1k", "change", "flint", "served 1,024-token cells, one process", "observed 2 min",
             ("target:bench", "baseline", "flint-weights"),
             make("bench", "MODEL=flint", "MODE=ar", "CELLS=1024", "PAIRS=1")),
    Workload("inkling-ref", "change", "inkling", "reference against llama.cpp from verified receipts",
             "observed 1 s cached", ("target:ref-check",), make("ref-check", "MODEL=inkling2-mini")),
    Workload("flint-matrix-ar", "scenario", "flint", "primary matrix, ar, six pairs", "estimated 60 min",
             ("target:bench", "baseline", "flint-weights"),
             make("bench", "MODEL=flint", "MODE=ar", "CELLS=1024,4096,32768", "PAIRS=6")),
    Workload("flint-quality-dev", "scenario", "flint", "12-case quality dev set", "observed 10 min",
             ("target:quality-flint", "baseline", "flint-weights"), make("quality", "MODEL=flint", "SET=dev")),
    Workload("flint-pi", "scenario", "flint", "8 Pi coding tasks", "observed 15 min",
             ("target:pi-harness-flint", "baseline", "flint-weights"), make("pi-harness", "MODEL=flint")),
    Workload("flint-parity", "scenario", "flint", "teacher-forced logits against rec_mlx", "estimated 60 min",
             ("target:parity-flint", "flint-weights"), make("parity", "MODEL=flint")),
    Workload("flint-quality-full", "full", "flint", "48-case quality set", "estimated 45 min",
             ("target:quality-flint", "baseline", "flint-weights"), make("quality", "MODEL=flint", "SET=full")),
    Workload("inkling-oracles", "full", "inkling", "rerun the 104 GB f32 oracles (pages heavily on 48 GiB)",
             "observed 60+ min", ("target:ref-oracles", "inkling-weights"), make("ref-check", "MODEL=inkling2-mini",
                                                                                   "REF_RERUN=1")),
    Workload("golden-suite", "full", "repo", "zig build test test-golden, milestone suite", "observed 9 min",
             ("zig",), ("lockf", str(GPU_LOCK), "zig", "build", "test", "test-golden", "-Dcpu=apple_m1")),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def newest(paths: list[Path]) -> float:
    return max((p.stat().st_mtime for root in paths for p in root.rglob("*") if p.is_file()), default=0.0)


def lock_sizes(prefix: str) -> dict[Path, int]:
    lock = json.loads((CAMPAIGN / "sources.lock.json").read_text())["inputs"]
    root = FLINT if prefix == "flint-int4" else INKLING
    return {root / i["name"].split("/", 1)[1]: i["size"] for i in lock if i["name"].startswith(prefix + "/")}


def write_fixture() -> None:
    sys.path.insert(0, str(CAMPAIGN / "prompts"))
    from flint_tokenizer import FlintTokenizer
    ids = FlintTokenizer(FLINT / "tokenizer.json").encode((CAMPAIGN / "prompts/dev/code-1024.txt").read_text())
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "code-1000.json").write_text(json.dumps(ids[:1000]))
    (FIXTURES / "code-1000-cont.json").write_text(json.dumps(ids[1000:1024]))


def defined_targets() -> set[str]:
    out = subprocess.run(["make", "-C", str(CAMPAIGN), "-pRrq", "-f", "Makefile"], capture_output=True, text=True).stdout
    return {m.group(1) for m in re.finditer(r"^([A-Za-z][\w.-]*):", out, re.M)}


def requirement(need: str, targets: set[str], fix_fixture: bool) -> str | None:
    """None when met, else the problem and its fix."""
    if need.startswith("target:"):
        name = need.split(":", 1)[1]
        return None if name in targets else f"make target `{name}` is not defined yet (its lane has not landed)"
    if need == "zig":
        pinned = json.loads((CAMPAIGN / "bootstrap.lock.json").read_text())["inputs"][0]["version"]
        found = shutil.which("zig")
        got = subprocess.run(["zig", "version"], capture_output=True, text=True).stdout.strip() if found else None
        return None if got == pinned else f"zig {got} is not the pinned {pinned}; install it"
    if need == "baseline":
        if not BASELINE_BIN.exists():
            return "frozen baseline binary missing; run `make -C campaign build-baseline`"
        got = sha256(BASELINE_BIN)
        return None if got == BASELINE_SHA else f"baseline binary sha256 {got[:12]} differs from campaign/baseline.md"
    if need in ("flint-weights", "inkling-weights"):
        sizes = lock_sizes("flint-int4" if need == "flint-weights" else "inkling-2-mini")
        bad = [p.name for p, size in sizes.items() if not p.exists() or p.stat().st_size != size]
        name = "flint-int4" if need == "flint-weights" else "inkling-2-mini"
        return None if sizes and not bad else f"{name} weights missing or wrong size ({', '.join(bad[:3])}); run `make -C campaign fetch NAMES={name}`"
    if need == "flint-run-fresh":
        if not FLINT_RUN.exists():
            return "tf-flint-run is not built; run `zig build tf-flint-run -Dcpu=apple_m1` at the repo root"
        sources = [REPO / "zig/src/families/flint", REPO / "zig/src/core", REPO / "zig/kernels/metal/flint",
                   REPO / "zig/kernels/metal/core", REPO / "zig/tests/flint"]
        stale = newest(sources) > FLINT_RUN.stat().st_mtime
        return "tf-flint-run is older than its sources; rebuild with `zig build tf-flint-run -Dcpu=apple_m1`" if stale else None
    if need == "flint-fixture":
        if fix_fixture and not (FIXTURES / "code-1000.json").exists():
            write_fixture()
        return None
    return f"unknown requirement {need}"


def host() -> list[str]:
    notes = []
    free = shutil.disk_usage(MAIN).free / 2**30
    notes.append(f"free disk {free:.0f} GiB" + (" (below 20 GiB)" if free < 20 else ""))
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    notes.append(pressure[-1] if pressure else "memory_pressure unavailable")
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    notes.append("thermal warning recorded" if "level" in therm and "No thermal warning" not in therm else "no thermal warning")
    servers = subprocess.run(["pgrep", "-fl", "tensorfold-native serve"], capture_output=True, text=True).stdout.strip()
    notes.append(f"{len(servers.splitlines())} tensorfold-native servers running (other lanes' runs are untouched)"
                 if servers else "no tensorfold-native servers running")
    holder = subprocess.run(["lsof", "-t", str(GPU_LOCK)], capture_output=True, text=True).stdout.split()
    notes.append(f"GPU lock held by pid {', '.join(holder)}; GPU workloads will wait" if holder else "GPU lock free")
    return notes


def select(args: argparse.Namespace) -> list[Workload]:
    by_name = {w.name: w for w in WORKLOADS}
    areas = {w.area for w in WORKLOADS}
    names = [n for n in args.names.split(",") if n]
    wanted_areas = [a for a in args.areas.split(",") if a]
    skips = [s for s in args.skip.split(",") if s]
    unknown = [n for n in names + skips if n not in by_name] + [a for a in wanted_areas if a not in areas]
    if unknown:
        raise SystemExit(f"unknown workload or area: {', '.join(unknown)} (names: `bench-list`; areas: {', '.join(sorted(areas))})")
    ceiling = TIERS.index(args.tier)
    if args.tier == "full" and not args.full_requested:
        raise SystemExit("the full tier runs only on an explicit request: use `make -C campaign bench-full FULL=requested`")
    chosen = [by_name[n] for n in names] if names else [w for w in WORKLOADS if TIERS.index(w.tier) <= ceiling]
    if names and not args.full_requested and any(w.tier == "full" for w in chosen):
        raise SystemExit("a named workload is in the full tier; it runs only through `bench-full FULL=requested`")
    if wanted_areas:
        chosen = [w for w in chosen if w.area in wanted_areas]
    chosen = [w for w in chosen if w.name not in skips]
    if not chosen:
        raise SystemExit("the selection matches no workload")
    return chosen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["list", "check", "run"])
    p.add_argument("--tier", default="edit", choices=TIERS)
    p.add_argument("--names", default="")
    p.add_argument("--areas", default="")
    p.add_argument("--skip", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--full-requested", action="store_true")
    args = p.parse_args()

    if args.action == "list":
        print(f"{'name':20} {'tier':9} {'area':8} {'cost':20} purpose")
        for w in WORKLOADS:
            print(f"{w.name:20} {w.tier:9} {w.area:8} {w.cost:20} {w.purpose}")
        return 0

    chosen = select(args)
    targets = defined_targets()
    problems = {w.name: [r for need in w.needs if (r := requirement(need, targets, args.action == "run" and not args.dry_run))]
                for w in chosen}
    print("host: " + "; ".join(host()))
    for w in chosen:
        print(f"{'ready  ' if not problems[w.name] else 'BLOCKED'} {w.name} ({w.tier}, {w.cost})")
        for problem in problems[w.name]:
            print(f"        {problem}")
    if any(problems.values()):
        print("preflight failed: nothing ran")
        return 1
    if args.action == "check":
        return 0
    if args.dry_run:
        for w in chosen:
            print(" ".join(w.command))
        return 0

    run_dir = WORK / "suite" / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True)
    summary = []
    for w in chosen:
        start = time.monotonic()
        with (run_dir / f"{w.name}.log").open("w") as log:
            code = subprocess.run(w.command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, TMPDIR=str(WORK / "tmp"),
                                           ZIG_GLOBAL_CACHE_DIR=str(WORK / "zig-global-cache"))).returncode
        seconds = time.monotonic() - start
        summary.append({"name": w.name, "tier": w.tier, "exit": code, "seconds": round(seconds, 1)})
        print(f"{'ok  ' if code == 0 else 'FAIL'} {w.name} {seconds:.1f} s  ({run_dir / (w.name + '.log')})")
    (run_dir / "summary.json").write_text(json.dumps({"host": host(), "workloads": summary}, indent=2) + "\n")
    print(f"summary {run_dir / 'summary.json'}; completion is not a performance verdict: read each workload's records")
    return 1 if any(s["exit"] for s in summary) else 0


if __name__ == "__main__":
    sys.exit(main())
