"""Inkling 2 Mini DFlash2 round table: rendered prompts, gated runs of tf-inkling2-loop-run's dflash modes, tables.

usage:
  run.py prompts RUNNER OUT.json [--cells 1024,4096] [--effort 0.9]
  run.py bench RUNNER PROMPTS.json NAME [--bits 4] [--tokens 512] [--reading entering|outputs|unnormed]
  run.py widths RUNNER PROMPTS.json NAME [--reps 9] [--load SECONDS]
  run.py table NAME [NAME ...]
  run.py project BENCH WIDTHS

`bench` and `widths` run under campaign/heavy.py behind the GPU and CPU admission gates, retrying a deferral, and
write NAME.jsonl (the runner's records), NAME.log and NAME.host.json (gate, thermal and memory readings) here.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
REPO = CAMPAIGN.parent
HEAVY = CAMPAIGN / "heavy.py"
TARGET = Path.home() / "models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
DRAFTER = Path.home() / "models/Inkling-2-Mini-DFlash2@4f9088ea66c2"
TOPICS = ("code", "prose", "structured")
SYSTEM, USER, TEXT, END = 200002, 200000, 200004, 200010
GATE = ('gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; '
        'perf-gate-ready -1 || exit 75; exit "$result"')


def tokenize(runner: Path, text: str, scratch: Path) -> list[int]:
    src = scratch / "text.txt"
    out = scratch / "ids.json"
    src.write_text(text)
    subprocess.run([str(runner), str(TARGET / "tokenizer.json"), str(src), "tokenize", str(out)], check=True,
                   capture_output=True)
    return json.loads(out.read_text())


def prompts(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    scratch = REPO / ".work/tmp/df1-prompts"
    scratch.mkdir(parents=True, exist_ok=True)
    effort = tokenize(runner, f"Thinking effort level: {args.effort}", scratch)
    head = [SYSTEM, TEXT, *effort, END, USER, TEXT]
    manifest = []
    for cell in [int(c) for c in args.cells.split(",")]:
        for topic in TOPICS:
            text = (CAMPAIGN / f"prompts/dev/{topic}-{cell}.txt").read_text()
            content = tokenize(runner, text, scratch)
            keep = cell - len(head) - 1
            if len(content) < keep:
                raise SystemExit(f"{topic}-{cell} has {len(content)} content tokens, fewer than {keep}")
            ids = head + content[:keep] + [END]
            manifest.append({"name": f"{topic}-{cell}", "ids": ids})
    Path(args.out).write_text(json.dumps(manifest))
    print(f"{args.out}: " + ", ".join(f"{p['name']} {len(p['ids'])}" for p in manifest))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def readings() -> dict:
    def out(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"unavailable: {e}"
    therm = out(["pmset", "-g", "therm"])
    pressure = out(["memory_pressure"]).splitlines()
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thermal": therm,
            "thermal_warning": "No thermal warning level has been recorded" not in therm,
            "memory_pressure": pressure[-1] if pressure else "", "gpu_gate": out(["gpu-gate-ready", "-s"]),
            "perf": out(["perf-gate-ready", "-s"]), "load": os.getloadavg()}


def gated(name: str, command: list[str], env: dict[str, str]) -> None:
    log = HERE / f"{name}.log"
    host = {"before": readings(), "attempts": []}
    full_env = dict(os.environ, TMPDIR=str(REPO / ".work/tmp"), INKLING_LOAD_LIMIT_GB="26",
                    GPU_GATE_MAX_GPU_PCT="60", GPU_GATE_MAX_UTIL_PCT="60", PERF_GATE_MAX_BUSY_CORES="4", **env)
    for attempt in range(12):
        start = time.time()
        with log.open("a") as f:
            f.write(f"# attempt {attempt} {time.strftime('%H:%M:%S')}: {' '.join(command)}\n")
            f.flush()
            code = subprocess.run(["python3", str(HEAVY), "--", "sh", "-c", GATE, "sh", *command], cwd=REPO,
                                  env=full_env, stdout=f, stderr=subprocess.STDOUT).returncode
        host["attempts"].append({"exit": code, "seconds": round(time.time() - start, 1)})
        if code != 75:
            break
        time.sleep(30)
    host["after"] = readings()
    host["env"] = env
    (HERE / f"{name}.host.json").write_text(json.dumps(host, indent=2) + "\n")
    if code != 0:
        raise SystemExit(f"{name}: exit {code}; see {log}")


def bench(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    out = HERE / f"{args.name}.jsonl"
    command = [str(runner), str(TARGET), str(Path(args.prompts).resolve()), "dflash", str(DRAFTER), str(args.tokens),
               str(out), str(args.bits), args.reading]
    gated(args.name, command, {})
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "drafter": str(DRAFTER), "prompts_sha256": sha256(Path(args.prompts)), "reading": args.reading}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def widths(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    out = HERE / f"{args.name}.jsonl"
    command = [str(runner), str(TARGET), str(Path(args.prompts).resolve()), "dflash-widths", str(args.reps), str(out),
               str(args.load)]
    gated(args.name, command, {})
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "prompts_sha256": sha256(Path(args.prompts))}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def records(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / f"{name}.jsonl").read_text().splitlines() if line.strip()]


def table(args: argparse.Namespace) -> None:
    for name in args.names:
        rows = records(name)
        meta = next(r for r in rows if r["kind"] == "meta") if any(r["kind"] == "meta" for r in rows) else {}
        specs = [r for r in rows if r["kind"] == "spec"]
        plains = {r["prompt"]: r for r in rows if r["kind"] == "plain"}
        widths_rows = [r for r in rows if r["kind"] == "width"]
        if specs:
            print(f"\n{name}: drafter bits {meta.get('drafter_bits')}, taps {meta.get('taps')}, "
                  f"embed norm {meta.get('embed_norm')}, residency {meta.get('draft_residency_bytes', 0) / 2**20:.0f} MiB")
            print("| Prompt | Tokens | Equal | Rounds | Committed/round | Round mean ms | p90 | Draft | Verify | Select | "
                  "Commit | Host | Spec tok/s | Plain ms/token | Plain tok/s | Speedup |")
            print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for s in specs:
                p = plains[s["prompt"]]
                print(f"| {s['prompt']} | {s['prompt_tokens']} | {'yes' if s['equal_plain'] else 'NO'} | {s['rounds']} | "
                      f"{s['mean_committed']:.2f} | {s['round_ms_mean']:.2f} | {s['round_ms_p90']:.2f} | "
                      f"{s['draft_ms_mean']:.2f} | {s['verify_ms_mean']:.2f} | {s['select_ms_mean']:.2f} | "
                      f"{s['commit_ms_mean']:.2f} | {s['host_ms_mean']:.2f} | {s['tok_s']:.1f} | "
                      f"{p['step_ms_mean']:.2f} | {p['tok_s']:.1f} | {s['tok_s'] / p['tok_s']:.2f} |")
        if widths_rows:
            print(f"\n{name}: grouped verification from identical states")
            print("| Prompt | Context | Path | Width | p50 ms | p90 ms | min | max | Forward p50 |")
            print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for w in widths_rows:
                print(f"| {w['prompt']} | {w['context']} | {w['path']} | {w['width']} | {w['ms_p50']:.2f} | "
                      f"{w['ms_p90']:.2f} | {w['ms_min']:.2f} | {w['ms_max']:.2f} | {w['forward_ms_p50']:.2f} |")


def project(args: argparse.Namespace) -> None:
    """Each width's round from the bench's full-depth rounds (their accepted drafts cut to the width) and the widths
    run's measured verify-plus-picks cost at the prompt's context; 200 tok/s needs 5 ms a committed token."""
    rows = records(args.bench)
    cost = {(w["prompt"], w["width"], w["path"]): w["ms_p50"] for w in records(args.widths) if w["kind"] == "width"}
    print("| Prompt | Width | Committed/round | Draft | Verify+picks p50 | Commit+host | Round ms | Budget ms | tok/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in (r for r in rows if r["kind"] == "spec"):
        rounds = [r for r in rows if r["kind"] == "round" and r["prompt"] == s["prompt"] and r["offered"] == 7]
        for width in (1, 2, 4, 8):
            committed = sum(1 + min(r["accepted"], width - 1) for r in rounds) / len(rounds)
            verify = cost[(s["prompt"], width, "grouped")]
            draft = s["draft_ms_mean"] if width > 1 else 0.0
            tail = s["commit_ms_mean"] + s["host_ms_mean"]
            total = draft + verify + tail
            print(f"| {s['prompt']} | {width} | {committed:.2f} | {draft:.2f} | {verify:.2f} | {tail:.2f} | {total:.2f} | "
                  f"{5 * committed:.1f} | {1000 * committed / total:.1f} |")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("runner")
    a.add_argument("out")
    a.add_argument("--cells", default="1024,4096")
    a.add_argument("--effort", default="0.9")
    a = sub.add_parser("bench")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--bits", type=int, default=4)
    a.add_argument("--tokens", type=int, default=512)
    a.add_argument("--reading", default="entering", choices=["entering", "outputs", "unnormed"])
    a = sub.add_parser("widths")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--reps", type=int, default=9)
    a.add_argument("--load", type=float, default=0.0)
    a = sub.add_parser("table")
    a.add_argument("names", nargs="+")
    a = sub.add_parser("project")
    a.add_argument("bench")
    a.add_argument("widths")
    args = p.parse_args()
    {"prompts": prompts, "bench": bench, "widths": widths, "table": table, "project": project}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
