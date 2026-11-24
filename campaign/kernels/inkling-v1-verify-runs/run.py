"""Inkling 2 Mini grouped verify (V1): gated runs of tf-inkling2-loop-run's dflash modes and the receipt's tables.

usage:
  run.py bench RUNNER PROMPTS.json NAME [--bits 4] [--tokens 512] [--reading entering|outputs|unnormed]
  run.py widths RUNNER PROMPTS.json NAME [--reps 9] [--load SECONDS] [--timeline ROUNDS]
  run.py table NAME [NAME ...]
  run.py compare BEFORE[,BEFORE...] AFTER[,AFTER...]
  run.py budgets BENCH[,BENCH...]
  run.py attribution LOG [ROUND ...]
  run.py roundtable NAME,NAME,NAME [--against DIR/NAME,...]

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
    gated(args.name, command, {"TF_METAL4_TIMELINE": str(args.timeline)} if args.timeline else {})
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
            print("| Prompt | Context | Path | Width | p50 ms | p90 ms | min | max | Forward p50 | GPU p50 |")
            print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for w in widths_rows:
                print(f"| {w['prompt']} | {w['context']} | {w['path']} | {w['width']} | {w['ms_p50']:.2f} | "
                      f"{w['ms_p90']:.2f} | {w['ms_min']:.2f} | {w['ms_max']:.2f} | {w['forward_ms_p50']:.2f} | "
                      f"{w.get('gpu_ms_p50', 0):.2f} |")
            exact = [r for r in rows if r["kind"] == "exact"]
            if exact:
                bad = [r for r in exact if r["rows_equal"] != r["width"]]
                print(f"\n{name}: {len(exact) - len(bad)} of {len(exact)} grouped widths bit-equal to one-row steps")


def width_cells(names: str) -> dict:
    cells: dict = {}
    for name in names.split(","):
        for w in (r for r in records(name) if r["kind"] == "width"):
            key = (1024 if w["context"] < 2048 else 4096, w["path"], w["width"])
            cells.setdefault(key, []).append(w)
    return cells


def compare(args: argparse.Namespace) -> None:
    """Each cell's p50 averaged over its prompts and processes, range over prompts and processes in brackets."""
    before, after = width_cells(args.before), width_cells(args.after)
    print("| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(before):
        b = [w["ms_p50"] for w in before[key]]
        a = [w["ms_p50"] for w in after.get(key, [])]
        if not a:
            continue
        g = [w.get("gpu_ms_p50", 0) for w in after[key]]
        bm, am = sum(b) / len(b), sum(a) / len(a)
        print(f"| {key[0]:,} | {key[1].replace('_', ' ')} | {key[2]} | {bm:.1f} ({min(b):.1f}–{max(b):.1f}) | "
              f"{am:.1f} ({min(a):.1f}–{max(a):.1f}) | {am / bm:.2f}× | {sum(g) / len(g):.1f} |")


def budgets(args: argparse.Namespace) -> None:
    """Decision 17: at most 20/25/30/40 ms a round at 4/5/6/8 mean committed tokens; 200 tok/s is 5 ms a token."""
    limits = {4: 20.0, 5: 25.0, 6: 30.0, 8: 40.0}
    rows = [r for name in args.benches.split(",") for r in records(name) if r["kind"] == "spec"]
    by = {}
    for r in rows:
        by.setdefault(r["prompt"], []).append(r)
    print("| Prompt | Committed/round | Round ms | Budget at that commitment ms | ms a committed token | Meets |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for prompt, rs in by.items():
        c = sum(r["mean_committed"] for r in rs) / len(rs)
        t = sorted(r["round_ms_mean"] for r in rs)[len(rs) // 2]
        budget = 5.0 * c
        print(f"| {prompt} | {c:.2f} | {t:.1f} | {budget:.1f} | {t / c:.2f} | {'yes' if t <= budget else 'no'} |")
    print("\nDecision 17 budgets: " + ", ".join(f"{k} committed: {v:.0f} ms" for k, v in limits.items()))



CATEGORIES = (
    ("routed experts, NVFP4", ("i2p_gu", "i2p_down", "i2_gu_rows", "i2_down_rows", "i2g_")),
    ("routed experts, affine 4-bit", ("i2a_expert_",)),
    ("router, pick, combine", ("i2a_float", "i2p_routepick", "i2p_combine")),
    ("attention projections", ("i2a_batch_projections", "i2r_projections")),
    ("o-projection and head", ("i2a_dense", "i2r_dense")),
    ("attention", ("i2v_split", "i2_batch_split", "i2_batch_combine", "i2v_store_rel", "i2p_batch_store_rel", "i2p_clip3",
                   "i2_sig_mul")),
    ("convolutions", ("i2v_conv", "i2p_batch_conv")),
    ("controllers", ("i2_controls_", "i2p_rms_pair", "i2a_batch_float_control")),
    ("depth mixes and norms", ("i2_batch_mix", "i2p_gather_rms", "i2p_rms", "i2p_scatter", "i2p_pass_copy", "i2_add")),
)


def timeline(text: str, name: str) -> tuple[float, dict, float]:
    """A round's span, its listed kernels' ms a round, and its last stage (the head) in ms."""
    span = float(text.split(f"metal 4 timeline: {name}: ")[1].split(" ms")[0])
    section = text.split(f"metal 4 timeline: {name} by kernel")[1].split("the longest single stages")[0]
    kernels = {}
    for line in section.splitlines()[1:]:
        parts = line.replace("info:", "").split()
        if len(parts) >= 4 and parts[0].replace(".", "", 1).isdigit():
            kernels[" ".join(parts[3:])] = float(parts[0])
    stages = text.split(f"metal 4 timeline: {name}: the longest single stages")[1].split("metal 4 timeline:")[0]
    last = max((int(l.split()[2]), float(l.split()[6])) for l in stages.splitlines() if "stage" in l and "of encoder" in l)
    head = [float(l.split()[6]) for l in stages.splitlines() if "stage" in l and f"stage {last[0]:>4}" in l][0]
    return span, kernels, head / 1e3


def attribution(args: argparse.Namespace) -> None:
    """Timeline ms a round by category for each named round; the head is the round's last stage."""
    text = (HERE / args.log).read_text() if (HERE / args.log).exists() else Path(args.log).read_text()
    names = args.rounds or ["inkling2-batch", "inkling2-group-1", "inkling2-group-2", "inkling2-group-4", "inkling2-group-8"]
    cols = {n: timeline(text, n) for n in names}
    print("| Category | " + " | ".join(n.replace("inkling2-", "") for n in names) + " |")
    print("| --- |" + " ---: |" * len(names))
    for label, prefixes in CATEGORIES:
        cells = []
        for n in names:
            span, kernels, head = cols[n]
            v = sum(ms for k, ms in kernels.items() if k.startswith(prefixes))
            if label == "o-projection and head":
                cells.append(f"{v - head:.2f} + {head:.2f}")
            else:
                cells.append(f"{v:.2f}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    other = []
    for n in names:
        span, kernels, head = cols[n]
        listed = sum(ms for k, ms in kernels.items() if any(k.startswith(p) for _, ps in CATEGORIES for p in ps))
        other.append(f"{span - listed:.2f}")
    print("| other kernels and stamps | " + " | ".join(other) + " |")
    print("| round span | " + " | ".join(f"{cols[n][0]:.2f}" for n in names) + " |")


def spec_rows(names: str) -> dict:
    by: dict = {}
    for name in names.split(","):
        path = (HERE / f"{name}.jsonl").resolve()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        plains = {r["prompt"]: r for r in rows if r["kind"] == "plain"}
        for r in rows:
            if r["kind"] == "spec":
                by.setdefault(r["prompt"], []).append((r, plains[r["prompt"]]))
    return by


def roundtable(args: argparse.Namespace) -> None:
    """DF1's round table: the median of the processes' means, range in brackets; --against adds its spec tok/s."""
    by = spec_rows(args.names)
    against = spec_rows(args.against) if args.against else {}

    def med(values: list) -> float:
        return sorted(values)[len(values) // 2]

    def cell(values: list, digits: int = 1) -> str:
        return f"{med(values):.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"
    head = ("| Prompt | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | "
            "Plain ms/token | Plain tok/s | Spec/plain |")
    print(head + (" Spec tok/s before |" if against else ""))
    print("| --- |" + " ---: |" * (11 + bool(against)))
    for prompt, pairs in by.items():
        specs = [r for r, _ in pairs]
        plains = [p for _, p in pairs]
        rounds = {r["rounds"] for r in specs}
        committed = {round(r["mean_committed"], 4) for r in specs}
        line = (f"| {prompt} | {'/'.join(str(v) for v in sorted(rounds))} | "
                f"{'/'.join(f'{v:.2f}' for v in sorted(committed))} | "
                f"{cell([r['round_ms_mean'] for r in specs])} | {cell([r['round_ms_p90'] for r in specs])} | "
                f"{med([r['draft_ms_mean'] for r in specs]):.2f} | {med([r['verify_ms_mean'] for r in specs]):.1f} | "
                f"{med([r['select_ms_mean'] + r['commit_ms_mean'] + r['host_ms_mean'] for r in specs]):.2f} | "
                f"{cell([r['tok_s'] for r in specs])} | {cell([p['step_ms_mean'] for p in plains], 2)} | "
                f"{cell([p['tok_s'] for p in plains])} | "
                f"{cell([r['tok_s'] / p['tok_s'] for r, p in pairs], 2)} |")
        if against:
            line += f" {cell([r['tok_s'] for r, _ in against.get(prompt, [])])} |"
        print(line)


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
    a.add_argument("--timeline", type=int, default=0)
    a = sub.add_parser("compare")
    a.add_argument("before")
    a.add_argument("after")
    a = sub.add_parser("budgets")
    a.add_argument("benches")
    a = sub.add_parser("roundtable")
    a.add_argument("names")
    a.add_argument("--against", default="")
    a = sub.add_parser("attribution")
    a.add_argument("log")
    a.add_argument("rounds", nargs="*")
    a = sub.add_parser("table")
    a.add_argument("names", nargs="+")
    a = sub.add_parser("project")
    a.add_argument("bench")
    a.add_argument("widths")
    args = p.parse_args()
    {"prompts": prompts, "bench": bench, "widths": widths, "table": table, "project": project, "compare": compare,
     "budgets": budgets, "attribution": attribution,
     "roundtable": roundtable}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
