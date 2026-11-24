"""Inkling 2 Mini tree verify with lookup chains (DF2b): runs of tf-inkling2-loop-run's dflash modes and the receipt's
tables.

usage:
  run.py prompts RUNNER OUT.json [--set dev|holdout] [--cells 1024,4096]
  run.py bench RUNNER PROMPTS.json NAME [--drafts chain|lookup|tree[,OPTION]|drafter_tree] [--tokens 512] [--untimed]
  run.py widths RUNNER PROMPTS.json NAME [--reps 9] [--load SECONDS] [--rest SECONDS] [--cells 0,1,8] [--timeline N]
  run.py attribution LOG ROUND[=LABEL] ...
  run.py kernels RUNNER PROMPTS.json NAME [--reps 24] [--picks PICKS.json]
  run.py calibrate NAME[,NAME...]
  run.py costs NAME[,NAME...]
  run.py pairs BEFORE[,BEFORE...] AFTER[,AFTER...]
  run.py budgets NAME[,NAME...]
  run.py same NAME,NAME[,NAME...]
  run.py kerneltable NAME
  run.py roundtable NAME[,NAME...]
  run.py widthtable NAME[,NAME...]
  run.py widthcompare BEFORE[,BEFORE...] AFTER[,AFTER...]
  run.py shapes NAME[,NAME...]
  run.py picks NAME PROMPT OUT.json

`bench` and `widths` run under campaign/heavy.py behind the GPU and CPU admission gates, retrying a deferral, and write
NAME.jsonl (the runner's records), NAME.log and NAME.host.json (gate, thermal and memory readings) here. `--untimed`
runs a bench under the GPU lock without the gates: its tokens, rounds and acceptance are deterministic, its times are
not measurements and the tables refuse them.
"""

import argparse
import hashlib
import json
import math
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
GATE = ('sleep "$REST"; gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; "$@"; result=$?; '
        'perf-gate-ready -1 || exit 75; exit "$result"')
Q_EDGES = (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9)
LENGTH_EDGES = (4, 6, 10, 18)
DEPTHS = 7


def tokenize(runner: Path, text: str, scratch: Path) -> list[int]:
    src = scratch / "text.txt"
    out = scratch / "ids.json"
    src.write_text(text)
    subprocess.run([str(runner), str(TARGET / "tokenizer.json"), str(src), "tokenize", str(out)], check=True,
                   capture_output=True)
    return json.loads(out.read_text())


def prompts(args: argparse.Namespace) -> None:
    """DF1's prompt rule (`inkling-v2-arith-runs/run.py prompts`) over the dev or holdout corpus files."""
    runner = Path(args.runner).resolve()
    scratch = REPO / ".work/tmp/df2b-prompts"
    scratch.mkdir(parents=True, exist_ok=True)
    effort = tokenize(runner, f"Thinking effort level: {args.effort}", scratch)
    head = [SYSTEM, TEXT, *effort, END, USER, TEXT]
    manifest = []
    for cell in [int(c) for c in args.cells.split(",")]:
        for topic in TOPICS:
            name = f"{topic}-{cell}" if args.set == "dev" else f"heldout-{topic}-{cell}"
            text = (CAMPAIGN / f"prompts/{args.set}/{name}.txt").read_text()
            content = tokenize(runner, text, scratch)
            keep = cell - len(head) - 1
            if len(content) < keep:
                raise SystemExit(f"{name} has {len(content)} content tokens, fewer than {keep}")
            manifest.append({"name": name, "ids": head + content[:keep] + [END]})
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


def gated(name: str, command: list[str], env: dict[str, str], rest: float = 0.0, timed: bool = True) -> None:
    """Runs `command` under heavy.py, behind the admission gates when `timed`; `rest` seconds idle inside the lock
    come first."""
    log = HERE / f"{name}.log"
    host = {"before": readings(), "attempts": [], "rest_s": rest, "timed": timed}
    full_env = dict(os.environ, TMPDIR=str(REPO / ".work/tmp"), INKLING_LOAD_LIMIT_GB="26", REST=str(rest),
                    GPU_GATE_MAX_GPU_PCT="60", GPU_GATE_MAX_UTIL_PCT="60", PERF_GATE_MAX_BUSY_CORES="4", **env)
    wrapped = ["sh", "-c", GATE, "sh", *command] if timed else command
    for attempt in range(12):
        start = time.time()
        with log.open("a") as f:
            f.write(f"# attempt {attempt} {time.strftime('%H:%M:%S')}: {' '.join(command)}\n")
            f.flush()
            code = subprocess.run(["python3", str(HEAVY), "--", *wrapped], cwd=REPO, env=full_env, stdout=f,
                                  stderr=subprocess.STDOUT).returncode
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
               str(out), "4", "entering", "incremental", args.drafts]
    gated(args.name, command, {}, timed=not args.untimed)
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "drafter": str(DRAFTER), "prompts_sha256": sha256(Path(args.prompts)), "drafts": args.drafts,
            "timed": not args.untimed}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def widths(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    out = HERE / f"{args.name}.jsonl"
    command = [str(runner), str(TARGET), str(Path(args.prompts).resolve()), "dflash-widths", str(args.reps), str(out),
               str(args.load)] + ([args.cells] if args.cells else [])
    gated(args.name, command, {"TF_METAL4_TIMELINE": str(args.timeline)} if args.timeline else {}, args.rest)
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "prompts_sha256": sha256(Path(args.prompts)), "timed": True}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def kernels(args: argparse.Namespace) -> None:
    runner = Path(args.runner).resolve()
    out = HERE / f"{args.name}.jsonl"
    command = [str(runner), str(TARGET), str(Path(args.prompts).resolve()), "dflash-kernels", str(args.reps), str(out)]
    if args.picks:
        command.append(str(Path(args.picks).resolve()))
    gated(args.name, command, {})
    meta = {"kind": "binary", "runner": str(runner), "runner_sha256": sha256(runner), "target": str(TARGET),
            "picks": args.picks, "timed": True}
    out.write_text(json.dumps(meta) + "\n" + out.read_text())


def records(name: str) -> list[dict]:
    path = HERE / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timed(name: str) -> list[dict]:
    rows = records(name)
    binary = next((r for r in rows if r["kind"] == "binary"), {})
    if not binary.get("timed", False):
        raise SystemExit(f"{name}: an untimed run; its times are not measurements")
    return rows


def qbin(q: float) -> int:
    return sum(q >= e for e in Q_EDGES)


def lbin(length: int) -> int:
    return sum(length >= e for e in LENGTH_EDGES)


def depths(parents: list[int]) -> list[int]:
    d = [0]
    for p in parents:
        d.append(d[p] + 1)
    return d


def calibrate(args: argparse.Namespace) -> None:
    """Each tree draft under an accepted parent, accepted or not: the drafter's alone by depth and probability bin, the
    lookup chain's alone in drafted rounds and in all rounds by place and match length bin, and both agreeing by
    probability bin, the priors `verify.Plan` starts from."""
    drafter = [[[0, 0] for _ in range(len(Q_EDGES) + 1)] for _ in range(DEPTHS)]
    lookup = [[[0, 0] for _ in range(len(LENGTH_EDGES) + 1)] for _ in range(15)]
    pooled = [[[0, 0] for _ in range(len(LENGTH_EDGES) + 1)] for _ in range(15)]
    agree = [[[0, 0] for _ in range(len(Q_EDGES) + 1)]]
    for name in args.names.split(","):
        for r in records(name):
            if r["kind"] != "round" or r.get("source") != "tree":
                continue
            d = depths(r["parents"])
            on, taken = [True] + [False] * r["offered"], [False] * (r["offered"] + 1)
            for j in range(r["offered"]):
                hit = (r["hits"] >> j) & 1
                parent = r["parents"][j]
                if not on[parent]:
                    continue
                if hit and not taken[parent]:
                    on[j + 1] = taken[parent] = True
                q, place = r["q"][j], r["chain"][j]
                if place >= 0:
                    pooled[place][lbin(r["match"])][0] += hit
                    pooled[place][lbin(r["match"])][1] += 1
                if q > 0 and place >= 0:
                    cell = agree[0][qbin(q)]
                elif q > 0:
                    cell = drafter[min(d[j + 1] - 1, DEPTHS - 1)][qbin(q)]
                elif r.get("drafted", True):
                    cell = lookup[place][lbin(r["match"])]
                else:
                    continue
                cell[0] += hit
                cell[1] += 1
    edges = ("<0.02",) + tuple(f">={e}" for e in Q_EDGES)
    lengths = ("3",) + tuple(f">={e}" for e in LENGTH_EDGES)
    print("drafter alone: hits/trials by depth (rows) and q bin (columns " + " ".join(edges) + ")")
    for d, row in enumerate(drafter):
        print(f"  {d}: " + "  ".join(f"{h}/{n}" for h, n in row))
    print("lookup alone: hits/trials by chain place (rows) and match length bin (columns " + " ".join(lengths) + ")")
    for j, row in enumerate(lookup):
        print(f"  {j}: " + "  ".join(f"{h}/{n}" for h, n in row))
    print("both: hits/trials by q bin: " + "  ".join(f"{h}/{n}" for h, n in agree[0]))

    def fit(table: list, weight: float = 8.0) -> list:
        """Each cell shrunk toward its bin's rate pooled over all rows (`weight` pseudo-trials), then made
        non-decreasing across bins by pooling adjacent violators, weighted by trials."""
        columns = len(table[0])
        pooled = []
        for b in range(columns):
            h = sum(row[b][0] for row in table)
            n = sum(row[b][1] for row in table)
            pooled.append((h + 1) / (n + 2))
        out = []
        for row in table:
            blocks = []
            for b in range(columns):
                w = row[b][1] + weight
                v = (row[b][0] + weight * pooled[b]) / w
                blocks.append([v, w, 1])
                while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
                    v2, w2, c2 = blocks.pop()
                    v1, w1, c1 = blocks.pop()
                    blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
            vals = [v for v, _, c in blocks for _ in range(c)]
            out.append([round(min(v, 0.995), 3) for v in vals])
        return out
    for label, table in (("drafter_prior", drafter), ("lookup_prior", lookup), ("pooled_prior", pooled),
                         ("agree_prior", agree)):
        print(f"{label}:")
        for row in fit(table):
            print("    .{ " + ", ".join(f"{v}" for v in row) + " },")


def costs(args: argparse.Namespace) -> None:
    """Measured verify ms (forward, picks, accept and commit) by verified rows over timed tree rounds, the drafter's
    ms a round, and the widths runs' p50 by width."""
    by: dict = {}
    draft: list = []
    for name in args.names.split(","):
        for r in timed(name):
            if r["kind"] == "round" and r.get("source") == "tree":
                by.setdefault(r["offered"] + 1, []).append(r["verify_ms"])
                if r["offered"] > 0:
                    draft.append(r["draft_ms"])
    print("| Rows | Rounds | Verify ms mean | p50 |")
    print("| ---: | ---: | ---: | ---: |")
    for rows in sorted(by):
        v = sorted(by[rows])
        print(f"| {rows} | {len(v)} | {sum(v) / len(v):.2f} | {v[len(v) // 2]:.2f} |")
    if draft:
        print(f"\ndraft ms a round: mean {sum(draft) / len(draft):.2f}, p50 {sorted(draft)[len(draft) // 2]:.2f}")


def spec_rows(names: str) -> dict:
    by: dict = {}
    for name in names.split(","):
        rows = timed(name)
        plains = {r["prompt"]: r for r in rows if r["kind"] == "plain"}
        for r in rows:
            if r["kind"] == "spec":
                by.setdefault(r["prompt"], []).append((name, r, plains[r["prompt"]]))
    return by


def med(values: list) -> float:
    v = sorted(values)
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def cell(values: list, digits: int = 1) -> str:
    return f"{med(values):.{digits}f} ({min(values):.{digits}f}–{max(values):.{digits}f})"


def roundtable(args: argparse.Namespace) -> None:
    """DF1's round table: the median of the processes' means, range in brackets."""
    by = spec_rows(args.names)
    print("| Prompt | Equal | Rounds | Committed/round | Round mean ms | Round p90 ms | Draft | Verify | Host | Spec tok/s | "
          "Plain ms/token | Plain tok/s | Spec/plain |")
    print("| --- | --- |" + " ---: |" * 11)
    for prompt, runs in by.items():
        specs = [r for _, r, _ in runs]
        plains = [p for _, _, p in runs]
        rounds = sorted({r["rounds"] for r in specs})
        committed = sorted({round(r["mean_committed"], 2) for r in specs})
        equal = all(r["equal_plain"] for r in specs)
        print(f"| {prompt} | {'yes' if equal else 'NO'} | {'/'.join(map(str, rounds))} | "
              f"{'/'.join(f'{c:.2f}' for c in committed)} | {cell([r['round_ms_mean'] for r in specs])} | "
              f"{cell([r['round_ms_p90'] for r in specs])} | {med([r['draft_ms_mean'] for r in specs]):.2f} | "
              f"{med([r['verify_ms_mean'] for r in specs]):.1f} | "
              f"{med([r['select_ms_mean'] + r['commit_ms_mean'] + r['host_ms_mean'] for r in specs]):.2f} | "
              f"{cell([r['tok_s'] for r in specs])} | {cell([p['step_ms_mean'] for p in plains], 2)} | "
              f"{cell([p['tok_s'] for p in plains])} | {cell([r['tok_s'] / p['tok_s'] for _, r, p in runs], 2)} |")


def pairs(args: argparse.Namespace) -> None:
    """Interleaved before/after processes paired in run order: per prompt, committed tokens a round and spec tok/s,
    each pair's ratio, and the geometric mean of ratios with a 95% interval over pairs (t, n - 1)."""
    before = [spec_rows(n) for n in args.before.split(",")]
    after = [spec_rows(n) for n in args.after.split(",")]
    if len(before) != len(after):
        raise SystemExit("pairs needs as many before runs as after runs")
    prompts_seen = list(before[0])
    print("| Prompt | Committed/round before | after | Spec tok/s before | after | After/before per pair | Geomean |")
    print("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
    all_logs = []
    for prompt in prompts_seen:
        cb, ca, tb, ta, ratios = [], [], [], [], []
        for b, a in zip(before, after):
            rb, ra = b[prompt][0][1], a[prompt][0][1]
            cb.append(rb["mean_committed"])
            ca.append(ra["mean_committed"])
            tb.append(rb["tok_s"])
            ta.append(ra["tok_s"])
            ratios.append(ra["tok_s"] / rb["tok_s"])
        logs = [math.log(x) for x in ratios]
        all_logs.extend(logs)
        print(f"| {prompt} | {med(cb):.2f} | {med(ca):.2f} | {cell(tb)} | {cell(ta)} | "
              f"{', '.join(f'{x:.2f}' for x in ratios)} | {math.exp(sum(logs) / len(logs)):.2f} |")
    n = len(all_logs)
    mean = sum(all_logs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in all_logs) / (n - 1)) if n > 1 else 0.0
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20,
         12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11}.get(n - 1, 2.0)
    half = t * sd / math.sqrt(n) if n > 1 else 0.0
    print(f"\nall prompt-pairs: geometric mean {math.exp(mean):.3f}, 95% interval {math.exp(mean - half):.3f}–"
          f"{math.exp(mean + half):.3f} (n = {n})")


def kerneltable(args: argparse.Namespace) -> None:
    """`dflash-kernels` medians: each projection kind alone at each row count, the one-row kernel against the shared one."""
    print("| Rows | Kind | One-row kernel µs | Shared kernel µs | Shared/one-row |")
    print("| ---: | --- | ---: | ---: | ---: |")
    for r in timed(args.names):
        if r["kind"] != "kernel":
            continue
        shared = f"{r['shared_us']:.1f}" if r["shared_kernel"] else f"{r['shared_us']:.1f} (one-row kernel)"
        print(f"| {r['rows']} | {r['op']} | {r['one_row_us']:.1f} | {shared} | {r['shared_us'] / r['one_row_us']:.2f} |")


def same(args: argparse.Namespace) -> None:
    """Whether processes of one build made the same rounds: per prompt, rounds, committed tokens, each round's tree
    and the output tokens' hash equal across the runs, and every run equal to plain greedy."""
    runs = [records(n) for n in args.names.split(",")]
    def key(rows: list) -> dict:
        out: dict = {}
        for r in rows:
            if r["kind"] == "spec":
                out.setdefault(r["prompt"], {})["spec"] = (r["rounds"], r["emitted"], r["tokens_sha256"], r["equal_plain"])
            if r["kind"] == "round":
                out.setdefault(r["prompt"], {}).setdefault("trees", []).append(
                    (r["offered"], r["accepted"], tuple(r["drafts"]), tuple(r.get("parents", ()))))
        return out
    keys = [key(r) for r in runs]
    bad = 0
    for prompt in keys[0]:
        ok = all(k.get(prompt) == keys[0][prompt] for k in keys[1:]) and keys[0][prompt]["spec"][3]
        bad += not ok
        rounds, emitted, digest, equal = keys[0][prompt]["spec"]
        print(f"{prompt}: {'same' if ok else 'DIFFERENT'} in {len(keys)} runs: {rounds} rounds, {emitted} tokens, "
              f"tokens {digest[:12]}, equal to plain {equal}")
    if bad:
        raise SystemExit(f"{bad} prompts differ")


def budgets(args: argparse.Namespace) -> None:
    """Decision 17: at most 20/25/30/40 ms a round at 4/5/6/8 mean committed tokens, 5 ms a committed token."""
    by = spec_rows(args.names)
    print("| Prompt | Committed/round | Round ms | Budget at that commitment ms | ms a committed token | Meets |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for prompt, runs in by.items():
        c = med([r["mean_committed"] for _, r, _ in runs])
        t = med([r["round_ms_mean"] for _, r, _ in runs])
        print(f"| {prompt} | {c:.2f} | {t:.1f} | {5 * c:.1f} | {t / c:.2f} | {'yes' if t <= 5 * c else 'no'} |")


def widthtable(args: argparse.Namespace) -> None:
    """Each width's p50 averaged over prompts and processes by context, range in brackets."""
    cells: dict = {}
    exact_bad, exact_all = 0, 0
    for name in args.names.split(","):
        for w in timed(name):
            if w["kind"] == "width":
                key = (1024 if w["context"] < 2048 else 4096, w["path"], w["width"])
                cells.setdefault(key, []).append(w)
            if w["kind"] == "exact":
                exact_all += 1
                exact_bad += w["rows_equal"] != w["width"]
    print("| Context | Path | Width | p50 ms | Forward p50 | GPU p50 |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for key in sorted(cells):
        ws = cells[key]
        print(f"| {key[0]:,} | {key[1].replace('_', ' ')} | {key[2]} | {cell([w['ms_p50'] for w in ws])} | "
              f"{med([w['forward_ms_p50'] for w in ws]):.1f} | {med([w['gpu_ms_p50'] for w in ws]):.1f} |")
    print(f"\n{exact_all - exact_bad} of {exact_all} grouped widths bit-equal to their one-row steps")


def width_cells(names: str) -> dict:
    cells: dict = {}
    for name in names.split(","):
        for w in (r for r in timed(name) if r["kind"] == "width"):
            cells.setdefault((1024 if w["context"] < 2048 else 4096, w["path"], w["width"]), []).append(w)
    return cells


def widthcompare(args: argparse.Namespace) -> None:
    """Each cell's p50 averaged over prompts and processes, before and after; widths only one side ran are listed."""
    before, after = width_cells(args.before), width_cells(args.after)
    print("| Context | Path | Width | Before p50 | After p50 | Change | After GPU p50 |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(set(before) | set(after)):
        b = [w["ms_p50"] for w in before.get(key, [])]
        a = [w["ms_p50"] for w in after.get(key, [])]
        g = [w["gpu_ms_p50"] for w in after.get(key, [])]
        bm = f"{sum(b) / len(b):.1f} ({min(b):.1f}–{max(b):.1f})" if b else "–"
        am = f"{sum(a) / len(a):.1f} ({min(a):.1f}–{max(a):.1f})" if a else "–"
        change = f"{(sum(a) / len(a)) / (sum(b) / len(b)):.2f}×" if a and b else "–"
        gpu = f"{sum(g) / len(g):.1f}" if g else "–"
        print(f"| {key[0]:,} | {key[1].replace('_', ' ')} | {key[2]} | {bm} | {am} | {change} | {gpu} |")


def shapes(args: argparse.Namespace) -> None:
    """What the trees held: rows verified, lookup rows, accepted depth and how often each width was chosen."""
    for name in args.names.split(","):
        rows = records(name)
        by: dict = {}
        for r in rows:
            if r["kind"] == "round" and r.get("source") == "tree":
                by.setdefault(r["prompt"], []).append(r)
        print(f"\n{name}")
        print("| Prompt | Rounds | Rows mean | Lookup rows mean | Accepted mean | From lookup | Rounds past 8 accepted | "
              "Widths chosen (rows: rounds) |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for prompt, rs in by.items():
            n = len(rs)
            hist: dict = {}
            for r in rs:
                hist[r["offered"] + 1] = hist.get(r["offered"] + 1, 0) + 1
            deep = sum(r["accepted"] >= 8 for r in rs)
            print(f"| {prompt} | {n} | {sum(r['offered'] + 1 for r in rs) / n:.2f} | {sum(r['looked'] for r in rs) / n:.2f} | "
                  f"{sum(r['accepted'] for r in rs) / n:.2f} | {sum(r['looked_accepted'] for r in rs) / n:.2f} | {deep} | "
                  f"{', '.join(f'{k}: {v}' for k, v in sorted(hist.items()))} |")


CATEGORIES = (
    ("routed experts, NVFP4", ("i2p_gu", "i2p_down", "i2t_gate_up", "i2t_down")),
    ("routed experts, affine 4-bit", ("i2a_expert_",)),
    ("router, pick, combine", ("i2a_float", "i2p_routepick", "i2p_combine")),
    ("attention projections", ("i2a_batch_projections", "i2t_projections")),
    ("o-projection and head", ("i2a_dense", "i2r_dense", "i2t_dense")),
    ("attention", ("i2v_split", "i2_batch_split", "i2_batch_combine", "i2v_store_rel", "i2p_batch_store_rel", "i2p_clip3",
                   "i2_sig_mul")),
    ("convolutions", ("i2v_conv", "i2p_batch_conv")),
    ("controllers", ("i2_controls_", "i2p_rms_pair", "i2a_batch_float_control")),
    ("depth mixes and norms", ("i2_batch_mix", "i2p_gather_rms", "i2p_rms", "i2p_scatter", "i2p_pass_copy", "i2_add")),
)


def timeline(text: str, name: str) -> tuple[float, dict]:
    """A named round's span and its listed kernels' ms a round, from a Metal 4 timeline log."""
    span = float(text.split(f"metal 4 timeline: {name}: ")[1].split(" ms")[0])
    section = text.split(f"metal 4 timeline: {name} by kernel")[1].split("the longest single stages")[0]
    kernels = {}
    for line in section.splitlines()[1:]:
        parts = line.replace("info:", "").split()
        if len(parts) >= 4 and parts[0].replace(".", "", 1).isdigit():
            kernels[" ".join(parts[3:])] = float(parts[0])
    return span, kernels


def attribution(args: argparse.Namespace) -> None:
    """Timeline ms a round by category; ROUND=LABEL names a round from its own log (`LOG:ROUND=LABEL` another's)."""
    cols = []
    for spec in args.rounds:
        source, _, rest = spec.rpartition(":") if ":" in spec else (args.log, "", spec)
        name, _, label = rest.partition("=")
        path = HERE / source if (HERE / source).exists() else Path(source)
        cols.append((label or name, timeline(path.read_text(), name)))
    print("| Category | " + " | ".join(label for label, _ in cols) + " |")
    print("| --- |" + " ---: |" * len(cols))
    for label, prefixes in CATEGORIES:
        cells = [sum(v for k, v in kernels.items() if k.startswith(prefixes)) for _, (_, kernels) in cols]
        print(f"| {label} | " + " | ".join(f"{v:.2f}" for v in cells) + " |")
    other = []
    for _, (span, kernels) in cols:
        listed = sum(v for k, v in kernels.items() if any(k.startswith(p) for _, ps in CATEGORIES for p in ps))
        other.append(f"{span - listed:.2f}")
    print("| other kernels and stamps | " + " | ".join(other) + " |")
    print("| round span | " + " | ".join(f"{span:.2f}" for _, (span, _) in cols) + " |")


def picks(args: argparse.Namespace) -> None:
    """A widths run's routed experts for one prompt (48 visits of its rows' picks), for `dflash-kernels`."""
    row = next(r for r in records(args.name) if r["kind"] == "picks" and r["prompt"] == args.prompt)
    Path(args.out).write_text(json.dumps(row["picks"]))
    print(f"{args.out}: {row['rows']} rows, {len(row['picks'])} picks")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("runner")
    a.add_argument("out")
    a.add_argument("--set", default="dev", choices=["dev", "holdout"])
    a.add_argument("--cells", default="1024,4096")
    a.add_argument("--effort", default="0.9")
    a = sub.add_parser("bench")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--drafts", default="tree", help="chain, lookup, tree or drafter_tree, a tree with ,nodes=N "
                   ",children=N ,skip=P")
    a.add_argument("--tokens", type=int, default=512)
    a.add_argument("--untimed", action="store_true")
    a = sub.add_parser("widths")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--reps", type=int, default=9)
    a.add_argument("--load", type=float, default=0.0)
    a.add_argument("--rest", type=float, default=0.0)
    a.add_argument("--cells", default="")
    a.add_argument("--timeline", type=int, default=0)
    a = sub.add_parser("attribution")
    a.add_argument("log")
    a.add_argument("rounds", nargs="+")
    for name in ("calibrate", "costs", "roundtable", "widthtable", "shapes", "budgets", "same", "kerneltable"):
        a = sub.add_parser(name)
        a.add_argument("names")
    a = sub.add_parser("kernels")
    a.add_argument("runner")
    a.add_argument("prompts")
    a.add_argument("name")
    a.add_argument("--reps", type=int, default=24)
    a.add_argument("--picks", default="")
    a = sub.add_parser("picks")
    a.add_argument("name")
    a.add_argument("prompt")
    a.add_argument("out")
    a = sub.add_parser("widthcompare")
    a.add_argument("before")
    a.add_argument("after")
    a = sub.add_parser("pairs")
    a.add_argument("before")
    a.add_argument("after")
    args = p.parse_args()
    {"prompts": prompts, "bench": bench, "widths": widths, "calibrate": calibrate, "costs": costs,
     "roundtable": roundtable, "pairs": pairs, "widthtable": widthtable, "shapes": shapes,
     "picks": picks, "kernels": kernels, "attribution": attribution, "budgets": budgets,
     "widthcompare": widthcompare, "same": same, "kerneltable": kerneltable}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
