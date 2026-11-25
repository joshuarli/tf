"""SV2 served runs: plain decode, a loaded but disabled drafter, and active speculation over the same prompts, one
session a call, and the tables that pair them across sessions."""

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))
sys.path.insert(0, str(CAMPAIGN / "kernels" / "inkling-sv1-serve-runs"))

import bench  # noqa: E402
import harness  # noqa: E402
from run import FOLLOW, KEY, PINS, SHORT, Log, ask, drafted, same  # noqa: E402

MODES = ("ar", "idle", "spec")
CURVE = CAMPAIGN / "comparators" / "prefill_curve.py"


def flags_env(mode: str, drafter: dict) -> tuple[list[str], dict]:
    """`ar` loads no drafter; `idle` loads it and serves `--no-drafts`; `spec` loads it and drafts."""
    return {"ar": (["--no-drafts"], {}), "idle": (["--no-drafts"], drafter), "spec": ([], drafter)}[mode]


def settle(server: harness.Server, limit_s: float = 180) -> None:
    """Wait until the host passes `bench.trial`'s own checks: a fresh load leaves the memory compressor busy."""
    own = harness.process_tree(server.process.pid)
    deadline = time.monotonic() + limit_s
    while (reasons := harness.conditions(own)["pressure_reasons"]) and time.monotonic() < deadline:
        time.sleep(10)
    if reasons:
        print(f"host still unsettled after {limit_s:.0f} s: {reasons}", flush=True)


def prompts(wanted: str, cells) -> dict[int, list[dict]]:
    by_cell = bench.select(bench.load_prompts(), list(cells), wanted)
    bench.verify_prompts(by_cell)
    pins = json.loads((harness.PROMPTS_DIR / f"lengths-{PINS}.json").read_text(encoding="utf-8"))["prompts"]
    for group in by_cell.values():
        for prompt in group:
            prompt["expected_prompt_tokens"] = (pins.get(prompt["name"]) or {}).get("prompt_tokens")
    return by_cell


def session(args) -> int:
    """One session: each mode's server in turn (order rotated by `--rotate`), every prompt's cold, warm and natural
    requests through `bench.trial`, a second turn that forks the first reply's cached state, then unseen prefill to
    the cap through `prefill_curve.py`."""
    os.environ.pop("INKLING_DRAFTER", None)
    binary = Path(args.bin).expanduser().resolve()
    model = harness.model_dir(KEY)
    drafter = harness.drafter_env(KEY)
    if not drafter:
        raise SystemExit("the pinned Inkling 2 Mini DFlash2 drafter is not fetched")
    cells = [int(c) for c in args.cells.split(",")]
    by_cell = prompts(args.set, cells)
    path = HERE / f"{args.name}.jsonl"
    path.touch()
    log = Log(HERE / f"{args.name}.checks.jsonl")
    hashes, host = harness.model_hashes(KEY), harness.host_summary(KEY)
    identity = {"role": "candidate", "path": str(binary), "sha256": harness.sha256_file(binary)}
    engine = f"tensorfold-native@{identity['sha256'][:12]}"
    log.row(kind="run", binary=str(binary), binary_sha256=identity["sha256"], set=args.set, cells=cells,
            rotate=args.rotate, drafter=drafter, curve=args.curve)
    modes = MODES[args.rotate % 3:] + MODES[:args.rotate % 3]
    for order, mode in enumerate(modes):
        flags, env = flags_env(mode, drafter)
        named = flags + [f"{k}={v}" for k, v in env.items()]
        with harness.Server(binary, model, flags, 2, harness.TMP / f"sv2-{args.name}-{mode}.log", env) as server:
            served = server.models()[0]
            for cell in sorted(by_cell):
                group = by_cell[cell]
                for prompt in group:
                    if prompt["expected_prompt_tokens"] is None:
                        content = (harness.REPO / prompt["path"]).read_text(encoding="utf-8")
                        prompt["expected_prompt_tokens"] = harness.rendered_prompt_tokens(
                            server.base, {"model": served, "messages": [{"role": "user", "content": content}]})
                contract = bench.contract(key=KEY, hashes=hashes, host=host, mode=mode, flags=named, cell=cell,
                                          cache_gib=2, prompts=group, copy_drafts=False)
                contract["context_policy"]["served_model"] = served
                for prompt in group:
                    settle(server)
                    bench.trial(path, run_id=args.name, experiment="sv2", session=args.name, engine=engine,
                                binary=identity, model_key=KEY, mode=mode, flags=named, cell=cell, cache_gib=2,
                                pair=args.rotate, order=order, prompt=prompt, server=server, model_id=served,
                                hashes=hashes, identity=contract)
                    messages = [{"role": "user", "content": (harness.REPO / prompt["path"]).read_text(encoding="utf-8")}]
                    first = ask(server.base, served, messages, draft=mode == "spec")
                    turn = messages + [{"role": "assistant", "content": first["message"].get("content") or ""},
                                       {"role": "user", "content": FOLLOW}]
                    fork = ask(server.base, served, turn, draft=mode == "spec")
                    log.row(kind="reply", prompt=prompt["name"], phase="first", mode=mode, **first)
                    log.row(kind="reply", prompt=prompt["name"], phase="fork", mode=mode, **fork)
            settle(server)
            subprocess.run([sys.executable, "-I", str(CURVE), "--base", server.base, "--engine", f"{args.name}-{mode}",
                            "--lengths", args.curve, "--reps", "2", "--out", str(HERE / f"{args.name}.curve.jsonl")],
                           check=True)
    print(f"trials in {path}; replies in {log.path}", flush=True)
    return 0


SHORTS = (SHORT, "Name the capital of France in one word.", "List three primary colours, comma separated, nothing else.")


def eos(args) -> int:
    """Short prompts that end on the model's own EOS before 1,024 tokens, served by each mode: every reply must equal
    `ar`'s and stop on EOS, and only `spec` drafts."""
    os.environ.pop("INKLING_DRAFTER", None)
    binary = Path(args.bin).expanduser().resolve()
    model, drafter = harness.model_dir(KEY), harness.drafter_env(KEY)
    log = Log(HERE / f"{args.name}.checks.jsonl")
    log.row(kind="run", binary=str(binary), binary_sha256=harness.sha256_file(binary), drafter=drafter)
    got: dict = {}
    for mode in MODES:
        flags, env = flags_env(mode, drafter)
        with harness.Server(binary, model, flags, 2, harness.TMP / f"sv2-{args.name}-{mode}.log", env) as server:
            served = server.models()[0]
            for i, text in enumerate(SHORTS):
                reply = ask(server.base, served, [{"role": "user", "content": text}], draft=mode == "spec",
                            max_tokens=1024, ignore_eos=False)
                log.row(kind="reply", prompt=f"short-{i}", phase="natural", mode=mode, **reply)
                got[(i, mode)] = reply
    for i in range(len(SHORTS)):
        base = got[(i, "ar")]
        for mode in MODES:
            r = got[(i, mode)]
            log.check(same(r, base) and r["finish_reason"] == "stop" and (mode == "spec") == (drafted(r) > 0),
                      f"short-{i} {mode}: {r['completion_tokens']} tokens, {r['finish_reason']}, drafted {drafted(r)}")
    print(f"{len(log.problems)} differences; records in {log.path}", flush=True)
    return 1 if log.problems else 0


def trials(names) -> list[dict]:
    return [r for name in names for line in (HERE / f"{name}.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("kind") == "trial"]


def replies(names) -> list[dict]:
    return [r for name in names for line in (HERE / f"{name}.checks.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("kind") == "reply"]


def curves(names) -> list[dict]:
    return [json.loads(line) for name in names if (HERE / f"{name}.curve.jsonl").exists()
            for line in (HERE / f"{name}.curve.jsonl").read_text().splitlines()]


def equal(args) -> int:
    """Every mode's replies against `ar` in the same session: cold, warm (a prompt-cache hit), natural EOS and the
    forked second turn; `ar` and `idle` must draft nothing and `spec` must draft."""
    problems = []
    for name in args.names:
        by = {(t["prompt"]["name"], t["mode"]): t for t in trials([name])}
        for (prompt, mode), t in sorted(by.items()):
            if not t["valid"]:
                problems.append(f"{name} {prompt} {mode}: invalid: {t['invalid_reasons']}")
            ref = by.get((prompt, "ar"))
            for phase in ("cold", "warm"):
                req, base = t["requests"].get(phase) or {}, (ref or {}).get("requests", {}).get(phase) or {}
                if req.get("completion") != base.get("completion"):
                    problems.append(f"{name} {prompt} {mode} {phase}: reply differs from ar")
                if (mode == "spec") != bool(req.get("drafted_tokens")):
                    problems.append(f"{name} {prompt} {mode} {phase}: drafted {req.get('drafted_tokens')}")
            warm = t["requests"].get("warm") or {}
            if not warm.get("cached_tokens"):
                problems.append(f"{name} {prompt} {mode}: warm request hit no cached tokens")
            nat, base = t.get("natural_termination") or {}, (ref or {}).get("natural_termination") or {}
            if (nat.get("completion"), nat.get("finish_reasons")) != (base.get("completion"), base.get("finish_reasons")):
                problems.append(f"{name} {prompt} {mode} natural: reply or finish differs from ar")
        rs = {(r["prompt"], r["phase"], r["mode"]): r for r in replies([name])}
        for (prompt, phase, mode), r in sorted(rs.items()):
            base = rs.get((prompt, phase, "ar"))
            if base is None or not same(r, base):
                problems.append(f"{name} {prompt} {phase} {mode}: reply differs from ar")
            if phase == "fork" and not r.get("cached_tokens"):
                problems.append(f"{name} {prompt} fork {mode}: no cached tokens")
            if (mode == "spec") != (drafted(r) > 0):
                problems.append(f"{name} {prompt} {phase} {mode}: drafted {drafted(r)}")
    print("\n".join(problems) if problems else f"every reply equal across modes in {', '.join(args.names)}")
    return 1 if problems else 0


def interval(ratios: list[float]) -> tuple[float, float, float]:
    """Geometric mean of per-session ratios and its 95% t interval (sessions are the independent unit)."""
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    if len(logs) < 2:
        return math.exp(mean), float("nan"), float("nan")
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(len(logs), 2.0)
    half = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def table(args) -> int:
    """Decode tok/s by prompt and mode (median over sessions), `spec`/`ar` and `idle`/`ar` per session with a 95%
    interval, prefill tok/s (cold, engine) and peak footprint; then the default rule's verdict per prompt."""
    ts = [t for t in trials(args.names) if t["valid"]]
    by: dict = {}
    for t in ts:
        by.setdefault((t["prompt"]["name"], t["cell"]), {}).setdefault(t["mode"], {})[t["run_id"]] = t
    print("| Prompt | ar tok/s | idle tok/s | spec tok/s | spec/ar (95%) | idle/ar (95%) | Prefill tok/s ar / spec | "
          "Committed/round | Peak GB ar / spec |")
    print("| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- |")
    verdicts = []
    for (name, cell), modes in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        def med(mode, f):
            vals = [f(t) for t in modes.get(mode, {}).values() if f(t) is not None]
            return statistics.median(vals) if vals else None
        tps = lambda t: t["requests"]["cold"].get("decode_tps")
        pre = lambda t: t["requests"]["cold"].get("engine_prefill_tps")
        peak = lambda t: ((t.get("footprint") or {}).get("peak_bytes") or 0) / 1e9 or None
        cpr = lambda t: t["requests"]["cold"].get("committed_per_verification")
        def ratio(mode):
            rs = [modes[mode][s]["requests"]["cold"]["decode_tps"] / modes["ar"][s]["requests"]["cold"]["decode_tps"]
                  for s in modes.get(mode, {}) if s in modes.get("ar", {})]
            return interval(rs) if rs else None
        spec, idle = ratio("spec"), ratio("idle")
        fmt = lambda r: f"{r[0]:.3f} ({r[1]:.3f}–{r[2]:.3f})" if r else "—"
        num = lambda v, p=1: f"{v:.{p}f}" if v is not None else "—"
        print(f"| {name} | {num(med('ar', tps))} | {num(med('idle', tps))} | {num(med('spec', tps))} | {fmt(spec)} | "
              f"{fmt(idle)} | {num(med('ar', pre), 0)} / {num(med('spec', pre), 0)} | {num(med('spec', cpr), 2)} | "
              f"{num(med('ar', peak), 2)} / {num(med('spec', peak), 2)} |")
        if spec:
            verdicts.append((name, spec[1] > 1.0))
    print()
    print("| Unseen prefill tokens | ar tok/s | idle tok/s | spec tok/s | spec/ar |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    rows = [r for r in curves(args.names) if r["kind"] == "curve"]
    for target in sorted({r["target"] for r in rows}):
        med = {m: statistics.median([r["client_prefill_tps"] for r in rows
                                     if r["target"] == target and r["engine"].endswith(f"-{m}")] or [float("nan")])
               for m in MODES}
        print(f"| {target:,} | {med['ar']:.0f} | {med['idle']:.0f} | {med['spec']:.0f} | {med['spec'] / med['ar']:.3f} |")
    print()
    for name, wins in verdicts:
        print(f"{name}: spec {'beats' if wins else 'does not beat'} ar (lower bound {'>' if wins else '<='} 1)")
    print("default rule:", "met" if verdicts and all(w for _, w in verdicts) else "not met")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("session")
    p.add_argument("--bin", default=str(harness.REPO / "zig-out/native/bin/tensorfold-native"))
    p.add_argument("--name", required=True)
    p.add_argument("--set", default="qualify")
    p.add_argument("--cells", default="1024,4096")
    p.add_argument("--curve", default="1024,4096,16384,31744", help="unseen prefill lengths, one-token replies")
    p.add_argument("--rotate", type=int, default=0, help="start the mode order at this offset (0, 1 or 2)")
    p = sub.add_parser("eos")
    p.add_argument("--bin", default=str(harness.REPO / "zig-out/native/bin/tensorfold-native"))
    p.add_argument("--name", required=True)
    for action in ("equal", "table"):
        sub.add_parser(action).add_argument("names", nargs="+")
    args = parser.parse_args()
    return {"session": session, "eos": eos, "equal": equal, "table": table}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
