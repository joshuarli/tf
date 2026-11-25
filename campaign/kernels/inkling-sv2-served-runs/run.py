"""Served runs: plain decode, a loaded drafter with per-request `draft: false`, and active speculation over the same
prompts, one session a call, and the tables that pair them within sessions. The 512-token trials measure
fixed-window generation throughput, not completed answers (see `tasks.py` in inkling-q4-qualify-runs)."""

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
from bench_evidence import token_equal  # noqa: E402
from run import FOLLOW, KEY, PINS, SHORT, Log, ask, drafted, same  # noqa: E402

MODES = ("ar", "disabled", "spec")
CURVE = CAMPAIGN / "comparators" / "prefill_curve.py"
DRAFTER_LOADED = "DFlash2 drafter"


def flags_env(mode: str, drafter: dict) -> tuple[list[str], dict]:
    """`ar` names no drafter (`--no-drafts`); `disabled` and `spec` load it, and `disabled`'s requests send
    `draft: false` (`request_draft`)."""
    return {"ar": (["--no-drafts"], {}), "disabled": ([], drafter), "spec": ([], drafter)}[mode]


def request_draft(mode: str) -> bool | None:
    """The request's draft field: `disabled` turns drafting off per request; the others keep the server default."""
    return False if mode == "disabled" else None


def configuration(mode: str, model: Path, hashes: dict, drafter: dict, cache_gib: int) -> dict:
    """What produced a result. Fixed engine facts name their source; the rest are this run's settings."""
    half = os.environ.get("INKLING_PREFILL_HALF")
    return {"checkpoint": {"dir": str(model), "hashes": hashes},
            "kv_cache": "f16 keys and values (engine fixed: zig/src/families/inkling2_loop/state.zig)",
            "prefill_rounding": "half experts and inputs (INKLING_PREFILL_HALF unset)" if half is None
            else f"INKLING_PREFILL_HALF={half}",
            "decode_arithmetic": "f32 one-row kernels; grouped verify rows bit-equal to them (Decision 18)",
            "drafter": None if mode == "ar" else {"dir": drafter["INKLING_DRAFTER"], "bits": "prepared q4",
                                                   "request_draft": request_draft(mode)},
            "lookup": "prompt lookup chains in tree rounds" if mode == "spec" else None,
            "sampler": {"temperature": 0, "seed": bench.SEED, "greedy": True},
            "context": 32768, "prompt_cache_gib": cache_gib}


def gate(path: Path, engine: str, phase: str, own: set[int]) -> None:
    """One host-conditions record beside a prefill curve; a curve with reasons on either side is not valid."""
    reasons = harness.conditions(own)["pressure_reasons"]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": 1, "kind": "conditions", "engine": engine, "phase": phase,
                                 "reasons": reasons}, sort_keys=True) + "\n")


def check_drafter(server: harness.Server, mode: str) -> None:
    """The runtime state the mode claims: a loaded drafter for `disabled` and `spec`, none for `ar`."""
    loaded = DRAFTER_LOADED in Path(server.log_path).read_text(encoding="utf-8", errors="replace")
    if loaded != (mode != "ar"):
        raise SystemExit(f"{mode}: the server {'loaded' if loaded else 'did not load'} a drafter")


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
        with harness.Server(binary, model, flags, 2, harness.TMP / f"sv2-{Path(args.name).name}-{mode}.log", env) as server:
            served = server.models()[0]
            check_drafter(server, mode)
            log.row(kind="server", mode=mode, flags=named, configuration=configuration(mode, model, hashes, drafter, 2))
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
                                hashes=hashes, identity=contract, draft=request_draft(mode))
                    messages = [{"role": "user", "content": (harness.REPO / prompt["path"]).read_text(encoding="utf-8")}]
                    first = ask(server.base, served, messages, draft=mode == "spec")
                    turn = messages + [{"role": "assistant", "content": first["message"].get("content") or ""},
                                       {"role": "user", "content": FOLLOW}]
                    fork = ask(server.base, served, turn, draft=mode == "spec")
                    log.row(kind="reply", prompt=prompt["name"], phase="first", mode=mode, **first)
                    log.row(kind="reply", prompt=prompt["name"], phase="fork", mode=mode, **fork)
            settle(server)
            curve_path = HERE / f"{args.name}.curve.jsonl"
            own = harness.process_tree(server.process.pid)
            gate(curve_path, f"{args.name}-{mode}", "before", own)
            subprocess.run([sys.executable, "-I", str(CURVE), "--base", server.base, "--engine", f"{args.name}-{mode}",
                            "--lengths", args.curve, "--reps", "2", "--out", str(curve_path)], check=True)
            gate(curve_path, f"{args.name}-{mode}", "after", own)
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
        with harness.Server(binary, model, flags, 2, harness.TMP / f"sv2-{Path(args.name).name}-{mode}.log", env) as server:
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


HEX64 = frozenset("0123456789abcdef")


def digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def stream_differences(a: dict, b: dict) -> list[str]:
    """Why two recorded streams are not the same output: token and output hashes, token count, finish reasons and
    tool output. Missing or malformed evidence is a difference, never a pass."""
    why = [] if token_equal(a, b) is True else ["token_sha256 missing, unverified or different"]
    if not (digest(a.get("output_sha256")) and a.get("output_sha256") == b.get("output_sha256")):
        why.append("output_sha256 missing or different")
    count = a.get("completion_tokens")
    if not (isinstance(count, int) and count > 0 and count == b.get("completion_tokens")):
        why.append("completion_tokens missing or different")
    if not (a.get("finish_reasons") and a.get("finish_reasons") == b.get("finish_reasons")):
        why.append("finish_reasons missing or different")
    if (a.get("tool_calls"), a.get("tool_sha256")) != (b.get("tool_calls"), b.get("tool_sha256")):
        why.append("tool output different")
    return why


def differences(trial_rows: list[dict], reply_rows: list[dict], session: str) -> list[str]:
    """Every mode against `ar` in one session: cold, warm (a prompt-cache hit) and natural requests by their stream
    evidence, and the saved first and forked replies by token ids, message and finish. Invalid trials are named but
    still compared; `ar` and `idle` must draft nothing and `spec` must draft."""
    problems = []
    by = {(t["prompt"]["name"], t["mode"]): t for t in trial_rows}
    for (prompt, mode), t in sorted(by.items()):
        if not t["valid"]:
            problems.append(f"{session} {prompt} {mode}: invalid: {t['invalid_reasons']}")
        ref = by.get((prompt, "ar"))
        if ref is None:
            problems.append(f"{session} {prompt}: no ar trial to compare {mode} against")
            continue
        streams = [("cold", t["requests"].get("cold") or {}, ref["requests"].get("cold") or {}),
                   ("warm", t["requests"].get("warm") or {}, ref["requests"].get("warm") or {}),
                   ("natural", t.get("natural_termination") or {}, ref.get("natural_termination") or {})]
        for phase, req, base in streams:
            if mode != "ar":
                problems += [f"{session} {prompt} {mode} {phase}: {why}" for why in stream_differences(req, base)]
            if phase != "natural" and (mode == "spec") != bool(req.get("drafted_tokens")):
                problems.append(f"{session} {prompt} {mode} {phase}: drafted {req.get('drafted_tokens')}")
        if not (t["requests"].get("warm") or {}).get("cached_tokens"):
            problems.append(f"{session} {prompt} {mode}: warm request hit no cached tokens")
    rs = {(r["prompt"], r["phase"], r["mode"]): r for r in reply_rows}
    for (prompt, phase, mode), r in sorted(rs.items()):
        base = rs.get((prompt, phase, "ar"))
        if base is None or not same(r, base):
            problems.append(f"{session} {prompt} {phase} {mode}: reply differs from ar")
        if phase == "fork" and not r.get("cached_tokens"):
            problems.append(f"{session} {prompt} fork {mode}: no cached tokens")
        if (mode == "spec") != (drafted(r) > 0):
            problems.append(f"{session} {prompt} {phase} {mode}: drafted {drafted(r)}")
    return problems


def equal(args) -> int:
    problems = [p for name in args.names for p in differences(trials([name]), replies([name]), name)]
    mismatches = [p for p in problems if ": invalid: " not in p]
    print("\n".join(problems) if problems else f"every reply equal across modes in {', '.join(args.names)}")
    return 1 if mismatches else 0


def interval(ratios: list[float]) -> tuple[float, float, float]:
    """Geometric mean of per-session ratios and its 95% t interval (sessions are the independent unit)."""
    logs = [math.log(r) for r in ratios]
    mean = statistics.fmean(logs)
    if len(logs) < 2:
        return math.exp(mean), float("nan"), float("nan")
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(len(logs), 2.0)
    half = t * statistics.stdev(logs) / math.sqrt(len(logs))
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half)


def session_of(run_id: str) -> str:
    return Path(run_id).name


def curve_rates(names: list[str]) -> dict:
    """Per session, mode and length, the median client prefill tok/s of curves whose host conditions passed on both
    sides; ungated curves (recorded before gating) are kept but marked."""
    out: dict = {}
    for name in names:
        rows = curves([name])
        gates = {(r["engine"], r["phase"]): r["reasons"] for r in rows if r["kind"] == "conditions"}
        for mode in MODES:
            engine = f"{name}-{mode}"
            if any(gates.get((engine, ph)) for ph in ("before", "after")):
                continue
            gated = (engine, "before") in gates and (engine, "after") in gates
            for r in rows:
                if r["kind"] == "curve" and r["engine"] == engine:
                    out.setdefault((session_of(name), mode, r["target"]), {"rates": [], "gated": gated})["rates"].append(
                        r["client_prefill_tps"])
    return {k: {"tps": statistics.median(v["rates"]), "gated": v["gated"]} for k, v in out.items()}


def table(args) -> int:
    """Fixed-window generation throughput (512 tokens): decode tok/s by prompt and mode, median over sessions with
    each mode's valid count, and `spec`/`ar` and `disabled`/`ar` over valid same-session pairs only (n pairs, 95%
    interval when n >= 2). Then unseen prefill paired within sessions."""
    ts = [t for t in trials(args.names) if t["valid"]]
    by: dict = {}
    for t in ts:
        by.setdefault((t["prompt"]["name"], t["cell"]), {}).setdefault(t["mode"], {})[session_of(t["run_id"])] = t
    present = [m for m in ("ar", "disabled", "idle", "spec") if any(m in v for v in by.values())]
    print("| Prompt | " + " | ".join(f"{m} tok/s (n)" for m in present) + " | spec/ar: pairs, ratio (95%) | "
          "disabled/ar: pairs, ratio (95%) | Committed/round (spec) |")
    print("| --- | " + " | ".join("---:" for _ in present) + " | --- | --- | ---: |")
    tps = lambda t: t["requests"]["cold"].get("decode_tps")
    for (name, cell), modes in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        def med(mode, f):
            vals = [f(t) for t in modes.get(mode, {}).values() if f(t) is not None]
            return (statistics.median(vals), len(vals)) if vals else (None, 0)
        def ratio(mode):
            rs = [tps(modes[mode][s]) / tps(modes["ar"][s]) for s in modes.get(mode, {}) if s in modes.get("ar", {})]
            if not rs:
                return "0 pairs"
            g, lo, hi = interval(rs)
            return f"{len(rs)}, {g:.3f}" + (f" ({lo:.3f}–{hi:.3f})" if len(rs) >= 2 else "")
        cells = []
        for m in present:
            v, n = med(m, tps)
            cells.append(f"{v:.1f} ({n})" if v is not None else "— (0)")
        cpr, _ = med("spec", lambda t: t["requests"]["cold"].get("committed_per_verification"))
        print(f"| {name} | " + " | ".join(cells) + f" | {ratio('spec')} | {ratio('disabled')} | "
              f"{f'{cpr:.2f}' if cpr is not None else '—'} |")
    rates = curve_rates(args.names)
    print()
    print("| Unseen prefill tokens | ar tok/s (sessions) | disabled/ar: pairs, ratio (95%) | spec/ar: pairs, ratio (95%) | gated |")
    print("| ---: | ---: | --- | --- | --- |")
    for target in sorted({k[2] for k in rates}):
        sessions = sorted({k[0] for k in rates if k[2] == target})
        ar = [rates[(s, "ar", target)]["tps"] for s in sessions if (s, "ar", target) in rates]
        def paired(mode):
            rs = [rates[(s, mode, target)]["tps"] / rates[(s, "ar", target)]["tps"] for s in sessions
                  if (s, mode, target) in rates and (s, "ar", target) in rates]
            if not rs:
                return "0 pairs"
            g, lo, hi = interval(rs)
            return f"{len(rs)}, {g:.3f}" + (f" ({lo:.3f}–{hi:.3f})" if len(rs) >= 2 else "")
        gated = all(v["gated"] for k, v in rates.items() if k[2] == target)
        print(f"| {target:,} | {statistics.median(ar):.0f} ({len(ar)}) | {paired('disabled')} | {paired('spec')} | "
              f"{'yes' if gated else 'no (recorded before gating)'} |" if ar else f"| {target:,} | — | — | — | — |")
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
