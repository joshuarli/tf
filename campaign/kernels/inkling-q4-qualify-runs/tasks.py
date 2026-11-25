"""Completed-task qualification: the quality suite's tasks served with EOS on and room to answer, under plain decode
and the speculative default, scored for success and format, with completion time, first-answer latency and the
reasoning/answer token split recorded, then a follow-up turn built from each completed answer.

usage:
  tasks.py run --bin BIN --name NAME [--set full|dev] [--modes ar,spec] [--prefill half|off]
  tasks.py table NAME [NAME...]
  tasks.py equal NAME [NAME...]
"""

import argparse
import datetime as dt
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))
sys.path.insert(0, str(CAMPAIGN / "quality"))
sys.path.insert(0, str(CAMPAIGN / "kernels" / "inkling-sv1-serve-runs"))

import harness  # noqa: E402
import score  # noqa: E402


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


quality = module("quality_run", CAMPAIGN / "quality" / "run.py")
served = module("sv2_run", CAMPAIGN / "kernels" / "inkling-sv2-served-runs" / "run.py")

KEY = "inkling-2-mini-oq4e-mtp"
FOLLOW_UP = "In one sentence, state the main assumption your answer relies on."
FOLLOW_UP_TOKENS = 1024
# Copy class by task kind: retrieval answers quote the prompt; code and reasoning write new text.
COPY_CLASS = {"retrieval": "copy-heavy", "code": "low-copy", "reasoning": "low-copy", "structured": "tool-continuation"}


# Tool continuation: the suite's tool cases with tool_choice auto (Inkling refuses grammar-forced choices), a fixed
# tool result, and the text the final answer must carry (compared without commas or spaces).
TOOL_RESULTS = {
    "structured-s09": ({"city": "Lisbon", "temperature_c": 17, "conditions": "clear"}, "17"),
    "structured-s10": ({"value": 7.77, "unit": "miles"}, "7.77"),
    "structured-s11": ({"results": [{"title": "Spring tides and the new moon"}, {"title": "Neap tides"},
                                    {"title": "Reading a tide table"}]}, "Neap"),
    "structured-s12": ({"city": "Lisbon", "population": 545923}, "545923"),
}


def unsupported(case: dict) -> str | None:
    """Requests the Inkling engine refuses by name, recorded as unsupported rather than as failures."""
    if case["kind"] == "structured" and case.get("mode") == "json_schema":
        return "structured output (response_format) is refused by the Inkling engine"
    if case["kind"] == "structured" and case.get("mode") == "tool_call" and case["id"] not in TOOL_RESULTS:
        return "grammar-constrained tool_choice is refused by the Inkling engine"
    return None


def reply_fields(reply: dict, seconds: float) -> dict:
    usage = reply.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    total = usage.get("completion_tokens")
    return {"content": reply["content"], "tool_calls": reply["tool_calls"], "finish_reasons": [reply["finish_reason"]],
            "completion_tokens": total, "reasoning_tokens": reasoning,
            "answer_tokens": total - reasoning if isinstance(total, int) and isinstance(reasoning, int) else None,
            "seconds": round(seconds, 3)}


def tool_task(server, model_id: str, case: dict, mode: str) -> dict:
    """A tool call scored by the suite's scorer, then the fixed tool result and a final answer that must use it."""
    body = quality.request_body(case, model_id) | {"tool_choice": "auto"}
    if (draft := served.request_draft(mode)) is not None:
        body["draft"] = draft
    began = time.monotonic()
    first = quality.send(server.base, body)
    out = {"turns": [reply_fields(first, time.monotonic() - began)]}
    passed, reason = score.score(case, first, Path("/nonexistent"))
    if not passed:
        return out | {"outcome": "fail", "reason": f"tool call: {reason}", "cap_hit": first["finish_reason"] == "length"}
    result, expected = TOOL_RESULTS[case["id"]]
    call = first["tool_calls"][0]
    turn = body["messages"] + [{"role": "assistant", "content": first["content"], "tool_calls": first["tool_calls"]},
                               {"role": "tool", "tool_call_id": call.get("id", "call-0"), "content": json.dumps(result)}]
    began = time.monotonic()
    final = quality.send(server.base, body | {"messages": turn})
    out["turns"].append(reply_fields(final, time.monotonic() - began))
    text = (final["content"] or "").replace(",", "").replace(" ", "")
    ok = final["finish_reason"] == "stop" and expected in text
    return out | {"outcome": "pass" if ok else "fail", "cap_hit": final["finish_reason"] == "length",
                  "reason": f"tool call ok; final answer {'carries' if ok else 'lacks'} {expected!r}"}


def stream_fields(run: dict) -> dict:
    """The recorded evidence of one streamed request (the fields served.stream_differences compares)."""
    keep = ("complete", "completion", "finish_reasons", "token_sha256", "output_sha256", "tool_calls", "tool_sha256",
            "completion_tokens", "prompt_tokens", "cached_tokens", "reasoning_tokens", "total_s", "decode_tps",
            "speculative")
    out = {k: run.get(k) for k in keep}
    out["completion_tokens"] = run.get("tokens")
    out["first_answer_s"] = (run.get("boundaries") or {}).get("first_content_s")
    out["first_output_s"] = (run.get("boundaries") or {}).get("first_output_s")
    reasoning = run.get("reasoning_tokens")
    out["answer_tokens"] = run["tokens"] - reasoning if isinstance(reasoning, int) and run.get("tokens") else None
    return out


def run(args) -> int:
    os.environ.pop("INKLING_DRAFTER", None)
    if args.prefill == "off":
        os.environ["INKLING_PREFILL_HALF"] = "off"
    else:
        os.environ.pop("INKLING_PREFILL_HALF", None)
    binary = Path(args.bin).expanduser().resolve()
    model = harness.model_dir(KEY)
    drafter = harness.drafter_env(KEY)
    document, _ = quality.load_suite()
    cases = quality.select(document, args.set)
    hashes = harness.model_hashes(KEY)
    path = HERE / f"{args.name}.jsonl"
    path.write_text("", encoding="utf-8")

    def row(**fields) -> None:
        fields["recorded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(fields, sort_keys=True) + "\n")

    row(kind="run", binary=str(binary), binary_sha256=harness.sha256_file(binary), set=args.set,
        prefill=args.prefill, max_tokens=quality.MAX_TOKENS, follow_up=FOLLOW_UP, cases_sha256=harness.sha256_file(
            quality.CASES))
    for mode in args.modes.split(","):
        flags, env = served.flags_env(mode, drafter)
        log = harness.TMP / f"q4-{args.name}-{mode}.log"
        with harness.Server(binary, model, flags, quality.PROMPT_CACHE_GIB, log, env) as server:
            model_id = server.models()[0]
            served.check_drafter(server, mode)
            row(kind="server", mode=mode, flags=flags + [f"{k}={v}" for k, v in env.items()],
                configuration=served.configuration(mode, model, hashes, drafter, quality.PROMPT_CACHE_GIB)
                | {"output": {"max_tokens": quality.MAX_TOKENS, "eos": "on"}})
            for case in cases:
                entry = {"kind": "task", "mode": mode, "id": case["id"], "task_kind": case["kind"],
                         "copy_class": COPY_CLASS[case["kind"]]}
                if why := unsupported(case):
                    row(**entry, outcome="unsupported", reason=why)
                    continue
                if case["id"] in TOOL_RESULTS:
                    served.settle(server)
                    began = time.monotonic()
                    entry |= tool_task(server, model_id, case, mode)
                    entry["seconds"] = round(time.monotonic() - began, 3)
                    row(**entry)
                    print(f"{mode} {case['id']}: {entry['outcome']} ({entry['reason']})", flush=True)
                    continue
                body = quality.request_body(case, model_id)
                count = harness.rendered_prompt_tokens(server.base, body)
                if problem := harness.context_problem(count, quality.MAX_TOKENS):
                    row(**entry, outcome="excluded", reason=problem, prompt_tokens=count)
                    continue
                served.settle(server)
                own = harness.process_tree(server.process.pid)
                before = harness.conditions(own)["pressure_reasons"]
                began = time.monotonic()
                first = harness.chat_request(server.base, model_id, None, messages=body["messages"],
                                             max_tokens=quality.MAX_TOKENS, ignore_eos=False, seed=quality.SEED,
                                             draft=served.request_draft(mode))
                elapsed = time.monotonic() - began
                after = harness.conditions(own)["pressure_reasons"]
                content = first["content_text"]
                capped = "length" in (first.get("finish_reasons") or [])
                passed, reason = score.score(case, {"content": content, "reasoning_content": "",
                                                    "tool_calls": []}, harness.WORK / "q4" / args.name / mode / case["id"])
                if capped:
                    passed, reason = False, f"cap hit at {quality.MAX_TOKENS} tokens: {reason}"
                entry |= {"outcome": "pass" if passed else "fail", "reason": reason, "cap_hit": capped,
                          "seconds": round(elapsed, 3), "valid": not before and not after,
                          "invalid_reasons": before + after, "answer_chars": len(content),
                          "stream": stream_fields(first)}
                if content and not capped:
                    turn = body["messages"] + [{"role": "assistant", "content": content},
                                               {"role": "user", "content": FOLLOW_UP}]
                    follow = harness.chat_request(server.base, model_id, None, messages=turn,
                                                  max_tokens=FOLLOW_UP_TOKENS, ignore_eos=False, seed=quality.SEED,
                                                  draft=served.request_draft(mode))
                    entry["follow_up"] = stream_fields(follow) | {"answer_chars": len(follow["content_text"])}
                row(**entry)
                print(f"{mode} {case['id']}: {entry['outcome']} in {entry['seconds']} s, "
                      f"{entry['stream']['completion_tokens']} tokens ({entry['stream']['reasoning_tokens']} reasoning)",
                      flush=True)
    return 0


def tasks(names) -> list[dict]:
    return [r for name in names for line in (HERE / f"{name}.jsonl").read_text().splitlines()
            if (r := json.loads(line)).get("kind") == "task"]


def equal(args) -> int:
    """Every task's first turn and follow-up under `spec` against `ar` in the same run, by stream evidence."""
    problems = []
    for name in args.names:
        by = {(t["id"], t["mode"]): t for t in tasks([name])}
        for (case, mode), t in sorted(by.items()):
            ref = by.get((case, "ar"))
            if mode == "ar" or t["outcome"] in ("unsupported", "excluded"):
                continue
            if ref is None:
                problems.append(f"{name} {case} {mode}: no ar task")
                continue
            if "turns" in t:
                keys = ("content", "tool_calls", "finish_reasons", "completion_tokens")
                if [{k: x.get(k) for k in keys} for x in t["turns"]] != [{k: x.get(k) for k in keys} for x in ref.get("turns", [])]:
                    problems.append(f"{name} {case} {mode}: tool turns differ from ar")
                continue
            problems += [f"{name} {case} {mode} first: {w}" for w in served.stream_differences(t["stream"], ref["stream"])]
            if ("follow_up" in t) != ("follow_up" in ref):
                problems.append(f"{name} {case} {mode}: follow-up present on one side only")
            elif "follow_up" in t:
                problems += [f"{name} {case} {mode} follow-up: {w}"
                             for w in served.stream_differences(t["follow_up"], ref["follow_up"])]
    print("\n".join(problems) if problems else f"every task equal across modes in {', '.join(args.names)}")
    return 1 if problems else 0


def table(args) -> int:
    """Outcomes by mode, then per scored task: success, cap hits, total seconds, first-answer latency, reasoning and
    answer tokens, by copy class. Unsupported and excluded tasks are listed, never counted as failures."""
    ts = tasks(args.names)
    for mode in sorted({t["mode"] for t in ts}):
        mine = [t for t in ts if t["mode"] == mode]
        counts = {o: sum(t["outcome"] == o for t in mine) for o in ("pass", "fail", "unsupported", "excluded")}
        caps = sum(bool(t.get("cap_hit")) for t in mine)
        invalid = sum(t.get("valid") is False for t in mine)
        print(f"{mode}: {counts['pass']}/{counts['pass'] + counts['fail']} passed, {caps} cap hits, "
              f"{counts['unsupported']} unsupported, {counts['excluded']} excluded, {invalid} timing-invalid")
    print()
    print("| Task | Copy class | Mode | Outcome | Cap hit | Seconds | First answer s | Reasoning tokens | Answer tokens | "
          "Total tokens | tok/s |")
    print("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    num = lambda v, p=1: f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"
    for t in sorted(ts, key=lambda t: (t["id"], t["mode"])):
        s = t.get("stream") or (t.get("turns") or [{}])[-1]
        print(f"| {t['id']} | {t['copy_class']} | {t['mode']} | {t['outcome']} | {'yes' if t.get('cap_hit') else ''} | "
              f"{num(t.get('seconds'))} | {num(s.get('first_answer_s'), 2)} | {s.get('reasoning_tokens', '—')} | "
              f"{s.get('answer_tokens', '—')} | {s.get('completion_tokens', '—')} | {num(s.get('decode_tps'))} |")
    print()
    print("| Copy class | Mode | Scored | Passed | Median seconds | Median first answer s | Total seconds |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cls in sorted({t["copy_class"] for t in ts}):
        for mode in sorted({t["mode"] for t in ts}):
            mine = [t for t in ts if t["copy_class"] == cls and t["mode"] == mode and t["outcome"] in ("pass", "fail")]
            if not mine:
                continue
            secs = [t["seconds"] for t in mine]
            firsts = [t["stream"]["first_answer_s"] for t in mine if (t.get("stream") or {}).get("first_answer_s") is not None]
            print(f"| {cls} | {mode} | {len(mine)} | {sum(t['outcome'] == 'pass' for t in mine)} | "
                  f"{statistics.median(secs):.1f} | {statistics.median(firsts) if firsts else float('nan'):.2f} | "
                  f"{sum(secs):.0f} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("run")
    p.add_argument("--bin", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--set", default="full", choices=("dev", "full"))
    p.add_argument("--modes", default="ar,spec")
    p.add_argument("--prefill", default="half", choices=("half", "off"))
    for action in ("table", "equal"):
        sub.add_parser(action).add_argument("names", nargs="+")
    args = parser.parse_args()
    return {"run": run, "table": table, "equal": equal}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
