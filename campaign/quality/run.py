"""Score one model binary on the fixed quality suite and write per-case results and totals."""

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "campaign" / "bench"))

import harness  # noqa: E402
import manifest as manifest_tool  # noqa: E402
import score  # noqa: E402

CASES = HERE / "cases.json"
MANIFEST = HERE / "manifest.json"
RESULTS = HERE / "results"
MAX_TOKENS = 8192
TEMPERATURE = 0
SEED = 1234
PROMPT_CACHE_GIB = 2
REQUEST_TIMEOUT = 1800


def needs_vision(case: dict) -> bool:
    """Whether a case carries an image or other non-text content part rather than a text prompt."""
    if case.get("kind") == "vision":
        return True
    prompt = case.get("prompt")
    if isinstance(prompt, list):
        return any(isinstance(part, dict) and part.get("type") in ("image_url", "image", "input_image")
                   for part in prompt)
    return bool(case.get("image") or case.get("media") or case.get("vision"))


def applicable(cases: list[dict], key: str) -> tuple[list[dict], list[dict]]:
    """Split a selection into scored cases and the cases a text-only model excludes with a recorded reason."""
    if key not in harness.INKLING_KEYS:
        return list(cases), []
    kept, excluded = [], []
    for case in cases:
        if needs_vision(case):
            excluded.append({"id": case["id"], "kind": case["kind"],
                             "reason": "Inkling 2 Mini is text-only in this campaign: image and vision parts are "
                                       "refused (zig/src/families/inkling2_loop/refuse.zig)"})
        else:
            kept.append(case)
    return kept, excluded


def load_suite() -> tuple[dict, dict]:
    document = json.loads(CASES.read_text(encoding="utf-8"))
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest_tool.build(document) != recorded:
        raise SystemExit(f"{MANIFEST} does not match {CASES} and the retrieval prompts; run manifest.py")
    return document, recorded


def select(document: dict, name: str) -> list[dict]:
    if name not in ("dev", "full"):
        raise SystemExit("SET must be dev or full")
    cases = [case for case in document["cases"] if name in case["sets"] or (name == "full" and "full" in case["sets"])]
    if name == "dev" and (len(cases) != 12 or len({case["kind"] for case in cases}) != 4):
        raise SystemExit(f"the dev subset must hold 12 cases across four kinds, got {len(cases)}")
    if name == "full" and len(cases) != 48:
        raise SystemExit(f"the full set must hold 48 cases, got {len(cases)}")
    return cases


def grammar_library(binary: Path) -> Path:
    return binary.parent.parent / "lib" / "libtfgrammar.dylib"


def reasoning_prompt(case: dict) -> str:
    item = case["item"]
    if case["category"] == "gsm8k":
        return (f"{item['question']}\n\nSolve it. Work it out briefly, then end with a last line of the form "
                "'Answer: <number>'.")
    choices = "\n".join(f"{'ABCD'[index]}. {choice}" for index, choice in enumerate(item["choices"]))
    return f"{item['question']}\n\n{choices}\n\nReply with the letter of the correct answer only."


def request_body(case: dict, model: str) -> dict:
    if case["kind"] == "retrieval":
        prompt = (REPO / case["prompt_file"]).read_text(encoding="utf-8")
    elif case["kind"] == "reasoning":
        prompt = reasoning_prompt(case)
    else:
        prompt = case["prompt"]
    body = {"model": model, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE, "seed": SEED, "stream": False,
            "messages": [{"role": "user", "content": prompt}]}
    if case["kind"] == "structured" and case["mode"] == "json_schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": case["name"], "schema": json.loads(case["schema"])}}
    if case["kind"] == "structured" and case["mode"] == "tool_call":
        body["tools"] = case["tools"]
        body["tool_choice"] = case["tool_choice"]
        body["parallel_tool_calls"] = False
    return body


def send(base: str, body: dict) -> dict:
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        document = json.loads(response.read())
    choice = (document.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {"content": message.get("content") or "", "reasoning_content": message.get("reasoning_content") or "",
            "tool_calls": message.get("tool_calls") or [], "finish_reason": choice.get("finish_reason"),
            "usage": document.get("usage") or {}}


def prompt_sha(case: dict, body: dict) -> str:
    return harness.sha256_text(json.dumps(body["messages"], sort_keys=True))


def run(args) -> int:
    document, recorded = load_suite()
    cases, excluded = applicable(select(document, args.set), args.model)
    if not cases:
        raise SystemExit(f"every {args.set} case is excluded for {args.model}: {excluded}")
    binary = Path(args.bin).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"binary {binary} is missing")
    key = harness.model_key(args.model)
    model = harness.model_dir(key)
    if not model.is_dir():
        raise SystemExit(f"model directory {model} is missing; run `make -C campaign fetch NAMES={key}`")
    if any(case["kind"] == "structured" for case in cases):
        library = grammar_library(binary)
        if not library.is_file():
            raise SystemExit(f"{library} is missing; run `make -C campaign quality-grammar` to build libtfgrammar "
                             "from the pinned xgrammar checkout")
    before = harness.conditions()
    blocking = quality_blockers(before["pressure_reasons"])
    if blocking:
        raise SystemExit(f"host cannot hold a quality run: {blocking}")
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"{args.model}-{args.set}-{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl = run_dir / "cases.jsonl"
    jsonl.write_text("", encoding="utf-8")
    binary_info = {"path": str(binary), "sha256": harness.sha256_file(binary)}
    hashes = harness.model_hashes(key)
    log_path = harness.TMP / f"quality-{run_id}.log"
    record = {"schema": 1, "run_id": run_id, "recorded_at": stamp.isoformat(), "set": args.set,
              "model": {"key": key, "dir": str(model), "hashes": hashes}, "binary": binary_info,
              "suite": {"cases": len(cases), "manifest_sha256": harness.sha256_file(MANIFEST),
                        "cases_sha256": harness.sha256_file(CASES)},
              "protocol": {"temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS,
                           "context_tokens": harness.CONTEXT_TOKENS,
                           "prompt_cache_gib": PROMPT_CACHE_GIB, "drafts": "product default",
                           "sandbox": "sandbox-exec, no network, writes only in .work/quality/<run>/<case>"},
              "host": harness.host_summary(key), "pressure_before": before["pressure"], "cases": []}
    if excluded:
        record["excluded"] = excluded
    started = time.monotonic()
    failures = 0
    context_excluded = 0
    with harness.Server(binary, model, [], PROMPT_CACHE_GIB, log_path) as server:
        served = server.models()
        if not served:
            raise SystemExit("server listed no model id")
        if not harness.served_model_matches(key, served[0]):
            raise SystemExit(f"server at {server.base} serves {served[0]!r}, not {harness.served_identity_name(key)!r}; "
                             "the run cannot be attributed to this model")
        record["server"] = {"argv": server.argv, "served": served[0], **server.timing()}
        print(f"serving {served[0]}; startup {server.timing()}", file=sys.stderr, flush=True)
        for index, case in enumerate(cases, 1):
            body = request_body(case, served[0])
            work = harness.WORK / "quality" / run_id / case["id"]
            entry = {"id": case["id"], "kind": case["kind"], "sets": case["sets"],
                     "prompt_sha256": prompt_sha(case, body), "request": {"max_tokens": MAX_TOKENS,
                                                                         "temperature": TEMPERATURE, "seed": SEED}}
            if "response_format" in body:
                entry["request"]["response_format"] = body["response_format"]
            if "tools" in body:
                entry["request"]["tools"] = body["tools"]
                entry["request"]["tool_choice"] = body["tool_choice"]
            began = time.monotonic()
            try:
                count = harness.rendered_prompt_tokens(server.base, body)
                entry["prompt_tokens"] = count
                if problem := harness.context_problem(count, MAX_TOKENS):
                    context_excluded += 1
                    excluded.append({"id": case["id"], "kind": case["kind"], "reason": problem,
                                     "prompt_tokens": count, "max_tokens": MAX_TOKENS})
                    record["excluded"] = excluded
                    print(f"{index}/{len(cases)} {case['id']}: excluded ({problem})", file=sys.stderr, flush=True)
                    continue
                began = time.monotonic()
                reply = send(server.base, body)
            except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
                detail = getattr(error, "read", lambda: b"")()
                failures += 1
                entry.update({"seconds": round(time.monotonic() - began, 3), "passed": False,
                              "reason": f"request failed: {type(error).__name__}: {error} {detail[:300]!r}"})
            else:
                entry.update({"finish_reason": reply["finish_reason"], "usage": reply["usage"],
                              "content": reply["content"][:4000], "reasoning": reply["reasoning_content"][:1000],
                              "tool_calls": reply["tool_calls"]})
                try:
                    passed, reason = score.score(case, reply, work)
                except Exception as error:  # noqa: BLE001 - a scorer fault is a recorded failure, not a crash
                    passed, reason = False, f"scoring failed: {type(error).__name__}: {error}"
                if reply["finish_reason"] == "length":
                    passed, reason = False, f"truncated at {MAX_TOKENS} tokens: {reason}"
                entry.update({"seconds": round(time.monotonic() - began, 3), "passed": bool(passed),
                              "reason": reason})
            record["cases"].append(entry)
            with jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
            print(f"{index}/{len(cases)} {case['id']} {case['kind']}: "
                  f"{'pass' if entry['passed'] else 'fail'} ({entry['reason']})",
                  file=sys.stderr, flush=True)
    record["seconds"] = round(time.monotonic() - started, 1)
    after = harness.conditions()
    record["pressure_after"] = after["pressure"]
    record["invalid_reasons"] = after["pressure_reasons"]
    kinds = {}
    for case in record["cases"]:
        block = kinds.setdefault(case["kind"], {"cases": 0, "passed": 0, "failed": 0})
        block["cases"] += 1
        block["passed" if case["passed"] else "failed"] += 1
    record["totals"] = {"cases": len(record["cases"]), "passed": sum(1 for c in record["cases"] if c["passed"]),
                        "failed": sum(1 for c in record["cases"] if not c["passed"]), "kinds": kinds}
    record["completion"] = {"complete": len(record["cases"]) == len(cases) and failures == 0,
                            "scored": len(record["cases"]), "requested": len(cases),
                            "scoring_failures": failures,
                            "context_excluded": context_excluded,
                            "qualification": "run `quality-qualify --result <this file> --baseline <frozen result>`"}
    out = run_dir / "result.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = RESULTS / f"latest-{args.model}-{args.set}.json"
    latest.write_text(json.dumps({"run_id": run_id, "result": str(out.relative_to(HERE))}, sort_keys=True) + "\n",
                      encoding="utf-8")
    totals = record["totals"]
    print(f"quality {totals['passed']}/{totals['cases']} passed; " +
          ", ".join(f"{kind} {block['passed']}/{block['cases']}" for kind, block in sorted(kinds.items())) +
          f"; wrote {out}")
    return 0 if record["completion"]["complete"] else 1


def quality_blockers(reasons: list[str]) -> list[str]:
    """The host conditions that refuse a quality run. Greedy outputs do not depend on GPU timing, so other
    GPU users are recorded but do not refuse it; memory pressure, which can stop the server, does."""
    return [reason for reason in reasons if not reason.startswith("gpu-gate-ready")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--set", default="dev", choices=("dev", "full"))
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN))
    parser.add_argument("--run-id", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
