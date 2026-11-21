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
    cases = select(document, args.set)
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
    if before["pressure_reasons"]:
        raise SystemExit(f"host is not in a measurable state: {before['pressure_reasons']}")
    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"{args.model}-{args.set}-{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    RESULTS.mkdir(parents=True, exist_ok=True)
    jsonl = RESULTS / f"{args.model}-{args.set}.jsonl"
    jsonl.write_text("", encoding="utf-8")
    binary_info = {"path": str(binary), "sha256": harness.sha256_file(binary)}
    hashes = harness.model_hashes(key)
    log_path = harness.TMP / f"quality-{run_id}.log"
    record = {"schema": 1, "run_id": run_id, "recorded_at": stamp.isoformat(), "set": args.set,
              "model": {"key": key, "dir": str(model), "hashes": hashes}, "binary": binary_info,
              "suite": {"cases": len(cases), "manifest_sha256": harness.sha256_file(MANIFEST),
                        "cases_sha256": harness.sha256_file(CASES)},
              "protocol": {"temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS,
                           "prompt_cache_gib": PROMPT_CACHE_GIB, "drafts": "product default",
                           "sandbox": "sandbox-exec, no network, writes only in .work/quality/<run>/<case>"},
              "host": harness.host_summary(), "pressure_before": before["pressure"], "cases": []}
    started = time.monotonic()
    failures = 0
    with harness.Server(binary, model, [], PROMPT_CACHE_GIB, log_path) as server:
        served = server.models()
        if not served:
            raise SystemExit("server listed no model id")
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
    out = RESULTS / f"{args.model}-{args.set}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    totals = record["totals"]
    print(f"quality {totals['passed']}/{totals['cases']} passed; " +
          ", ".join(f"{kind} {block['passed']}/{block['cases']}" for kind, block in sorted(kinds.items())) +
          f"; wrote {out}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--set", default="dev", choices=("dev", "full"))
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN))
    parser.add_argument("--run-id", default=None)
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
