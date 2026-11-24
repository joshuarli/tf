"""SV1 served runs: DFlash2 draft-on replies against draft-off replies, and one bounded single-stream timing pass."""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))

import bench  # noqa: E402
import harness  # noqa: E402

KEY = "inkling-2-mini-oq4e-mtp"
PINS = "inkling-2-mini"
CELLS = (1024, 4096)
SHORT = "Is 7 a prime number? Answer with one word."
FOLLOW = "In one sentence, which risk you named matters most?"


def dev_prompts(cells) -> dict[int, list[dict]]:
    manifest = bench.load_prompts()
    by_cell = bench.select(manifest, list(cells), "dev")
    bench.verify_prompts(by_cell)
    pins = json.loads((harness.PROMPTS_DIR / f"lengths-{PINS}.json").read_text(encoding="utf-8"))["prompts"]
    for prompts in by_cell.values():
        for prompt in prompts:
            prompt["expected_prompt_tokens"] = (pins.get(prompt["name"]) or {}).get("prompt_tokens")
    return by_cell


def ask(base: str, model: str, messages: list[dict], *, draft: bool, max_tokens: int = bench.OUTPUT_TOKENS,
        ignore_eos: bool = True) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": bench.SEED,
            "ignore_eos": ignore_eos, "draft": draft, "return_token_ids": True, "stream": False}
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.loads(response.read())
    choice = reply["choices"][0]
    runtime = reply.get("tensorfold") or {}
    usage = reply.get("usage") or {}
    return {"draft": draft, "max_tokens": max_tokens, "ignore_eos": ignore_eos,
            "seconds": round(time.perf_counter() - sent, 3), "message": choice["message"],
            "finish_reason": choice.get("finish_reason"), "token_ids": runtime.get("token_ids"),
            "token_sha256": runtime.get("token_sha256"), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "speculative": reply.get("speculative")}


def same(a: dict, b: dict) -> bool:
    text = json.dumps(a["message"], sort_keys=True).encode() == json.dumps(b["message"], sort_keys=True).encode()
    return bool(a["token_ids"]) and a["token_ids"] == b["token_ids"] and text and a["finish_reason"] == b["finish_reason"]


def drafted(reply: dict) -> int:
    return (reply.get("speculative") or {}).get("drafted") or 0


class Log:
    def __init__(self, path: Path):
        self.path, self.problems = path, []
        path.write_text("", encoding="utf-8")

    def row(self, **fields) -> None:
        fields["recorded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(fields, sort_keys=True) + "\n")

    def check(self, ok: bool, what: str) -> None:
        print(f"{'equal' if ok else 'DIFFERS'}: {what}", flush=True)
        self.row(kind="check", ok=ok, what=what)
        if not ok:
            self.problems.append(what)


def equal(args) -> int:
    os.environ.pop("INKLING_DRAFTER", None)
    binary = Path(args.bin).expanduser().resolve()
    model = harness.model_dir(KEY)
    env = harness.drafter_env(KEY)
    if not env:
        raise SystemExit("the pinned Inkling 2 Mini DFlash2 drafter is not fetched")
    log = Log(HERE / f"{args.name}.jsonl")
    log.row(kind="run", binary=str(binary), binary_sha256=harness.sha256_file(binary), model=str(model), env=env)
    by_cell = dev_prompts(CELLS)
    on_replies = {}
    with harness.Server(binary, model, [], 2, harness.TMP / f"sv1-{args.name}-on.log", env) as server:
        served = server.models()[0]
        for cell in CELLS:
            for prompt in by_cell[cell]:
                name = prompt["name"]
                messages = [{"role": "user", "content": (harness.REPO / prompt["path"]).read_text(encoding="utf-8")}]
                replies = {"on": ask(server.base, served, messages, draft=True),
                           "off": ask(server.base, served, messages, draft=False),
                           "again": ask(server.base, served, messages, draft=True),
                           "natural_on": ask(server.base, served, messages, draft=True, ignore_eos=False),
                           "natural_off": ask(server.base, served, messages, draft=False, ignore_eos=False)}
                for phase, reply in replies.items():
                    log.row(kind="reply", prompt=name, phase=phase, server="drafter", **reply)
                on_replies[name] = replies["on"]
                log.check(same(replies["on"], replies["off"]), f"{name}: 512 tokens, draft on (cold) against off")
                log.check(same(replies["again"], replies["off"]) and (replies["again"]["cached_tokens"] or 0) > 0,
                          f"{name}: draft on with a prompt-cache hit of {replies['again']['cached_tokens']} tokens against off")
                log.check(same(replies["natural_on"], replies["natural_off"]),
                          f"{name}: EOS kept ({replies['natural_on']['finish_reason']}), draft on against off")
                log.check(drafted(replies["on"]) > 0, f"{name}: the draft-on reply drafted {drafted(replies['on'])} tokens")
                if name == "code-1024":
                    turn = messages + [{"role": "assistant", "content": replies["on"]["message"].get("content") or ""},
                                       {"role": "user", "content": FOLLOW}]
                    follow_on = ask(server.base, served, turn, draft=True)
                    follow_off = ask(server.base, served, turn, draft=False)
                    log.row(kind="reply", prompt=name, phase="follow_on", server="drafter", **follow_on)
                    log.row(kind="reply", prompt=name, phase="follow_off", server="drafter", **follow_off)
                    log.check(same(follow_on, follow_off) and (follow_on["cached_tokens"] or 0) > 0,
                              f"{name}: a second turn reusing {follow_on['cached_tokens']} cached tokens, draft on against off")
        short = [{"role": "user", "content": SHORT}]
        short_on = ask(server.base, served, short, draft=True, max_tokens=1024, ignore_eos=False)
        short_off = ask(server.base, served, short, draft=False, max_tokens=1024, ignore_eos=False)
        log.row(kind="reply", prompt="short", phase="natural_on", server="drafter", **short_on)
        log.row(kind="reply", prompt="short", phase="natural_off", server="drafter", **short_off)
        log.check(same(short_on, short_off) and short_on["finish_reason"] == "stop",
                  f"short: natural EOS after {short_on['completion_tokens']} tokens ({short_on['finish_reason']}), draft on against off")
    with harness.Server(binary, model, [], 2, harness.TMP / f"sv1-{args.name}-off.log", {}) as server:
        served = server.models()[0]
        for cell in CELLS:
            for prompt in by_cell[cell]:
                name = prompt["name"]
                messages = [{"role": "user", "content": (harness.REPO / prompt["path"]).read_text(encoding="utf-8")}]
                plain = ask(server.base, served, messages, draft=True)
                log.row(kind="reply", prompt=name, phase="no_drafter", server="no-drafter", **plain)
                log.check(same(plain, on_replies[name]) and drafted(plain) == 0,
                          f"{name}: a server without INKLING_DRAFTER against the drafter server's draft-on reply")
    print(f"{len(log.problems)} differences; records in {log.path}", flush=True)
    return 1 if log.problems else 0


def against(args) -> int:
    os.environ.pop("INKLING_DRAFTER", None)
    binary = Path(args.bin).expanduser().resolve()
    model = harness.model_dir(KEY)
    rows = [json.loads(line) for line in (HERE / f"{args.reference}.jsonl").read_text(encoding="utf-8").splitlines()]
    drafted_on = {r["prompt"]: r for r in rows if r.get("kind") == "reply" and r.get("phase") == "on"}
    log = Log(HERE / f"{args.name}.jsonl")
    sha = harness.sha256_file(binary)
    log.row(kind="run", binary=str(binary), binary_sha256=sha, model=str(model), reference=args.reference)
    by_cell = dev_prompts(CELLS)
    with harness.Server(binary, model, [], 2, harness.TMP / f"sv1-{args.name}.log", {}) as server:
        served = server.models()[0]
        for cell in CELLS:
            for prompt in by_cell[cell]:
                name = prompt["name"]
                messages = [{"role": "user", "content": (harness.REPO / prompt["path"]).read_text(encoding="utf-8")}]
                plain = ask(server.base, served, messages, draft=True)
                log.row(kind="reply", prompt=name, phase="reference", server=sha[:12], **plain)
                log.check(same(plain, drafted_on[name]) and drafted(plain) == 0,
                          f"{name}: tensorfold-native {sha[:12]} against {args.reference}'s draft-on reply")
    print(f"{len(log.problems)} differences; records in {log.path}", flush=True)
    return 1 if log.problems else 0


def timing(args) -> int:
    os.environ.pop("INKLING_DRAFTER", None)
    binary = Path(args.bin).expanduser().resolve()
    model = harness.model_dir(KEY)
    env = harness.drafter_env(KEY)
    if not env:
        raise SystemExit("the pinned Inkling 2 Mini DFlash2 drafter is not fetched")
    cells = [int(c) for c in args.cells.split(",")]
    by_cell = dev_prompts(cells)
    path = HERE / f"{args.name}.jsonl"
    path.touch()
    hashes = harness.model_hashes(KEY)
    host = harness.host_summary(KEY)
    identity = {"role": "candidate", "path": str(binary), "sha256": harness.sha256_file(binary)}
    engine = f"tensorfold-native@{identity['sha256'][:12]}"
    modes = [("ar", ["--no-drafts"], {}), ("drafter", [], env)]
    for index, cell in enumerate(cells):
        for order, (mode, flags, extra) in enumerate(modes if index % 2 == 0 else list(reversed(modes))):
            named = flags + [f"{k}={v}" for k, v in extra.items()]
            log = harness.TMP / f"sv1-{args.name}-{cell}-{mode}.log"
            with harness.Server(binary, model, flags, 2, log, extra) as server:
                served = server.models()[0]
                for prompt in by_cell[cell]:
                    if prompt["expected_prompt_tokens"] is None:
                        content = (harness.REPO / prompt["path"]).read_text(encoding="utf-8")
                        prompt["expected_prompt_tokens"] = harness.rendered_prompt_tokens(
                            server.base, {"model": served, "messages": [{"role": "user", "content": content}]})
                        prompt["length_source"] = "measured by this server's /v1/tokenize"
                contract = bench.contract(key=KEY, hashes=hashes, host=host, mode=mode, flags=named, cell=cell,
                                          cache_gib=2, prompts=by_cell[cell], copy_drafts=False)
                contract["context_policy"]["served_model"] = served
                for prompt in by_cell[cell]:
                    bench.trial(path, run_id=args.name, experiment=args.name, session=args.name, engine=engine,
                                binary=identity, model_key=KEY, mode=mode, flags=named, cell=cell, cache_gib=2,
                                pair=1, order=order, prompt=prompt, server=server, model_id=served, hashes=hashes,
                                identity=contract)
    print(f"trials in {path}", flush=True)
    return 0


def table(args) -> int:
    rows = [json.loads(line) for name in args.names for line in (HERE / f"{name}.jsonl").read_text().splitlines()]
    trials = [r for r in rows if r.get("kind") == "trial"]
    by = {(t["prompt"]["name"], t["mode"]): t for t in trials}
    print("| Prompt | Prompt tokens | Off cold tok/s | On cold tok/s | On/off | Committed/round | Round ms | "
          "Draft ms | Verify ms | Commit ms | Accepted/drafted | Valid |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for name in sorted({t["prompt"]["name"] for t in trials}, key=lambda n: (int(n.rsplit("-", 1)[1]), n)):
        off, on = by.get((name, "ar")), by.get((name, "drafter"))
        if not off or not on:
            continue
        a, b = off["requests"]["cold"], on["requests"]["cold"]
        spec = b.get("speculative") or {}
        proposer = spec.get("proposer") or {}
        rounds = spec.get("rounds") or 0
        spec_rounds = proposer.get("rounds") or 0
        phase = lambda key: proposer.get(key, 0) / spec_rounds if spec_rounds else float("nan")  # noqa: E731
        print(f"| {name} | {b.get('prompt_tokens')} | {a['decode_tps']:.1f} | {b['decode_tps']:.1f} | "
              f"{b['decode_tps'] / a['decode_tps']:.2f} | {(b['tokens'] - 1) / rounds:.2f} | "
              f"{1000 * b['decode_s'] / rounds:.1f} | {phase('draft_ms'):.1f} | {phase('verify_ms'):.1f} | "
              f"{phase('commit_ms'):.2f} | {spec.get('accepted')}/{spec.get('drafted')} | "
              f"{off['valid'] and on['valid']} |")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("equal", "against", "timing"):
        p = sub.add_parser(action)
        p.add_argument("--bin", default=str(harness.REPO / "zig-out/native/bin/tensorfold-native"))
        p.add_argument("--name", required=True)
        if action == "against":
            p.add_argument("--reference", required=True)
        if action == "timing":
            p.add_argument("--cells", default="1024,4096")
    p = sub.add_parser("table")
    p.add_argument("names", nargs="+")
    args = parser.parse_args()
    return {"equal": equal, "against": against, "timing": timing, "table": table}[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
