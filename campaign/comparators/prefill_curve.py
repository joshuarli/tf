"""Measure unseen-prompt prefill across lengths on a running OpenAI-compatible server.

Each request carries a slice of the 32K corpus that no earlier request in the run has sent, with a
distinct leading line so no engine can reuse a prefix past the chat template. `max_tokens` is 1, so
time to first token is prefill plus one step. The first request after readiness is recorded as
`first` and kept out of the curve. The shortest length then takes one prompt from each corpus file,
which calibrates that file's characters per token; every longer length sends --reps unseen prompts,
each from its own reserved span so a resize reuses that span.

usage: prefill_curve.py --base URL --engine NAME [--lengths 1024,4096,...] --out FILE.jsonl
"""

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCES = [CAMPAIGN / "prompts" / "dev" / f"{t}-32768.txt" for t in ("code", "prose", "structured")] + \
          [CAMPAIGN / "prompts" / "holdout" / f"heldout-{t}-32768.txt" for t in ("code", "prose", "structured")]
CAP = 32768


class Corpus:
    """Unused slices of the six corpus files. Each request reserves its own span, so a resize reuses that span."""

    SLACK = 1.12

    def __init__(self):
        self.texts = [p.read_text(encoding="utf-8") for p in SOURCES]
        self.offsets = [0] * len(self.texts)
        self.ratios = [3.6] * len(self.texts)

    def remaining_tokens(self, i: int) -> float:
        return (len(self.texts[i]) - self.offsets[i]) / self.ratios[i]

    def reserve(self, target: int, source: int | None = None) -> tuple[int, int, int]:
        if source is None:
            source = max(range(len(self.texts)), key=self.remaining_tokens)
        chars = int(target * self.ratios[source] * self.SLACK)
        start = self.offsets[source]
        if len(self.texts[source]) - start < chars:
            raise SystemExit(f"the corpus has no unused {target}-token span left")
        self.offsets[source] = start + chars
        return source, start, chars

    def text(self, span: tuple[int, int, int], target: int) -> tuple[str, str]:
        source, start, reserved = span
        chars = min(reserved, int(target * self.ratios[source]))
        return self.texts[source][start:start + chars], f"{SOURCES[source].name}[{start}:{start + chars}]"


def request(base: str, model: str, content: str) -> dict:
    body = {"model": model, "max_tokens": 1, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": content}]}
    sent = time.perf_counter()
    raw = urllib.request.urlopen(urllib.request.Request(f"{base}/v1/chat/completions", json.dumps(body).encode(),
                                                        {"Content-Type": "application/json"}), timeout=1800).read()
    wall = time.perf_counter() - sent
    reply = json.loads(raw)
    usage = reply.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {"wall_s": wall, "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens") or (reply.get("timings") or {}).get("cache_n") or 0,
            "timings": reply.get("timings")}


def model_id(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as response:
        return json.loads(response.read())["data"][0]["id"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--lengths", default="1024,4096,8192,16384,31744")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    if max(lengths) + 1 > CAP:
        raise SystemExit(f"a {max(lengths)}-token prompt plus its reply exceeds the {CAP}-token cap")
    model = model_id(args.base)
    corpus = Corpus()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sources = range(len(SOURCES))
    calibration = [("first", lengths[0], 0, corpus.reserve(lengths[0], 0))] + \
        [("curve", lengths[0], i, corpus.reserve(lengths[0], i)) for i in sources]
    with out.open("a", encoding="utf-8") as stream:
        def send(kind: str, target: int, rep: int, span: tuple[int, int, int]) -> None:
            for attempt in range(3):
                text, source = corpus.text(span, target)
                content = f"Request {kind}-{target}-{rep}-{time.time_ns()}. Summarize the following.\n\n{text}"
                try:
                    result = request(args.base, model, content)
                except urllib.error.HTTPError as error:
                    if error.code != 400 or target < CAP // 2 or attempt == 2:
                        raise
                    print(f"{kind} {target}: refused ({error.code}), shrinking", flush=True)
                    corpus.ratios[span[0]] *= 0.95
                    continue
                tokens = result["prompt_tokens"]
                if not tokens:
                    raise SystemExit(f"{args.engine} reported no prompt_tokens")
                corpus.ratios[span[0]] = len(text) / tokens
                if kind == "first" or abs(tokens - target) <= 0.06 * target or tokens >= CAP - 1:
                    break
                print(f"{kind} {target}: {tokens} tokens, resizing", flush=True)
            fresh = tokens - result["cached_tokens"]
            record = {"schema": 2, "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "engine": args.engine, "model": model, "kind": kind, "target": target, "rep": rep,
                      "source": source, "attempts": attempt + 1, **result, "uncached_tokens": fresh,
                      "client_prefill_tps": fresh / result["wall_s"]}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            print(f"{kind} {target} rep {rep}: {tokens} tokens ({result['cached_tokens']} cached) in "
                  f"{result['wall_s']:.3f} s, {record['client_prefill_tps']:.0f} tok/s", flush=True)

        for item in calibration:
            send(*item)
        rest = [(n, r) for n in lengths[1:] for r in range(args.reps)]
        spans = {key: corpus.reserve(key[0]) for key in sorted(rest, key=lambda key: -key[0])}
        for n, r in rest:
            send("curve", n, r, spans[(n, r)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
