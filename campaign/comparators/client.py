"""Campaign-compatible HTTP client for comparator engines.

`harness.chat_request` times only `content`/`reasoning_content` deltas. rec streams
`delta.reasoning`, so this wrapper renames that field per choice delta and otherwise
reuses bench_stream.Collector, bench_evidence, bench_metrics and bench_timing, giving
a run dict with the same shape harness.request_record exports.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
if str(CAMPAIGN / "bench") not in sys.path:
    sys.path.insert(0, str(CAMPAIGN / "bench"))

import harness  # noqa: E402  (adds tools/ to sys.path)
import bench_metrics  # noqa: E402
import bench_stream  # noqa: E402
import bench_timing  # noqa: E402

ALIASES = ("reasoning", "analysis", "thinking")


def normalize(line: bytes) -> bytes:
    stripped = line.decode().strip()
    if not stripped.startswith("data:"):
        return line
    payload = stripped[5:]
    if not payload.strip() or payload.strip() == "[DONE]":
        return line
    try:
        chunk = json.loads(payload)
    except ValueError:
        return line
    changed = False
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta")
        if not isinstance(delta, dict) or "reasoning_content" in delta:
            continue
        for alias in ALIASES:
            value = delta.get(alias)
            if isinstance(value, str) and value:
                delta["reasoning_content"] = value
                changed = True
                break
        if not delta.get("content") and not delta.get("reasoning_content") and choice.get("text"):
            del choice["text"]
            delta["content"] = choice["text"]
            changed = True
    return (("data: " + json.dumps(chunk)).encode() if changed else line)


def chat_request(base: str, model: str, content: str, *, max_tokens: int = 512, temperature: float = 0,
                 ignore_eos: bool = True, seed: int | None = 1234, timeout: float = 1800) -> dict:
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos,
            "messages": [{"role": "user", "content": content}]}
    if seed is not None and temperature == 0:
        body["seed"] = seed
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    collector = bench_stream.Collector(sent)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        collector.read(normalize(raw) for raw in response)
    completed = time.perf_counter()
    if collector.end == "error":
        raise RuntimeError(f"server reported an error: {collector.error}")
    complete = collector.end == "done"
    observed = collector.evidence.finish(collector.usage, collector.runtime, complete)
    pieces = collector.pieces
    first, last = (pieces[0][0], pieces[-1][0]) if pieces else (None, None)
    out = {"sent": sent, "completed": completed, "first": first, "last": last, "pieces": pieces,
           "ttft_s": (first - sent) if complete and first is not None else None,
           "decode_s": (last - first) if complete and first is not None and last > first else None,
           "total_s": (completed - sent) if complete else None, "complete": complete,
           "completion": collector.completion(), "finish_reasons": collector.receipt()["finish_reasons"],
           "runtime": collector.runtime, "speculative": collector.speculative or {},
           **observed}
    tokens = observed["tokens"]
    decode_s = out["decode_s"]
    out["decode_tps"] = (tokens - 1) / decode_s if tokens and tokens > 1 and decode_s else None
    out.update(bench_metrics.request(out))
    out.update(bench_timing.request(out))
    return out
