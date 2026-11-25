"""replycheck.py OUT.json BIN_A BIN_B: the same fixed prompts on fresh servers of two builds, replies compared.

Each build serves twice under heavy.py: --parallel 1 --context 32768 for four greedy replies of 48 tokens (prompt
cache 0, so every prompt prefills cold), then --parallel 8 --context 4096 (B2's sessions) for its ready footprint
only. Prompts are fixed corpus slices with no per-run text, so both builds see identical token ids.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CAMPAIGN = Path("/Users/josh/repos/tf/.work/wt/p5c-chunks/campaign")
MODEL = Path.home() / "models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
TMP = Path("/Users/josh/repos/tf/.work/tmp/p5c/replycheck")


def corpus(name, start, chars):
    folder = "holdout" if name.startswith("heldout") else "dev"
    return (CAMPAIGN / "prompts" / folder / f"{name}-32768.txt").read_text(encoding="utf-8")[start:start + chars]


PROMPTS = [
    "Summarize the following.\n\n" + corpus("heldout-code", 0, 3600),
    "Summarize the following.\n\n" + corpus("prose", 0, 3600),
    "Summarize the following.\n\n" + corpus("structured", 0, 9000),
    "Name three prime numbers.",
]


def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def serve(binary, port, parallel, context, log):
    env = dict(os.environ, INKLING_LOAD_LIMIT_GB="26", TENSORFOLD_NO_UPDATE_CHECK="1")
    argv = [sys.executable, str(CAMPAIGN / "heavy.py"), "--", str(binary), "serve", str(MODEL), "--host", "127.0.0.1",
            "--port", str(port), "--parallel", str(parallel), "--prompt-cache-gib", "0", "--no-update-check",
            "--context", str(context), "--temperature", "0"]
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    t0 = time.monotonic()
    while get(f"http://127.0.0.1:{port}/health", 2) is None:
        if proc.poll() is not None:
            raise SystemExit(f"server exited {proc.returncode}")
        time.sleep(0.2)
    return proc, time.monotonic() - t0


def stop(proc):
    os.killpg(os.getpgid(proc.pid), 15)
    proc.wait(timeout=60)


def chat(port, text):
    body = {"model": "m", "max_tokens": 48, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        reply = json.loads(r.read())
    msg = reply["choices"][0]["message"]
    return {"content": msg.get("content"), "reasoning": msg.get("reasoning_content"),
            "prompt_tokens": reply["usage"]["prompt_tokens"], "completion_tokens": reply["usage"]["completion_tokens"]}


TMP.mkdir(parents=True, exist_ok=True)
out = {"builds": {}}
for label, binary in (("a", sys.argv[2]), ("b", sys.argv[3])):
    rec = {"binary": binary}
    with open(TMP / f"{label}-1.log", "wb") as log:
        proc, ready = serve(binary, 47181, 1, 32768, log)
        try:
            rec["ready_s_1x32768"] = round(ready, 2)
            rec["ready_footprint_1x32768"] = (get("http://127.0.0.1:47181/memory") or {}).get("process", {}).get("physical_footprint_bytes")
            rec["replies"] = [chat(47181, p) for p in PROMPTS]
        finally:
            stop(proc)
    with open(TMP / f"{label}-8.log", "wb") as log:
        proc, ready = serve(binary, 47182, 8, 4096, log)
        try:
            time.sleep(1)
            rec["ready_s_8x4096"] = round(ready, 2)
            rec["ready_footprint_8x4096"] = (get("http://127.0.0.1:47182/memory") or {}).get("process", {}).get("physical_footprint_bytes")
        finally:
            stop(proc)
    out["builds"][label] = rec
a, b = out["builds"]["a"]["replies"], out["builds"]["b"]["replies"]
out["equal"] = [x == y for x, y in zip(a, b)]
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "builds"}),
      {l: {k: v for k, v in r.items() if k != "replies"} for l, r in out["builds"].items()})
