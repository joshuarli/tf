import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODELS = {"flint": Path.home() / "models" / "flint-int4@3b93f9053b53"}
MODES = {"ar": ["--no-drafts"]}
TOKENS = 512
PROMPT = "Explain, step by step, how a hash map handles collisions, then compare open addressing with chaining."


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_ready(base: str, proc: subprocess.Popen, limit_s: float) -> None:
    deadline = time.monotonic() + limit_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode} before it was ready")
        try:
            with urllib.request.urlopen(f"{base}/v1/models", timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise SystemExit(f"server not ready after {limit_s:.0f} s")


def main() -> int:
    p = argparse.ArgumentParser(description="Serve one greedy fixed-length request and check its draft counters.")
    p.add_argument("--bin", required=True)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--mode", choices=sorted(MODES), required=True)
    p.add_argument("--load-limit-gb", type=int, required=True)
    p.add_argument("--log", required=True)
    args = p.parse_args()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    cmd = [args.bin, "serve", str(MODELS[args.model]), "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "1", "--prompt-cache-gib", "0", *MODES[args.mode]]
    env = dict(os.environ, FLINT_LOAD_LIMIT_GB=str(args.load_limit_gb))
    with open(args.log, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            wait_ready(base, proc, 600)
            body = {"model": args.model, "messages": [{"role": "user", "content": PROMPT}],
                    "max_tokens": TOKENS, "temperature": 0, "ignore_eos": True, "stream": False}
            reply = post(f"{base}/v1/chat/completions", body, 900)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    tokens = reply.get("usage", {}).get("completion_tokens")
    spec = reply.get("speculative")
    print(json.dumps({"completion_tokens": tokens, "speculative": spec, "decode_tokens_per_round": ((tokens or 0) - 1) / spec["rounds"] if isinstance(spec, dict) and spec.get("rounds") else None}, sort_keys=True))
    problems = []
    if tokens != TOKENS:
        problems.append(f"completion_tokens {tokens}, want {TOKENS}")
    if not isinstance(spec, dict):
        problems.append("reply has no speculative counters")
    else:
        if spec.get("drafted") != 0:
            problems.append(f"drafted {spec.get('drafted')}, want 0")
        decoded = (tokens or 0) - 1
        if spec.get("rounds") != decoded:
            problems.append(f"{spec.get('rounds')} decode rounds for {decoded} tokens after the prefill token, want 1.00 per round")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
