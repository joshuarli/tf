"""Start one server, run prefill_curve.py against it, and stop it.

NAME is a comparator directory under campaign/comparators, or `tensorfold` for the campaign's own
native server (bench harness Server, prompt cache off so every prompt prefills cold).

usage: serve_curve.py NAME --model-dir DIR --out FILE.jsonl [--binary PATH] [prefill_curve options]
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run import load_server

import harness  # noqa: E402 - run.py puts campaign/bench on the path

HERE = Path(__file__).resolve().parent
NATIVE = harness.REPO / "zig-out" / "native" / "bin" / "tensorfold-native"


class Comparator:
    def __init__(self, name: str, model_dir: Path, log_path: Path):
        self.module = load_server(name)
        self.model_dir, self.log_path = model_dir, log_path
        self.port = harness.free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.argv = self.module.argv(model_dir, self.port, 0, 1)

    def start(self, timeout: float) -> int:
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(self.argv, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env=dict(os.environ, **self.module.env(self.model_dir, 0)))
        self.module.wait_ready(self.base, timeout)
        return self.process.pid

    def stop(self) -> None:
        self.module.stop(self.process)
        self.log.close()


class Native:
    def __init__(self, binary: Path, model_dir: Path, log_path: Path):
        key = harness.model_key_for_dir(model_dir)
        self.server = harness.Server(binary, model_dir, harness.default_serve_args(key), 0, log_path)

    def start(self, timeout: float) -> int:
        self.server.start(timeout)
        self.base, self.argv = self.server.base, self.server.argv
        return self.server.process.pid

    def stop(self) -> None:
        self.server.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=NATIVE)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--out", required=True, type=Path)
    args, rest = parser.parse_known_args()
    log_path = harness.TMP / f"curve-{args.name}-{time.strftime('%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    engine = Native(args.binary, args.model_dir, log_path) if args.name == "tensorfold" else \
        Comparator(args.name, args.model_dir, log_path)
    launch = time.perf_counter()
    try:
        pid = engine.start(args.ready_timeout)
        ready = time.perf_counter() - launch
        own = harness.process_tree(pid)
        before = harness.conditions(own)
        code = subprocess.run([sys.executable, str(HERE / "prefill_curve.py"), "--base", engine.base, "--engine",
                               args.name, "--out", str(args.out), *rest]).returncode
        after = harness.conditions(own)
    finally:
        engine.stop()
    record = {"schema": 1, "kind": "conditions", "engine": args.name, "model_dir": str(args.model_dir),
              "argv": engine.argv, "spawn_to_ready_s": round(ready, 3), "host": harness.host_summary(),
              "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "before": before, "after": after,
              "valid": not (before["pressure_reasons"] + after["pressure_reasons"])}
    with args.out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"{args.name} ready in {ready:.1f} s; conditions valid={record['valid']}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
