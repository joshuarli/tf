"""curve.py --binary BIN --model-dir DIR --out FILE.jsonl [prefill_curve options]: serve_curve.py's native trial plus
the server's load time and footprint (harness.Server's timing and /memory samples) in its conditions record."""
import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CAMPAIGN / "bench"))
import harness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    args, rest = parser.parse_known_args()
    log = args.log or harness.TMP / f"curve-p6-{time.strftime('%H%M%S')}.log"
    key = harness.model_key_for_dir(args.model_dir)
    server = harness.Server(args.binary, args.model_dir, harness.default_serve_args(key), 0, log)
    launch = time.perf_counter()
    server.start(900)
    try:
        ready = time.perf_counter() - launch
        own = harness.process_tree(server.process.pid)
        before = harness.conditions(own)
        code = subprocess.run([sys.executable, str(CAMPAIGN / "comparators/prefill_curve.py"), "--base", server.base,
                               "--engine", "tensorfold", "--out", str(args.out), *rest]).returncode
        time.sleep(1)
        memory = server.stats()
        timing = server.timing()
        after = harness.conditions(own)
    finally:
        server.stop()
    record = {"schema": 1, "kind": "conditions", "engine": "tensorfold", "binary": str(args.binary),
              "model_dir": str(args.model_dir), "argv": server.argv, "spawn_to_ready_s": round(ready, 3),
              "timing": timing, "memory": memory, "host": harness.host_summary(),
              "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "before": before, "after": after,
              "server_log": str(log), "valid": not (before["pressure_reasons"] + after["pressure_reasons"])}
    with args.out.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"ready in {ready:.1f} s; footprint ready {memory['ready_bytes']} peak {memory['peak_bytes']}; "
          f"conditions valid={record['valid']}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
