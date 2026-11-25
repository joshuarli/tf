"""firstprobe.py --binary BIN --model-dir DIR --out FILE.jsonl: serve_curve.py's run at 1K, plus load time and footprint.

One fresh server (prompt cache 0), the same first request and six unseen 1K prompts as prefill_curve.py, then the
server's own load and footprint figures: harness.Server's timing and its /memory samples (ready, steady, peak).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CAMPAIGN = Path("/Users/josh/repos/tf/.work/wt/p5c-chunks/campaign")
sys.path.insert(0, str(CAMPAIGN / "bench"))
import harness  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True, type=Path)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
log = harness.TMP / f"firstprobe-{time.strftime('%H%M%S')}.log"
key = harness.model_key_for_dir(args.model_dir)
server = harness.Server(args.binary, args.model_dir, harness.default_serve_args(key), 0, log)
server.start(900)
try:
    own = harness.process_tree(server.process.pid)
    before = harness.conditions(own)
    code = subprocess.run([sys.executable, str(CAMPAIGN / "comparators/prefill_curve.py"), "--base", server.base,
                           "--engine", "tensorfold", "--out", str(args.out), "--lengths", "1024"]).returncode
    time.sleep(1)
    stats = server.stats()
    timing = server.timing()
    after = harness.conditions(own)
finally:
    server.stop()
record = {"schema": 1, "kind": "conditions", "binary": str(args.binary), "timing": timing, "memory": stats,
          "before": before, "after": after, "valid": not (before["pressure_reasons"] + after["pressure_reasons"]),
          "server_log": str(log)}
with args.out.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
print(f"load {timing}; memory {stats}; valid {record['valid']}", flush=True)
sys.exit(code)
