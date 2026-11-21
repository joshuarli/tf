"""Records, for scorecard.py, why an engine has no measured cells.

Each entry is a `kind: unavailable` record in the campaign schema. A deferred
engine is one the owner steering did not measure this pass; an omitted setting
is one the engine cannot express, which the measured records also carry.
"""

import datetime as dt
import json
import sys
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))

import harness  # noqa: E402

RECORDS = [
    ("exo-main", "greedy", "deferred: not needed until a competitive claim; Exo needs a single-node cluster and an npm dashboard build, then its API serves the same checkpoint but with no ignore_eos or prompt-cache size"),
    ("exo-2533", "greedy", "deferred: not needed until a competitive claim; the PR #2533 head is pinned but not built this pass"),
    ("nanoattention", "greedy", "deferred: not needed until a competitive claim; it is public MIT Zig/Metal and built here (zig-out/bin/flint session), but this pass measured no full 512-token cell"),
]


def main() -> int:
    out_dir = harness.RUNS_DIR / dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "comparator-omissions.jsonl"
    path.touch()
    with path.open("a", encoding="utf-8") as stream:
        for engine, mode, reason in RECORDS:
            record = {"schema": 1, "kind": "unavailable", "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                      "run_id": "comparator-omissions", "engine": engine, "binary": None, "mode": mode,
                      "cell": None, "reason": reason, "host": harness.host_summary()}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
    print(f"wrote {path.relative_to(CAMPAIGN.parent)}: {len(RECORDS)} omission records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
