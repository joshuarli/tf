"""Compare two quality runs case by case: pass/fail, and whether each reply's content and reasoning are equal.

usage: quality_same.py BEFORE_RUN_DIR AFTER_RUN_DIR
"""

import json
import sys
from pathlib import Path


def cases(folder: str) -> dict:
    return {r["id"]: r for r in map(json.loads, (Path(folder) / "cases.jsonl").read_text().splitlines())}


def main() -> None:
    before, after = cases(sys.argv[1]), cases(sys.argv[2])
    same = 0
    for cid in sorted(before):
        b, a = before[cid], after.get(cid)
        if a is None:
            print(f"{cid}: missing after")
            continue
        equal = all(b.get(k) == a.get(k) for k in ("content", "reasoning", "tool_calls", "finish_reason"))
        same += equal
        print(f"{cid}: before {'pass' if b['passed'] else 'fail'}, after {'pass' if a['passed'] else 'fail'}, "
              f"reply {'equal' if equal else 'DIFFERS'}")
    passed = (sum(r["passed"] for r in before.values()), sum(r["passed"] for r in after.values()))
    print(f"{same} of {len(before)} replies equal; passed before {passed[0]}, after {passed[1]}")


if __name__ == "__main__":
    main()
