"""Hidden acceptance: answers.md matches what the source actually does."""

import sys
from pathlib import Path

repo = Path(sys.argv[1])


def main() -> int:
    path = repo / "answers.md"
    if not path.is_file():
        return 1
    fields = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    expected = {
        "workers": "2",
        "retry_limit": "8",
        "webhook_after_attempt": "drop",
        "dead_letter": "backfill,ingest",
    }
    for key, value in expected.items():
        if fields.get(key) != value:
            print(f"{key}: {fields.get(key)!r} != {value!r}", file=sys.stderr)
            return 1
    if (repo / "retry.py").read_text(encoding="utf-8").find("webhook") < 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
