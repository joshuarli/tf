"""Hidden acceptance: status_counts.txt has the derived counts, sorted by status."""

import sys
from pathlib import Path

repo = Path(sys.argv[1])


def main() -> int:
    path = repo / "status_counts.txt"
    if not path.is_file():
        return 1
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if lines != ["error: 4", "ok: 6"]:
        print(lines, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
