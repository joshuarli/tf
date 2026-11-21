"""Hidden acceptance: coverage is the union, and merge itself is unchanged in behavior."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import intervals  # noqa: E402
from intervals import Interval  # noqa: E402


def main() -> int:
    if intervals.total_covered([Interval(0, 4), Interval(2, 6), Interval(10, 12)]) != 8:
        return 1
    if intervals.total_covered([Interval(1, 5), Interval(2, 3), Interval(4, 8)]) != 7:
        return 1
    if intervals.total_covered([]) != 0:
        return 1
    if intervals.merge([Interval(5, 7), Interval(1, 2)]) != [Interval(1, 2), Interval(5, 7)]:
        return 1
    return subprocess.run([sys.executable, str(repo / "test_intervals.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
