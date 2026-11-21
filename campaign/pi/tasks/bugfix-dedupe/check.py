"""Hidden acceptance: dedupe keeps first occurrences in order and is linear on a large input."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import dedupe as module  # noqa: E402


def main() -> int:
    if module.dedupe([3, 1, 3, 2, 1]) != [3, 1, 2]:
        return 1
    if module.dedupe([]) != []:
        return 1
    big = list(range(20000)) + list(range(20000))
    if module.dedupe(big) != list(range(20000)):
        return 1
    return subprocess.run([sys.executable, str(repo / "test_dedupe.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
