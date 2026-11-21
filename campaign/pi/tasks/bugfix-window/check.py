"""Hidden acceptance: full windows only, and the ValueError still fires."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import window  # noqa: E402


def main() -> int:
    if window.rolling_max([1, 3, 2, 5], 2) != [3, 3, 5]:
        return 1
    if window.rolling_max([4, 1, 7], 1) != [4, 1, 7]:
        return 1
    if window.rolling_mean([2, 4, 6, 8], 2) != [3.0, 5.0, 7.0]:
        return 1
    if window.rolling_mean([5], 2) != []:
        return 1
    for bad in (0, -1):
        try:
            window.rolling_max([1], bad)
        except ValueError:
            pass
        else:
            return 1
    return subprocess.run([sys.executable, str(repo / "test_window.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
