"""Hidden acceptance: distinct_ratio exists, is correct, and the task's own tests cover it."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import rates  # noqa: E402


def main() -> int:
    if not hasattr(rates, "distinct_ratio"):
        return 1
    if rates.distinct_ratio("") != 0.0:
        return 1
    if rates.distinct_ratio("a b a b") != 0.5:
        return 1
    if rates.distinct_ratio("a b c") != 1.0:
        return 1
    if "distinct_ratio" not in (repo / "test_rates.py").read_text(encoding="utf-8"):
        return 1
    return subprocess.run([sys.executable, str(repo / "test_rates.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
