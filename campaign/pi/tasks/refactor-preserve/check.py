"""Hidden acceptance: behavior is identical and the requested shape is present."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import report  # noqa: E402


def main() -> int:
    expect = {0: "F", 69.9: "F", 70: "C", 79.9: "C", 80: "B", 89.9: "B", 90: "A", 100: "A"}
    for mean, grade in expect.items():
        if report.grade_bucket(mean) != grade:
            return 1
    if report.summarize([]) != {"count": 0, "total": 0, "mean": 0}:
        return 1
    if report.summarize([200, 10]) != {"count": 2, "total": 110, "mean": 55.0}:
        return 1
    source = (repo / "report.py").read_text(encoding="utf-8")
    if "for " not in source.split("def grade_bucket", 1)[1]:
        return 1
    return subprocess.run([sys.executable, str(repo / "test_report.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
