"""Hidden acceptance: the new defaults flow through settings and the client, and tests agree."""

import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))

import settings  # noqa: E402
import client  # noqa: E402


def main() -> int:
    if settings.DEFAULT_TIMEOUT_S != 4 or settings.DEFAULT_RETRIES != 5:
        return 1
    if client.Client().describe() != "timeout=4 retries=5":
        return 1
    if client.Client(retries=1).describe() != "timeout=4 retries=1":
        return 1
    source = (repo / "test_client.py").read_text(encoding="utf-8")
    if "timeout=4" not in source or "retries=5" not in source:
        return 1
    return subprocess.run([sys.executable, str(repo / "test_client.py")], cwd=repo, capture_output=True).returncode


if __name__ == "__main__":
    raise SystemExit(main())
