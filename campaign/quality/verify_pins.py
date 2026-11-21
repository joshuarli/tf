"""Verify an xgrammar checkout's tag, HEAD and every submodule commit against the recorded pins."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINS = HERE / "xgrammar.pins.json"


def capture(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--pins", type=Path, default=PINS)
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    if not (args.dir / ".git").exists():
        print(f"{args.dir} is not a git checkout", file=sys.stderr)
        return 1
    problems = []
    tag = capture(["git", "-C", str(args.dir), "describe", "--tags", "--exact-match"]).strip()
    if tag != pins["tag"]:
        problems.append(f"tag {tag!r} != {pins['tag']!r}")
    head = capture(["git", "-C", str(args.dir), "rev-parse", "HEAD"]).strip()
    if head != pins["commit"]:
        problems.append(f"HEAD {head} != {pins['commit']}")
    status = capture(["git", "-C", str(args.dir), "submodule", "status", "--recursive"])
    found = {}
    for line in status.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            found[fields[1]] = fields[0].lstrip("+-")
    for path, commit in sorted(pins["submodules"].items()):
        if found.get(path) != commit:
            problems.append(f"submodule {path} at {found.get(path)} != {commit}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"xgrammar {pins['tag']} at {head} and {len(pins['submodules'])} submodules match the pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
