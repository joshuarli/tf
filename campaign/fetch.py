import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from buildlib.locks import FetchError, load_sources, fetch_input

REPO_NAMES = {
    "flint-int4": "flint-int4",
    "flint-dflash2": "Flint-DFlash2",
    "inkling-2-mini": "Inkling-2-Mini",
    "inkling-2-mini-tiny": "tiny-Inkling2LoopForConditionalGeneration",
    "inkling-2-mini-oq4e-mtp": "Inkling-2-Mini-oQ4e-mtp",
    "inkling-2-mini-nvfp4-xl-gguf": "Inkling-2-Mini-NVFP4-XL-GGUF",
    "inkling-2-mini-dflash2": "Inkling-2-Mini-DFlash2",
    "inkling-2-mini-ud-q4km-gguf": "Inkling-2-Mini-GGUF",
}


def source_root() -> Path:
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=Path(__file__).parent, capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent / ".work" / "src"


def target_dir(dest: Path, model: str, rev: str, store: str) -> Path:
    if store == "src":
        return source_root() / f"{model}@{rev[:12]}"
    return dest / f"{REPO_NAMES[model]}@{rev[:12]}"


def already_valid(path: Path, size: int | None, sha256: str) -> bool:
    if not path.is_file():
        return False
    if size is not None and path.stat().st_size != size:
        return False
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest() == sha256


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch pinned model inputs.")
    parser.add_argument("--only", nargs="*", action="append", metavar="MODELKEY",
                        help="fetch only these model keys; repeatable")
    parser.add_argument("--dest", default=str(Path.home() / "models"),
                        help="destination root (default: ~/models)")
    args = parser.parse_args()

    only = set()
    for group in args.only or []:
        only.update(group)
    dest = Path(args.dest).expanduser()

    lock = load_sources(Path(__file__).parent / "sources.lock.json")
    group: dict[str, str] = {}
    for inp in lock.inputs:
        group.setdefault(inp.extra["model"], inp.version)
    unknown = only - set(group)
    if unknown:
        print(f"unknown model key(s): {' '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    for inp in lock.inputs:
        model = inp.extra["model"]
        if only and model not in only:
            continue
        fname = inp.name.split("/", 1)[1]
        target = target_dir(dest, model, inp.version, inp.extra.get("store", "models")) / fname
        if already_valid(target, inp.size, inp.sha256):
            print(f"skip {inp.name}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            fetch_input(inp, target)
        except FetchError as error:
            print(f"error {inp.name}: {error}", file=sys.stderr)
            return 1
        print(f"fetched {inp.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
