"""Build the current TF raw-token comparator using the native runner's module graph."""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
SOURCE = BASE / "native.zig"
RUNNER = REPO / "zig/src/families/inkling2_loop/run.zig"
SOURCE_PATHS = ["zig", "build.zig", "build.zig.zon", "campaign/comparators/mlx-vlm"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity():
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS], cwd=REPO, text=True,
    ).splitlines()
    if untracked:
        raise RuntimeError("Untracked files in native source paths must be added to git before building: " + ", ".join(untracked))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", *SOURCE_PATHS], cwd=REPO)
    return {
        "source_revision": revision,
        "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "native_runner_sha256": sha256(SOURCE),
    }


def native_command(log, output):
    """Retain the build's complete module/framework wiring, replacing only its root."""
    for line in log.splitlines():
        try:
            command = shlex.split(line.removeprefix("info(verbose): ").strip())
        except ValueError:
            continue
        for i, arg in enumerate(command):
            if not arg.startswith("-Mroot="):
                continue
            root = Path(arg.split("=", 1)[1])
            if not root.is_absolute():
                root = REPO / root
            if root.resolve() != RUNNER:
                continue
            command[i] = "-Mroot=" + str(SOURCE)
            command = [arg for arg in command if arg != "--listen=-" and not arg.startswith("-femit-bin")]
            command.append("-femit-bin=" + str(output))
            return command
    raise RuntimeError("Zig's verbose build did not report the Inkling runner compiler command")


def write_receipt(path, receipt):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "build/bin/native-compare")
    args = parser.parse_args()
    # Resolve a caller-supplied relative output before subprocesses switch to REPO.
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    zig_version = subprocess.check_output(["zig", "version"], cwd=REPO, text=True).strip()
    command = [
        "zig", "build", "tf-inkling2-loop-run", "-Dcpu=apple_m1",
        "--prefix", str(BASE / "build"), "--cache-dir", str(BASE / ".zig-cache"), "--verbose",
    ]
    # Always build: an old output and receipt are never relabelled as the current source.
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
    native = native_command(result.stderr + "\n" + result.stdout, output)
    subprocess.run(native, cwd=REPO, check=True)
    if source_identity() != identity:
        raise RuntimeError("TF source changed during the build; rerun to produce a trustworthy receipt")
    receipt = {
        "schema_version": 1,
        "type": "tf_inkling_native_build",
        "binary_path": str(output),
        "binary_sha256": sha256(output),
        **identity,
        "source_diff_paths": SOURCE_PATHS,
        "zig_version": zig_version,
        "build_command": command,
        "native_build_command": native,
    }
    path = Path(str(output) + ".json")
    write_receipt(path, receipt)
    print(f"Built {output}; source and binary receipt: {path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, file=sys.stderr, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        raise SystemExit(exc.returncode) from exc
