"""Prepare the pinned source and build only nano, without changing its inference code."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


SOURCE_COMMIT = "6f782a8dace846e4e64a5a12cbdf5c0252166199"
CAMPAIGN = Path(__file__).resolve().parents[2]
COMMON = Path(subprocess.check_output(
    ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
    cwd=CAMPAIGN, text=True).strip()).parent
WORK = COMMON / ".work"
BUILD = WORK / "build" / f"nanoattention-inkling@{SOURCE_COMMIT[:12]}"
BINARY = BUILD / "zig-out" / "bin" / "nano"
RECEIPT = BUILD / "setup.json"
BUILD_ANCHOR = b"    b.installArtifact(nano);\n"
BUILD_ADDITION = (
    b'    const campaign_nano = b.addInstallArtifact(nano, .{});\n'
    b'    b.step("campaign-nano", "Install only the nano comparator").dependOn(&campaign_nano.step);\n'
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def source_input() -> dict:
    entries = json.loads((CAMPAIGN / "sources.lock.json").read_text())["inputs"]
    matches = [entry for entry in entries if entry.get("model") == "nanoattention"]
    if len(matches) != 1 or matches[0]["version"] != SOURCE_COMMIT:
        raise ValueError("nanoattention source lock does not match the reviewed commit")
    return matches[0]


def prepare(archive: Path, build: Path, expected_sha: str, expected_size: int) -> dict:
    if not archive.is_file():
        raise ValueError("pinned archive missing; run make -C campaign fetch NAMES=nanoattention")
    if archive.stat().st_size != expected_size or sha256(archive) != expected_sha:
        raise ValueError("nanoattention source archive failed its pinned size/SHA-256 check")
    build.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".nanoattention-prepare-", dir=build.parent) as temp:
        staging = Path(temp)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(staging, filter="data")
        source = staging / f"nanoattention-{SOURCE_COMMIT}"
        upstream_build = (source / "build.zig").read_bytes()
        if upstream_build.count(BUILD_ANCHOR) != 1:
            raise ValueError("pinned nano build registration was not found exactly once")
        generated_build = upstream_build.replace(BUILD_ANCHOR, BUILD_ANCHOR + BUILD_ADDITION)
        (source / "build.zig").write_bytes(generated_build)
        files = sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
        if not files or not (source / "LICENSE").is_file():
            raise ValueError("nanoattention archive lacks its source files or license")
        digest = hashlib.sha256()
        for relative in files:
            original = source / relative
            if original.is_symlink():
                raise ValueError(f"unexpected source symlink: {relative}")
            file_sha = sha256(original)
            digest.update(f"{relative.as_posix()}\0{file_sha}\n".encode())
            if build.exists():
                existing = build / relative
                if existing.is_symlink() or not existing.is_file() or sha256(existing) != file_sha:
                    raise ValueError(f"existing comparator source differs: {relative}; preserve it and use a clean build")
        if not build.exists():
            source.rename(build)
    return {
        "schema": 1,
        "status": "prepared",
        "source_url": "https://github.com/justrach/nanoattention",
        "source_commit": SOURCE_COMMIT,
        "source_archive_sha256": expected_sha,
        "upstream_build_sha256": hashlib.sha256(upstream_build).hexdigest(),
        "build_sha256": hashlib.sha256(generated_build).hexdigest(),
        "build_addition_sha256": hashlib.sha256(BUILD_ADDITION).hexdigest(),
        "prepared_source_sha256": digest.hexdigest(),
        "source_files": len(files),
        "build_target": "campaign-nano",
        "optimize": "ReleaseSafe",
        "upstream_zig_version": "0.17.0",
        "zig_version": None,
        "binary": str(build / "zig-out" / "bin" / "nano"),
        "binary_sha256": None,
        "scope": "nano-only build; no Flint proof check or Metal/model qualification claimed",
    }


def write_receipt(receipt: Path, metadata: dict) -> None:
    temporary = receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="verify and extract without requiring a Mac")
    parser.add_argument("--zig", default="zig", help="Zig executable; its actual version is recorded")
    args = parser.parse_args()
    try:
        source = source_input()
        archive = WORK / "src" / f"nanoattention@{SOURCE_COMMIT[:12]}" / source["name"].split("/", 1)[1]
        metadata = prepare(archive, BUILD, source["sha256"], source["size"])
        if args.prepare_only and RECEIPT.is_file():
            previous = json.loads(RECEIPT.read_text())
            if (previous.get("status") == "built" and BINARY.is_file()
                    and previous.get("prepared_source_sha256") == metadata["prepared_source_sha256"]
                    and previous.get("binary_sha256") == sha256(BINARY)):
                metadata = previous
        write_receipt(RECEIPT, metadata)
        if not args.prepare_only:
            if platform.system() != "Darwin" or platform.machine() not in ("arm64", "aarch64"):
                raise ValueError("source prepared; building nano requires Apple Silicon macOS and the Metal SDK")
            metadata["zig_version"] = subprocess.check_output([args.zig, "version"], text=True).strip()
            metadata["host"] = {"system": platform.system(), "machine": platform.machine()}
            command = [args.zig, "build", "campaign-nano", "-Doptimize=ReleaseSafe",
                       "--cache-dir", str(BUILD / ".zig-cache"),
                       "--global-cache-dir", str(WORK / "zig-global-cache"),
                       "--prefix", str(BUILD / "zig-out")]
            metadata["command"] = command
            metadata["status"] = "building"
            write_receipt(RECEIPT, metadata)
            (WORK / "tmp").mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=BUILD, env={**os.environ, "TMPDIR": str(WORK / "tmp")})
            metadata["build_exit"] = result.returncode
            metadata["status"] = "built" if result.returncode == 0 and BINARY.is_file() else "build-failed"
            if metadata["status"] == "built":
                metadata["binary_sha256"] = sha256(BINARY)
            write_receipt(RECEIPT, metadata)
            if metadata["status"] != "built":
                raise ValueError(f"nano build failed (exit {result.returncode}); see {RECEIPT}")
        print(json.dumps({"receipt": str(RECEIPT), **metadata}, sort_keys=True))
        return 0
    except (OSError, ValueError, tarfile.TarError, subprocess.CalledProcessError) as error:
        print(f"nanoattention setup: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
