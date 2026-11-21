"""Run one memory-heavy campaign command under the GPU lock and a memory ceiling.

usage: heavy.py [--cap-gb N] -- COMMAND [ARG...]

The command runs as `lockf -k <gpu.lock> COMMAND` in its own process group. Once a
second, the summed physical footprint of that process tree is compared against
the cap; on a breach the whole group gets SIGTERM, then SIGKILL after 10 s, and
heavy.py exits 3. Otherwise it exits with the command's status. `-k` keeps the
lock file: without it lockf unlinks the file on exit, and a waiter that already
opened the old file and a newcomer that creates a new one both hold "the" lock.

The campaign as a whole stays under 32 GB so the host remains usable: one heavy
command at a time (the lock) gets 28 GB, leaving 4 GB for builds, clients and
agents. Flint servers set FLINT_LOAD_LIMIT_GB and Inkling 2 Mini servers set
INKLING_LOAD_LIMIT_GB to fit inside the heavy cap.
"""

import argparse
import ctypes
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent
MAIN_ROOT = Path(subprocess.run(["git", "-C", str(CAMPAIGN), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                capture_output=True, text=True, check=True).stdout.strip()).parent
GPU_LOCK = MAIN_ROOT / ".work" / "gpu.lock"

HEAVY_CAP_GB = 28
FLINT_LOAD_LIMIT_GB = 27
INKLING_LOAD_LIMIT_GB = 26

RUSAGE_INFO_V2 = 2
PHYS_FOOTPRINT_OFFSET = 72
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def footprint(pid: int) -> int:
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V2, buf) != 0:
        return 0
    return struct.unpack_from("<Q", buf.raw, PHYS_FOOTPRINT_OFFSET)[0]


def tree(root: int) -> list[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        pid, ppid = (int(field) for field in line.split())
        children.setdefault(ppid, []).append(pid)
    found, stack = [], [root]
    while stack:
        pid = stack.pop()
        found.append(pid)
        stack.extend(children.get(pid, []))
    return found


def group_members(group: int) -> list[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,pgid="], capture_output=True, text=True).stdout
    return [int(pid) for pid, pgid in (line.split() for line in out.splitlines()) if int(pgid) == group]


def stop(child: subprocess.Popen) -> None:
    for sig, grace in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(child.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            child.poll()
            if not group_members(child.pid):
                return
            time.sleep(0.2)
    print(f"heavy: process group {child.pid} still has members {group_members(child.pid)}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cap-gb", type=float, default=HEAVY_CAP_GB)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command")
    cap = int(args.cap_gb * 1e9)
    child = subprocess.Popen(["lockf", "-k", str(GPU_LOCK), *command], start_new_session=True)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (stop(child), sys.exit(130)))
    peak = 0
    while child.poll() is None:
        total = sum(footprint(pid) for pid in set(tree(child.pid)) | set(group_members(child.pid)))
        peak = max(peak, total)
        if total > cap:
            print(f"heavy: process tree footprint {total / 1e9:.1f} GB exceeded the {args.cap_gb:g} GB cap; "
                  f"stopping {' '.join(command[:2])}", file=sys.stderr)
            stop(child)
            return 3
        time.sleep(1)
    print(f"heavy: peak footprint {peak / 1e9:.1f} GB (cap {args.cap_gb:g} GB)", file=sys.stderr)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
