"""Run one memory-heavy campaign command under the GPU lock and a memory ceiling.

usage: heavy.py [--cap-gb N] -- COMMAND [ARG...]

The command runs as `lockf -k <gpu.lock> COMMAND` in its own process group.
Once a second, each process in the tree is charged the larger of its physical
footprint and its resident size, plus the growth of host-wide wired memory since
the command started (Metal wires mapped weight files outside any process's
footprint or resident size; other processes' wiring is charged too, which errs
toward stopping), and the total is compared against the cap. On a breach the
whole group gets SIGTERM, then SIGKILL after 10 s, and heavy.py exits 3;
critical host memory pressure stops it the same way with exit 4. Otherwise it
exits with the command's status. The final receipt line separates the charged
peak, the footprint, the resident size and the host's pressure, swap and
pagein changes. `-k` keeps the lock file: without it lockf unlinks the file on
exit, and a waiter that already opened the old file and a newcomer that
creates a new one both hold "the" lock. A heavy.py started inside another one's
tree (a make target that wraps its own heavy commands) sees CAMPAIGN_GPU_LOCK_HELD
and runs under the held lock instead of waiting on it forever.

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
RESIDENT_SIZE_OFFSET = 64
LOCK_HELD_ENV = "CAMPAIGN_GPU_LOCK_HELD"
PRESSURE_CRITICAL = 4
_libproc = ctypes.CDLL("/usr/lib/libproc.dylib")


def usage(pid: int) -> tuple[int, int]:
    """(physical footprint, resident size). The footprint charges anonymous and GPU memory but not clean
    file-backed pages, so mapped weights only show in the resident size."""
    buf = ctypes.create_string_buffer(512)
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V2, buf) != 0:
        return 0, 0
    resident, phys = struct.unpack_from("<QQ", buf.raw, RESIDENT_SIZE_OFFSET)
    return phys, resident


def pressure_level() -> int:
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return 0


def swap_used_mb() -> float | None:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    for part in out.split("  "):
        if part.strip().startswith("used"):
            return float(part.split("=")[1].strip().rstrip("M"))
    return None


def wired_bytes() -> int:
    """Host-wide wired memory. Metal wires the weight files a model maps into GPU address space, and those
    pages are charged to no process's footprint or resident size."""
    page, wired = 16384, 0
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif line.startswith("Pages wired down:"):
            wired = int(line.split(":")[1].strip().rstrip("."))
    return wired * page


def pageins() -> int | None:
    for line in subprocess.run(["vm_stat"], capture_output=True, text=True).stdout.splitlines():
        if line.startswith("Pageins:"):
            return int(line.split(":")[1].strip().rstrip("."))
    return None


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
    held = os.environ.get(LOCK_HELD_ENV) == str(GPU_LOCK)
    argv = command if held else ["lockf", "-k", str(GPU_LOCK), *command]
    child = subprocess.Popen(argv, start_new_session=True, env={**os.environ, LOCK_HELD_ENV: str(GPU_LOCK)})
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: (stop(child), sys.exit(130)))
    peak_footprint = peak_resident = peak_charged = peak_wired_growth = 0
    wired_start = wired_bytes()
    worst_pressure = pressure_level()
    swap_before, pageins_before = swap_used_mb(), pageins()

    def receipt(status: str) -> None:
        swap_after, pageins_after = swap_used_mb(), pageins()
        swap = f"{swap_after - swap_before:+.0f} MB" if swap_before is not None and swap_after is not None else "unknown"
        pages = f"{pageins_after - pageins_before}" if pageins_before is not None and pageins_after is not None else "unknown"
        print(f"heavy: {status}: peak charged {peak_charged / 1e9:.1f} GB of the {args.cap_gb:g} GB cap "
              f"(footprint {peak_footprint / 1e9:.1f} GB, resident {peak_resident / 1e9:.1f} GB, "
              f"host wired growth {peak_wired_growth / 1e9:.1f} GB); "
              f"host pressure worst {worst_pressure}, host swap {swap}, host pageins {pages}", file=sys.stderr)

    while child.poll() is None:
        members = set(tree(child.pid)) | set(group_members(child.pid))
        samples = [usage(pid) for pid in members]
        footprint_total = sum(phys for phys, _ in samples)
        resident_total = sum(resident for _, resident in samples)
        wired_growth = max(0, wired_bytes() - wired_start)
        peak_wired_growth = max(peak_wired_growth, wired_growth)
        charged = sum(max(phys, resident) for phys, resident in samples) + wired_growth
        peak_footprint = max(peak_footprint, footprint_total)
        peak_resident = max(peak_resident, resident_total)
        peak_charged = max(peak_charged, charged)
        level = pressure_level()
        worst_pressure = max(worst_pressure, level)
        if charged > cap:
            print(f"heavy: process tree charged {charged / 1e9:.1f} GB exceeded the {args.cap_gb:g} GB cap; "
                  f"stopping {' '.join(command[:2])}", file=sys.stderr)
            stop(child)
            receipt("stopped over cap")
            return 3
        if level >= PRESSURE_CRITICAL:
            print(f"heavy: host memory pressure is critical (level {level}); stopping {' '.join(command[:2])}",
                  file=sys.stderr)
            stop(child)
            receipt("stopped on host pressure")
            return 4
        time.sleep(1)
    receipt("done")
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
