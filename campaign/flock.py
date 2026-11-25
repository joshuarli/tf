import fcntl
import math
import os
import subprocess
import sys
import time


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: flock.py WAIT_SECONDS LOCKFILE [COMMAND ...]", file=sys.stderr)
        return 2
    try:
        wait_seconds = float(sys.argv[1])
    except ValueError:
        print("flock.py: WAIT_SECONDS must be a non-negative number", file=sys.stderr)
        return 2
    if wait_seconds < 0 or math.isnan(wait_seconds):
        print("flock.py: WAIT_SECONDS must be a non-negative number", file=sys.stderr)
        return 2

    lock_fd = os.open(sys.argv[2], os.O_CREAT | os.O_RDWR, 0o666)
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 1
            time.sleep(min(0.05, remaining))

    if len(sys.argv) == 3:
        return 0

    try:
        result = subprocess.run(sys.argv[3:], pass_fds=(lock_fd,), check=False)
    except OSError as error:
        status = 127 if isinstance(error, FileNotFoundError) else 126
        print(f"flock.py: failed to execute {sys.argv[3]}: {error.strerror}", file=sys.stderr)
        return status
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
