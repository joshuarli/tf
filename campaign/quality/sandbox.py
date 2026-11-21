"""Run generated Python once in a per-case directory under a macOS sandbox, a wall limit and rlimits."""

import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from eval_sandbox import mac_profile  # noqa: E402

WALL_S = 30
CPU_S = 20
FILE_BYTES = 16 << 20
OUTPUT_CHARS = 4000


def run(code: str, work: Path, wall_s: float = WALL_S, cpu_s: int = CPU_S) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work / "main.py").write_text(code, encoding="utf-8")
    started = time.monotonic()
    proc = subprocess.Popen(
        ["sandbox-exec", "-p", mac_profile(str(work)), sys.executable, "-I", "main.py"], cwd=work,
        env={"PATH": os.environ.get("PATH", os.defpath)}, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, preexec_fn=lambda: _set_limits(cpu_s))
    ended = "exit"
    try:
        out, err = proc.communicate(timeout=wall_s)
    except subprocess.TimeoutExpired:
        ended = "timeout"
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
    code_exit = proc.returncode
    return {"stdout": out.decode("utf-8", "replace")[:OUTPUT_CHARS],
            "stderr": err.decode("utf-8", "replace")[:OUTPUT_CHARS], "exit": code_exit, "ended": ended,
            "seconds": round(time.monotonic() - started, 3),
            "passed": ended == "exit" and code_exit == 0}


def _set_limits(cpu_s: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
