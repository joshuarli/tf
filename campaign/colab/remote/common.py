"""Shared by the remote scripts: the runtime's identity and the result file the host downloads."""

import json
import os
import platform
import subprocess
from pathlib import Path


def sh(*argv: str) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def runtime() -> dict:
    info = {"python": platform.python_version(), "kernel": platform.release(),
            "ram_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1),
            "cpus": os.cpu_count()}
    smi = sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader")
    if smi:
        name, driver, memory, cap = [v.strip() for v in smi.splitlines()[0].split(",")]
        info.update(gpu=name, driver=driver, vram=memory, compute_capability=cap)
    nvcc = sh("nvcc", "--version")
    info["nvcc"] = nvcc.splitlines()[-1] if nvcc else None
    return info


def write(result: dict) -> None:
    out = Path(os.environ["TF_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result.get("status"), "out": str(out)}))
