import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SWIFT_PROBE = r"""
import Metal
let device = MTLCreateSystemDefaultDevice()
if let device {
    print("name=\(device.name)")
    print("recommendedMaxWorkingSetSize=\(device.recommendedMaxWorkingSetSize)")
    print("hasUnifiedMemory=\(device.hasUnifiedMemory)")
    print("maxBufferLength=\(device.maxBufferLength)")
    print("supportsFamily.apple7=\(device.supportsFamily(.apple7))")
    print("supportsFamily.apple8=\(device.supportsFamily(.apple8))")
    print("supportsFamily.apple9=\(device.supportsFamily(.apple9))")
    print("supportsFamily.apple10=\(device.supportsFamily(.apple10))")
    print("supportsFamily.metal3=\(device.supportsFamily(.metal3))")
    print("supportsFamily.metal4=\(device.supportsFamily(.metal4))")
} else {
    print("name=none")
}
"""


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def sysctl(key: str) -> str:
    return run(["sysctl", "-n", key])


def metal_probe() -> dict:
    found = shutil.which("swift")
    if not found:
        return {"swift": None}
    proc = subprocess.run([found, "-e", SWIFT_PROBE], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"swift": found, "error": proc.stderr.strip()[-500:]}
    out: dict = {"swift": found}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def metal_toolchain() -> dict:
    metal = run(["xcrun", "-f", "metal"])
    match = re.search(r"MetalToolchain-v([0-9.]+?)\.[a-z0-9]+/", metal)
    return {
        "metal_path": metal,
        "toolchain_version": match.group(1) if match else None,
        "metal_version": run(["xcrun", "metal", "--version"]).splitlines()[0],
        "sdk_path": run(["xcrun", "--sdk", "macosx", "--show-sdk-path"]),
    }


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def displays() -> dict:
    out = {}
    for line in run_text(["system_profiler", "SPDisplaysDataType"]).splitlines():
        stripped = line.strip()
        if "Metal Support" in stripped:
            out["metal_support"] = stripped.split(":", 1)[-1].strip()
        if "Total Number of Cores" in stripped:
            m = re.search(r"(\d+)", stripped)
            out["gpu_cores"] = int(m.group(1)) if m else None
    return out


def main() -> int:
    host = {
        "sysctl": {
            "brand": sysctl("machdep.cpu.brand_string"),
            "perflevel0_name": sysctl("hw.perflevel0.name"),
            "perflevel1_name": sysctl("hw.perflevel1.name"),
            "perflevel0_physicalcpu": int(sysctl("hw.perflevel0.physicalcpu")),
            "perflevel1_physicalcpu": int(sysctl("hw.perflevel1.physicalcpu")),
            "ncpu": int(sysctl("hw.ncpu")),
            "memsize_bytes": int(sysctl("hw.memsize")),
        },
        "gpu": displays(),
        "sw_vers": {
            "product_name": run(["sw_vers", "-productName"]),
            "product_version": run(["sw_vers", "-productVersion"]),
            "build_version": run(["sw_vers", "-buildVersion"]),
        },
        "xcodebuild": run(["xcodebuild", "-version"]).splitlines(),
        "sdk_version": run(["xcrun", "--sdk", "macosx", "--show-sdk-version"]),
        "metal_toolchain": metal_toolchain(),
        "zig": run(["zig", "version"]),
        "free_disk_bytes": shutil.disk_usage("/").free,
        "pmset_therm": run_text(["pmset", "-g", "therm"]).strip(),
        "metal": metal_probe(),
    }
    out_path = Path(__file__).parent / "host.json"
    out_path.write_text(json.dumps(host, indent=2) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
