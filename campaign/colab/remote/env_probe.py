# Runs after common.py in one composed file; runtime() and write() come from it.
import hashlib
import os
import subprocess
import tarfile
import urllib.request
from pathlib import Path

result = {"experiment": "env-probe", "runtime": runtime(), "status": "error"}
try:
    driver = result["runtime"].get("driver")
    major = int(driver.split(".")[0]) if driver else None
    result["cuda13_r580"] = {"driver": driver, "meets_r580": major is not None and major >= 580,
                             "nvcc": result["runtime"].get("nvcc")}
    src = Path("/content/tf-src")
    marker = src / ".tf-source-fingerprint"
    want = os.environ["TF_SOURCE_FP"]
    if not marker.exists() or marker.read_text().strip() != want:
        subprocess.run(["rm", "-rf", str(src)], check=True)
        src.mkdir(parents=True)
        with tarfile.open("/content/tf-src.tar.gz") as tar:
            tar.extractall(src, filter="data")
        marker.write_text(want + "\n")
    result["source_fingerprint"] = marker.read_text().strip()
    zig_dir = Path(f"/content/zig-{os.environ['TF_ZIG_VERSION']}")
    zig = zig_dir / "zig"
    if not zig.exists():
        archive = Path("/content/zig.tar.xz")
        urllib.request.urlretrieve(os.environ["TF_ZIG_URL"], archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != os.environ["TF_ZIG_SHA256"]:
            raise RuntimeError(f"zig tarball sha256 {digest} does not match the pin")
        with tarfile.open(archive) as tar:
            top = tar.getnames()[0].split("/")[0]
            tar.extractall("/content", filter="data")
        Path("/content", top).rename(zig_dir)
    version = subprocess.run([str(zig), "version"], capture_output=True, text=True).stdout.strip()
    steps = subprocess.run([str(zig), "build", "-l"], cwd=src, capture_output=True, text=True, timeout=240,
                           env=dict(os.environ, ZIG_GLOBAL_CACHE_DIR="/content/zig-cache"))
    result["zig"] = {"version": version, "pinned": os.environ["TF_ZIG_VERSION"],
                     "build_steps_exit": steps.returncode, "build_steps": len(steps.stdout.splitlines()),
                     "stderr_tail": steps.stderr[-600:]}
    result["status"] = "ok" if version == os.environ["TF_ZIG_VERSION"] and steps.returncode == 0 else "error"
except Exception as error:
    result["reason"] = f"{type(error).__name__}: {error}"
write(result)
