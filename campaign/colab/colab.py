"""Budgeted Colab runs through the official `colab` CLI: probe, smoke and named benches.

Every GPU run has a compute-unit budget and a wall-clock deadline, is refused when its estimate exceeds either the
budget or the balance, checks the live usage rate right after allocation, retrieves its artifacts, and stops the
runtime it created on success, failure, timeout or interruption. Receipts go to .work/colab/.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REMOTE = HERE / "remote"
REPO = HERE.parent.parent
MAIN_ROOT = Path(subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip() or REPO / ".git").parent
WORK = MAIN_ROOT / ".work" / "colab"
LOCK = HERE.parent / "bootstrap.lock.json"


@dataclasses.dataclass(frozen=True)
class Accelerator:
    """`rate_ceiling` is a conservative compute-units-per-hour bound used only to refuse runs before allocation; the
    CLI publishes no price list, so the live rate measured after allocation must not exceed it."""
    name: str
    sm: int | None
    rate_ceiling: float
    tensorfold_cuda: str


ACCELERATORS = {
    "CPU": Accelerator("CPU", None, 0.2, "no GPU"),
    "T4": Accelerator("T4", 75, 2.0, "unsupported: sm_75 is below the CUDA build's sm_80 floor"),
    "L4": Accelerator("L4", 89, 5.0, "source-build target only; no Inkling 2 Mini CUDA path exists"),
}
NEEDS_APPROVAL = {"A100", "H100", "G4"}


@dataclasses.dataclass(frozen=True)
class Experiment:
    name: str
    script: str
    purpose: str
    min_sm: int | None
    needs_source: bool
    seconds: int


EXPERIMENTS = {
    "smoke": Experiment("smoke", "smoke.py", "confirm the accelerator and run an f32 GEMM against an fp64 reference",
                        None, False, 120),
    "env-probe": Experiment("env-probe", "env_probe.py",
                            "driver, toolkit, SM, RAM and VRAM against CUDA 13 / R580; pinned Zig on the working tree",
                            None, True, 300),
    "precision-error": Experiment("precision-error", "precision_error.py",
                                  "operand-precision error of fp16, bf16 and fp8 GEMMs at Inkling expert shapes", 75,
                                  False, 240),
}
SETUP_SECONDS = 180


EXPECTED_DEVICE = {"T4": "T4", "L4": "L4"}


def colab_gpu(session: str) -> str:
    return session.split("-")[1].upper()


class Refused(Exception):
    """A run that must not allocate, or must stop now: budget, balance, auth, quota or an unsupported accelerator."""


Runner = Callable[[list[str], float | None], subprocess.CompletedProcess]


def default_runner(argv: list[str], timeout: float | None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


AUTH_MARKERS = ("To authorize colab-cli", "authorization code", "401", "403", "DefaultCredentialsError")


class Colab:
    """The official CLI with ADC auth and session state isolated under .work/colab/, so this harness only ever sees,
    and only ever stops, the runtimes it created."""

    def __init__(self, runner: Runner = default_runner, work: Path = WORK, binary: str = "colab"):
        self.runner, self.work, self.binary = runner, work, binary
        self.work.mkdir(parents=True, exist_ok=True)
        self.owned_path = self.work / "owned.json"

    def cli(self, *args: str, timeout: float | None = 120) -> subprocess.CompletedProcess:
        argv = [self.binary, "--auth=adc", "--config", str(self.work / "sessions.json"), *args]
        result = self.runner(argv, timeout)
        text = (result.stdout or "") + (result.stderr or "")
        if any(marker in text for marker in AUTH_MARKERS):
            raise Refused("colab is not authenticated for ADC; run `gcloud auth application-default login` with the "
                          "scopes in `colab skill`")
        return result

    def usage(self) -> dict:
        result = self.cli("usage")
        text = result.stdout + result.stderr
        balance = re.search(r"Current balance:\s*([\d.]+)", text)
        rate = re.search(r"Usage rate:\s*([\d.]+)", text)
        active = re.search(r"Active assignments:\s*(\d+)", text)
        if result.returncode != 0 or not balance or not rate:
            raise Refused(f"compute-unit balance unavailable (exit {result.returncode}): {text.strip()[:300]}")
        return {"balance": float(balance.group(1)), "rate": float(rate.group(1)),
                "active": int(active.group(1)) if active else None}

    def owned(self) -> list[str]:
        return json.loads(self.owned_path.read_text()) if self.owned_path.exists() else []

    def _set_owned(self, names: list[str]) -> None:
        self.owned_path.write_text(json.dumps(sorted(set(names))) + "\n")

    def new(self, name: str, gpu: str) -> None:
        self._set_owned(self.owned() + [name])
        args = ["new", "-s", name] + ([] if gpu == "CPU" else ["--gpu", gpu])
        result = self.cli(*args, timeout=600)
        if result.returncode != 0:
            text = (result.stdout + result.stderr).strip()
            reason = "no quota or entitlement" if "400" in text or "quota" in text.lower() else "allocation failed"
            raise Refused(f"{reason} for {gpu} (exit {result.returncode}): {text[:300]}")

    def stop(self, name: str) -> bool:
        result = self.cli("stop", "-s", name, timeout=180)
        text = (result.stdout + result.stderr).lower()
        gone = result.returncode == 0 or "not found" in text
        if gone:
            self._set_owned([n for n in self.owned() if n != name])
        return gone


SOURCE_PATHS = ("build.zig", "build.zig.zon", "zig")


def source_fingerprint(repo: Path = REPO, paths: tuple[str, ...] = SOURCE_PATHS) -> dict:
    """The commit plus a sha256 over the path and bytes of every tracked and untracked, non-ignored file under the
    build inputs, so a dirty tree has its own identity and the uploaded copy can be checked against it."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z", "--", *paths], cwd=repo,
                           capture_output=True, text=True).stdout.split("\0")
    files = sorted(f for f in files if f and (repo / f).is_file())
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.encode() + b"\0" + (repo / f).read_bytes() + b"\0")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout != ""
    return {"commit": head, "dirty": dirty, "fingerprint": digest.hexdigest(), "files": files}


def source_tarball(source: dict, out: Path, repo: Path = REPO) -> Path:
    with tarfile.open(out, "w:gz") as tar:
        for f in source["files"]:
            tar.add(repo / f, arcname=f)
    return out


def estimate(gpu: str, experiments: list[Experiment]) -> tuple[float, float]:
    seconds = SETUP_SECONDS + sum(e.seconds for e in experiments)
    return seconds, ACCELERATORS[gpu].rate_ceiling * seconds / 3600


def admit(gpu: str, names: list[str], max_cu: float, balance: float) -> tuple[list[Experiment], float, float]:
    """Refuse before allocation: an unapproved or unknown accelerator, an experiment the accelerator cannot run, or
    an estimate above the budget or the balance."""
    if gpu in NEEDS_APPROVAL:
        raise Refused(f"{gpu} needs the owner's explicit approval; this harness admits only {', '.join(ACCELERATORS)}")
    if gpu not in ACCELERATORS:
        raise Refused(f"unknown accelerator {gpu!r}; choose one of {', '.join(ACCELERATORS)}")
    unknown = [n for n in names if n not in EXPERIMENTS]
    if not names or unknown:
        raise Refused(f"unknown or empty experiment selection {unknown or names}; known: {', '.join(EXPERIMENTS)}")
    chosen = [EXPERIMENTS[n] for n in names]
    sm = ACCELERATORS[gpu].sm
    for e in chosen:
        if e.min_sm is not None and (sm is None or sm < e.min_sm):
            raise Refused(f"{e.name} needs sm_{e.min_sm} or newer; {gpu} cannot run it")
    seconds, cu = estimate(gpu, chosen)
    if cu > max_cu:
        raise Refused(f"estimate {cu:.3f} CU ({seconds:.0f} s at <= {ACCELERATORS[gpu].rate_ceiling}/h) exceeds "
                      f"MAX_CU={max_cu}")
    if cu > balance:
        raise Refused(f"estimate {cu:.3f} CU exceeds the balance {balance:.2f} CU; buy nothing, stop here")
    return chosen, seconds, cu


def run(gpu: str, names: list[str], max_cu: float, colab: Colab | None = None,
        clock: Callable[[], float] = time.monotonic) -> dict:
    """Allocate one runtime, run the named experiments in it in order, retrieve each result, release the runtime."""
    colab = colab or Colab()
    started = clock()
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    name = f"tf-{gpu.lower()}-{uuid.uuid4().hex[:8]}"
    out = colab.work / f"{stamp}-{name}"
    out.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint()
    receipt = {"session": name, "gpu": gpu, "names": names, "max_cu": max_cu, "commit": source["commit"],
               "dirty": source["dirty"], "source_fingerprint": source["fingerprint"],
               "tensorfold_cuda": ACCELERATORS.get(gpu, Accelerator(gpu, None, 0, "")).tensorfold_cuda,
               "results": {}, "exclusions": [], "status": "refused"}

    def finish(status: str) -> dict:
        receipt["status"] = status
        receipt["wall_s"] = round(clock() - started, 1)
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"colab: {status}: receipt {out / 'receipt.json'}", file=sys.stderr)
        return receipt

    try:
        before = colab.usage()
        chosen, seconds, cu = admit(gpu, names, max_cu, before["balance"])
    except Refused as refusal:
        receipt["reason"] = str(refusal)
        return finish("refused")
    receipt.update(balance_before=before["balance"], estimate_cu=round(cu, 4), deadline_s=seconds)
    deadline = started + seconds
    previous = signal.signal(signal.SIGTERM, _interrupt)
    status = "error"
    try:
        colab.new(name, gpu)
        live = colab.usage()
        receipt["live_rate"] = round(live["rate"] - before["rate"], 3)
        if receipt["live_rate"] > ACCELERATORS[gpu].rate_ceiling:
            raise Refused(f"live rate {receipt['live_rate']}/h exceeds the {gpu} ceiling "
                          f"{ACCELERATORS[gpu].rate_ceiling}/h")
        if any(e.needs_source for e in chosen):
            tar = source_tarball(source, out / "source.tar.gz")
            _check(colab.cli("upload", "-s", name, str(tar), "/content/tf-src.tar.gz", timeout=600), "upload source")
        for e in chosen:
            remaining = deadline - clock()
            if remaining <= 0:
                receipt["exclusions"].append({"name": e.name, "reason": "deadline reached before it started"})
                continue
            receipt["results"][e.name] = _experiment(colab, name, e, source, out, remaining)
        failed = [n for n, r in receipt["results"].items() if r.get("status") != "ok"]
        status = "ok" if not failed and not receipt["exclusions"] else "error"
    except Refused as refusal:
        receipt["reason"] = str(refusal)
        status = "refused"
    except subprocess.TimeoutExpired:
        receipt["reason"] = "wall-clock deadline reached"
    except KeyboardInterrupt:
        receipt["reason"] = "interrupted"
        status = "interrupted"
    except Exception as error:
        receipt["reason"] = f"{type(error).__name__}: {error}"
    finally:
        signal.signal(signal.SIGTERM, previous)
        receipt["released"] = colab.stop(name)
        if not receipt["released"]:
            receipt["reason"] = (receipt.get("reason", "") + f"; STOP FAILED: run `make -C campaign colab-cleanup`"
                                 f" (owned session {name})").lstrip("; ")
            status = "error"
        try:
            after = colab.usage()
            receipt["balance_after"] = after["balance"]
            receipt["consumed_cu"] = round(before["balance"] - after["balance"], 2)
        except Refused as refusal:
            receipt["consumed_cu"] = None
            receipt["usage_after"] = str(refusal)
    return finish(status)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _check(result: subprocess.CompletedProcess, what: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{what} failed (exit {result.returncode}): {(result.stdout + result.stderr)[-300:]}")


def _experiment(colab: Colab, session: str, e: Experiment, source: dict, out: Path, remaining: float) -> dict:
    remote_out = f"/content/tf-out/{e.name}"
    zig = next(i for i in json.loads(LOCK.read_text())["inputs"] if i["role"] == "colab-toolchain")
    env = [f"TF_OUT={remote_out}", f"TF_SOURCE_FP={source['fingerprint']}", f"TF_ZIG_URL={zig['url']}",
           f"TF_ZIG_SHA256={zig['sha256']}", f"TF_ZIG_VERSION={zig['version']}"]
    composed = out / f"{e.name}.remote.py"
    composed.write_text((REMOTE / "common.py").read_text() + "\n\n" + (REMOTE / e.script).read_text())
    argv = ["exec", "-s", session, "-f", str(composed), "--timeout", str(int(remaining))]
    for item in env:
        argv += ["--env", item]
    result = colab.cli(*argv, timeout=remaining)
    (out / f"{e.name}.log").write_text(result.stdout + result.stderr)
    local = out / f"{e.name}.json"
    fetched = colab.cli("download", "-s", session, f"{remote_out}/result.json", str(local), timeout=300)
    if fetched.returncode != 0 or not local.exists():
        return {"status": "error", "reason": f"result not retrieved (exec exit {result.returncode})"}
    data = json.loads(local.read_text())
    want = EXPECTED_DEVICE.get(colab_gpu(session))
    got = data.get("runtime", {}).get("gpu") or ""
    if want and want not in got:
        return {"status": "error", "reason": f"runtime reports {got or 'no GPU'}, not the requested {want}"}
    if e.needs_source and data.get("source_fingerprint") != source["fingerprint"]:
        return {"status": "error", "reason": "remote source fingerprint does not match the working tree"}
    if result.returncode != 0 and data.get("status") == "ok":
        data["status"] = "error"
    return data


def probe(colab: Colab | None = None) -> int:
    """Readiness without allocation: CLI version, ADC auth, balance and rate, and this harness's owned sessions."""
    colab = colab or Colab()
    version = colab.cli("version")
    print(f"cli: {(version.stdout or version.stderr).strip()}")
    try:
        usage = colab.usage()
    except Refused as refusal:
        print(f"not ready: {refusal}")
        return 1
    print(f"balance: {usage['balance']:.2f} CU; account usage rate {usage['rate']:.2f}/h; "
          f"active assignments {usage['active']}")
    print(f"owned sessions: {', '.join(colab.owned()) or 'none'}")
    for gpu, a in ACCELERATORS.items():
        print(f"{gpu}: refuse above {a.rate_ceiling} CU/h; TensorFold CUDA: {a.tensorfold_cuda}")
    print(f"needs approval: {', '.join(sorted(NEEDS_APPROVAL))}")
    print("ready")
    return 0


def cleanup(colab: Colab | None = None) -> int:
    colab = colab or Colab()
    owned = colab.owned()
    if not owned:
        print("no owned sessions")
        return 0
    left = [name for name in owned if not colab.stop(name)]
    for name in owned:
        print(f"{name}: {'still running' if name in left else 'stopped'}")
    return 1 if left else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probe")
    sub.add_parser("cleanup")
    sub.add_parser("list")
    for command in ("smoke", "bench"):
        p = sub.add_parser(command)
        p.add_argument("--gpu", required=True)
        p.add_argument("--max-cu", type=float, required=True)
        if command == "bench":
            p.add_argument("--names", required=True)
    args = parser.parse_args(argv)
    if args.command == "probe":
        return probe()
    if args.command == "cleanup":
        return cleanup()
    if args.command == "list":
        for e in EXPERIMENTS.values():
            sm = f"sm_{e.min_sm}+" if e.min_sm else "any runtime"
            print(f"{e.name}: {sm}, ~{e.seconds} s{', uploads the working tree' if e.needs_source else ''}: {e.purpose}")
        return 0
    names = ["smoke"] if args.command == "smoke" else [n for n in args.names.replace(",", " ").split() if n]
    receipt = run(args.gpu.upper(), names, args.max_cu)
    print(json.dumps({k: receipt.get(k) for k in ("status", "reason", "estimate_cu", "live_rate", "consumed_cu",
                                                  "released")}))
    return 0 if receipt["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
