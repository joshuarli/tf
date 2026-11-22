"""Paired, engine-only Inkling timings on identical raw token IDs; run the Make targets in README.md."""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))
import harness

SOURCE_COMMIT = "6f782a8dace846e4e64a5a12cbdf5c0252166199"
BUILD = harness.WORK / "build" / f"nanoattention-inkling@{SOURCE_COMMIT[:12]}"
NANO = BUILD / "zig-out/bin/nano"
NATIVE = harness.WORK / "build/nanoattention-native/bin/native-compare"
OPTIMIZED = dict.fromkeys((
    "expert_simd", "expert_wide_dot", "expert_gate_pair", "expert_prefill", "expert_wide_k",
    "expert_bf16_tiles", "expert_mlx_scale", "bf16_linear_tile", "bf16_prefill_matrix", "bf16_decode_vec",
    "fuse_bf16_cast", "concurrent_scratch", "relative_split", "relative_split_4", "relative_tile_prefill",
), "true") | {"bf16_prefill_tile": "8"}
VOCAB = 200058


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def workload(ids, steps):
    if not isinstance(ids, list) or len(ids) < 2:
        raise ValueError("nano bench needs at least two prompt tokens (bos plus nonempty code)")
    if any(type(token) is not int or not 0 <= token < VOCAB for token in ids):
        raise ValueError("prompt IDs must be integers in the Inkling vocabulary")
    if type(steps) is not int or steps < 2:
        raise ValueError("steps must include the prefill prediction and at least one decode prediction")
    if problem := harness.context_problem(len(ids), steps):
        raise ValueError(problem)
    # Upstream takes bos, then code[:prompt_tokens-1]. No repeated or retokenized text here.
    return {"bos": ids[0], "code": ids[1:], "decode": [0] * (steps - 1)}


def nano_command(model, tokens_file, prompt_tokens, steps, variant):
    knobs = {"batch": "128"} | (OPTIMIZED if variant == "optimized" else {})
    command = [str(NANO), "bench", str(model), "--context", str(harness.CONTEXT_TOKENS),
               "--prompt-tokens", str(prompt_tokens), "--decode-tokens", str(steps - 1),
               "--decode-mode", "greedy", "--bench-tokens", str(tokens_file), "--repeats", "1"]
    for key, value in knobs.items():
        command.extend(["--knob", f"{key}={value}"])
    return command


def parse_result(stdout, stderr, engine, count, steps):
    kind = "tf_inkling_bench" if engine == "tf" else "nano_bench"
    matches = []
    for line in (stdout + "\n" + stderr).splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("type") == kind:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {kind} receipt, found {len(matches)}")
    row = matches[0]
    expected = {"model_type": "inkling2_loop", "prompt_tokens": count, "decode_tokens": steps - 1,
                "warmup_runs": 1}
    if engine == "tf":
        expected |= {"generated_tokens": steps, "context": harness.CONTEXT_TOKENS,
                     "cache_capacity_tokens": harness.CONTEXT_TOKENS}
        if row.get("warmup_trial_match") is not True:
            raise ValueError("TF warmup and timed output were not verified equal")
    else:
        expected |= {"version": 1, "repeats": 1, "decode_mode": "free-running greedy (fixed length)"}
    for key, value in expected.items():
        if row.get(key) != value or isinstance(row.get(key), bool):
            raise ValueError(f"{kind} {key}: expected {value!r}, got {row.get(key)!r}")
    predictions = row.get("predictions")
    if (not isinstance(predictions, list) or len(predictions) != steps
            or any(type(token) is not int or not 0 <= token < VOCAB for token in predictions)):
        raise ValueError("receipt has wrong prediction count or invalid token IDs")
    times = {}
    for key in ("prefill_seconds", "decode_seconds"):
        value = row.get(key)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid {key}: {value!r}")
        times[key] = value
    return {"raw": row, **times, "prefill_tokens_per_second": count / times["prefill_seconds"],
            "decode_tokens_per_second": (steps - 1) / times["decode_seconds"],
            "predictions": predictions, "predictions_sha256": digest_ids(predictions)}


def output_agreement(left, right):
    if len(left) != len(right):
        raise ValueError("comparison requires equal output lengths")
    first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
    return {"identical": first is None, "first_different_generated_index": first,
            "matching_prefix_tokens": len(left) if first is None else first,
            "tf_token": None if first is None else left[first],
            "nano_token": None if first is None else right[first],
            "qualification": "diagnostic greedy agreement only; teacher-forced logit parity not established"}


def worker(job_path):
    """Inside heavy.py's lock, bracket just this owned process and preserve failures as evidence."""
    job = json.loads(Path(job_path).read_text())
    own = {os.getpid()}
    before = harness.conditions(own)
    reasons = list(before["pressure_reasons"])
    if before["pressure"]["gpu_gate"].get("status") != "quiet":
        reasons.append("pre-run GPU gate did not report quiet")
    row = {"command": job["command"], "before": before, "invalid_reasons": reasons, "returncode": 75}
    if not reasons:
        gpu_before = harness.gpu_time_by_pid(strict=True)
        start = time.monotonic()
        child = subprocess.Popen(job["command"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        own.add(child.pid)
        try:
            stdout, stderr = child.communicate(timeout=job["timeout"])
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                stdout, stderr = child.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                stdout, stderr = child.communicate()
            reasons.append("engine command exceeded its timeout")
        elapsed = time.monotonic() - start
        gpu_after = harness.gpu_time_by_pid(strict=True)
        after = harness.conditions(own)
        users, contention = harness.gpu_contention(gpu_before, gpu_after, elapsed, own)
        reasons.extend(after["pressure_reasons"] + contention)
        if (set(gpu_before) - own) != (set(gpu_after) - own):
            reasons.append("GPU client set changed; interval could not be measured")
        if any(gpu_after[pid][1] < gpu_before[pid][1] for pid in set(gpu_before) & set(gpu_after) - own):
            reasons.append("GPU time counter reset; interval could not be measured")
        if after["pressure"]["gpu_gate"].get("status") != "quiet":
            reasons.append("post-run GPU gate did not report quiet")
        if child.returncode != 0:
            reasons.append(f"engine exited {child.returncode}")
        row |= {"stdout": stdout, "stderr": stderr, "returncode": child.returncode,
                "process_seconds": elapsed, "after": after, "other_gpu_users": users,
                "contention_window": "whole engine process, including load and warmup"}
    print(json.dumps(row), flush=True)
    return 0 if not reasons else 75


def attempt(command, prefix, timeout):
    job = prefix.with_suffix(".job.json")
    job.write_text(json.dumps({"command": command, "timeout": timeout}))
    env = dict(os.environ, TMPDIR=str(harness.TMP), INKLING_LOAD_LIMIT_GB="26",
               TENSORFOLD_NO_UPDATE_CHECK="1")
    outer = [sys.executable, str(harness.HEAVY), "--", sys.executable, str(Path(__file__)), "--worker", str(job)]
    child = subprocess.Popen(outer, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    timed_out = interrupted = False
    try:
        stdout, stderr = child.communicate(timeout=timeout + 120)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        timed_out = not interrupted
        # Snapshot only groups in this wrapper's descendant tree before asking it to stop.
        groups = set()
        for pid in harness.process_tree(child.pid):
            try:
                group = os.getpgid(pid)
                if group != os.getpgrp():
                    groups.add(group)
            except ProcessLookupError:
                pass
        # heavy.py handles SIGTERM by stopping its separate, owned process group.
        child.terminate()
        try:
            stdout, stderr = child.communicate(timeout=25)
        except subprocess.TimeoutExpired:
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.kill()
            stdout, stderr = child.communicate(timeout=5)
    prefix.with_suffix(".stdout.log").write_text(stdout)
    prefix.with_suffix(".stderr.log").write_text(stderr)
    try:
        row = json.loads(stdout)
    except ValueError:
        row = {"invalid_reasons": ["worker did not emit a complete receipt"]}
    row |= {"heavy_exit": child.returncode, "heavy_receipt": stderr,
            "interrupted": interrupted,
            "stdout_log": str(prefix.with_suffix(".stdout.log")),
            "stderr_log": str(prefix.with_suffix(".stderr.log"))}
    if timed_out or interrupted:
        row["invalid_reasons"].append("interrupted by user" if interrupted else "outer watchdog timed out")
    if child.returncode != 0:
        row["invalid_reasons"].append(f"memory/lock wrapper exited {child.returncode}")
    return row


def verify_inputs():
    """Check the checkpoint once, before timing, including the actual packed weight bytes."""
    model = harness.model_dir("inkling-2-mini")
    for entry in harness.model_entries("inkling-2-mini"):
        if entry["name"].endswith("mtp.safetensors"):
            continue  # Both engines are autoregressive; the unused MTP head is not loaded.
        path = model / entry["name"].split("/", 1)[1]
        if (not path.is_file() or path.stat().st_size != entry["size"]
                or harness.sha256_file(path) != entry["sha256"]):
            raise ValueError(f"checkpoint pin mismatch: {path}; run make -C campaign fetch NAMES=inkling-2-mini")
    return model


def source_identity(binary):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=harness.REPO, text=True).strip()
    paths = ["zig", "build.zig", "build.zig.zon", "campaign/comparators/mlx-vlm"]
    if git("ls-files", "--others", "--exclude-standard", "--", *paths):
        raise ValueError("untracked TF source makes build identity ambiguous; git-add source and rebuild")
    diff = subprocess.check_output(["git", "diff", "HEAD", "--", "zig", "build.zig", "build.zig.zon",
                                    "campaign/comparators/mlx-vlm"], cwd=harness.REPO)
    current = {"source_revision": git("rev-parse", "HEAD"), "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "dirty": bool(git("status", "--porcelain")), "binary_sha256": harness.sha256_file(binary),
            "driver_sha256": harness.sha256_file(Path(__file__)),
            "native_runner_sha256": harness.sha256_file(HERE.parent / "mlx-vlm/native.zig"),
            "zig_version": subprocess.check_output(["zig", "version"], text=True).strip()}
    built = json.loads(Path(str(binary) + ".json").read_text())
    for key in ("source_revision", "source_diff_sha256", "binary_sha256", "native_runner_sha256", "zig_version"):
        if built.get(key) != current[key]:
            raise ValueError(f"native build receipt differs at {key}; rerun nanoattention-setup")
    return {**current, "build_receipt": built}


def host_identity():
    recorded = harness.host_summary("inkling-2-mini")
    def sysctl(key):
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    if (sysctl("machdep.cpu.brand_string") != recorded["product"]
            or int(sysctl("hw.memsize")) != recorded["memory_bytes"]):
        raise ValueError("campaign host.json is for a different Mac; run make -C campaign host")
    return {**recorded, "live_platform": platform.platform(),
            "live_sw_vers": subprocess.check_output(["sw_vers"], text=True).strip()}


def run(args):
    prompts = []
    for name in args.prompts.split(","):
        path = CAMPAIGN / "prompts/inkling" / f"{name}.json"
        ids = json.loads(path.read_text())
        tokens = workload(ids, args.steps)
        prompts.append((name, path, ids, tokens))
    if args.plan:
        print(json.dumps({"status": "plan_only", "context_tokens": harness.CONTEXT_TOKENS,
                          "generated_tokens": args.steps, "decode_steps": args.steps - 1,
                          "pairs": args.pairs, "variant": args.variant,
                          "prompts": [{"name": n, "tokens": len(ids), "sha256": digest_ids(ids)}
                                      for n, _, ids, _ in prompts]}, indent=2))
        return 0
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ValueError("Inkling Metal comparison requires your Apple Silicon Mac; use --plan on other hosts")
    if not shutil.which("gpu-gate-ready"):
        raise ValueError("gpu-gate-ready is required; timings cannot qualify without host-condition gates")
    for binary in (NANO, NATIVE):
        if not binary.is_file():
            raise ValueError(f"missing {binary}; run make -C campaign nanoattention-setup")
    setup = json.loads((BUILD / "setup.json").read_text())
    if (setup.get("status") != "built" or setup.get("source_commit") != SOURCE_COMMIT
            or setup.get("binary_sha256") != harness.sha256_file(NANO)):
        raise ValueError("nano build does not match its pinned setup receipt; run nanoattention-setup")
    print("Verifying the original checkpoint hashes before timing...", flush=True)
    model = verify_inputs()
    identity = {"tf": source_identity(NATIVE), "nanoattention": setup,
                "checkpoint": harness.model_hashes("inkling-2-mini"),
                "host": host_identity(),
                "gpu_gate_max_gpu_pct": harness.gpu_time_limit_pct(),
                "gpu_gate_max_util_pct": os.environ.get("GPU_GATE_MAX_UTIL_PCT", "default")}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / "results" / f"inkling-{stamp}.jsonl"
    logs = harness.TMP / f"nano-inkling-{stamp}"
    logs.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    def write(row):
        with output.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write({"kind": "engine_comparison_manifest", "protocol": "inkling-raw-greedy-v1", "identity": identity,
           "context_tokens": harness.CONTEXT_TOKENS, "generated_tokens": args.steps, "pairs": args.pairs,
           "variant": args.variant, "timing_scope": "engine prefill plus first argmax, then steps-1 decode; "
           "excludes model load and one full warmup; not served TTFT",
           "arithmetic": {"tf": "campaign native float32 activation path",
                          "nanoattention": "upstream BF16 path; optimized adds BF16 expert and FP16 attention tiles"},
           "qualification": "performance investigation; no matched-arithmetic or logit-parity claim"})
    failed = False
    for name, _, ids, tokens in prompts:
        token_path = logs / f"{name}.json"
        token_path.write_text(json.dumps(tokens))
        raw_path = logs / f"{name}-raw.json"
        raw_path.write_text(json.dumps(ids))
        for pair in range(1, args.pairs + 1):
            rows = {}
            for engine in (("tf", "nanoattention") if pair % 2 else ("nanoattention", "tf")):
                command = ([str(NATIVE), str(model), str(raw_path), str(args.steps), "1"] if engine == "tf" else
                           nano_command(model, token_path, len(ids), args.steps, args.variant))
                print(f"{name} pair={pair} engine={engine}", flush=True)
                row = attempt(command, logs / f"{name}-{pair}-{engine}", args.timeout)
                row |= {"kind": "engine_comparison_trial", "engine": engine, "prompt": name, "pair": pair,
                        "prompt_tokens": len(ids), "prompt_ids_sha256": digest_ids(ids),
                        "context_tokens": harness.CONTEXT_TOKENS}
                if not row["invalid_reasons"]:
                    try:
                        row["result"] = parse_result(row["stdout"], row["stderr"], engine, len(ids), args.steps)
                    except ValueError as error:
                        row["invalid_reasons"].append(str(error))
                row["valid_timing"] = not row["invalid_reasons"]
                failed |= not row["valid_timing"]
                write(row)
                rows[engine] = row
                if row["interrupted"]:
                    print(f"Stopped by user; receipt: {output}", file=sys.stderr)
                    return 130
                if row["heavy_exit"] in (3, 4):
                    print(f"Stopped on memory limit/pressure; receipt: {output}", file=sys.stderr)
                    return row["heavy_exit"]
            if all(row["valid_timing"] for row in rows.values()):
                tf, nano = (rows[engine]["result"] for engine in ("tf", "nanoattention"))
                comparison = {"kind": "engine_comparison_pair", "prompt": name, "pair": pair,
                              "agreement": output_agreement(tf["predictions"], nano["predictions"]),
                              "nano_over_tf_decode_rate": nano["decode_tokens_per_second"] / tf["decode_tokens_per_second"],
                              "nano_over_tf_prefill_rate": nano["prefill_tokens_per_second"] / tf["prefill_tokens_per_second"]}
                write(comparison)
                print(json.dumps(comparison), flush=True)
    print(f"Receipts: {output}", flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="code-512,code-1024")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--variant", choices=("reference", "optimized"), default="optimized")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pairs < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("pairs and timeout must be positive")
    try:
        return worker(args.worker) if args.worker else run(args)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"nanoattention: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
