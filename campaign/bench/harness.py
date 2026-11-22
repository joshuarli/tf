"""Shared model, hashing, server-lifecycle, HTTP and host-condition helpers for the campaign benchmark tools."""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from context_policy import CONTEXT_TOKENS, context_problem, rendered_prompt_tokens

REPO = Path(__file__).resolve().parents[2]


def _main_root() -> Path:
    """The checkout that owns .work, which is the main clone even when this code runs in a lane worktree."""
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=REPO,
                            capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


MAIN = _main_root()
CAMPAIGN = REPO / "campaign"
WORK = MAIN / ".work"
GPU_LOCK = WORK / "gpu.lock"
HEAVY = Path(__file__).resolve().parents[1] / "heavy.py"
TMP = WORK / "tmp"
TOOLS = REPO / "tools"
BASELINE_BIN = WORK / "wt" / "baseline" / "zig-out" / "native" / "bin" / "tensorfold-native"
SOURCES_LOCK = CAMPAIGN / "sources.lock.json"
HOST_RECORD = CAMPAIGN / "host.json"
PROMPTS_DIR = CAMPAIGN / "prompts"
PROMPT_MANIFEST = PROMPTS_DIR / "manifest.json"
RUNS_DIR = CAMPAIGN / "runs"

REPO_NAMES = {"flint-int4": "flint-int4", "flint-dflash2": "Flint-DFlash2", "inkling-2-mini": "Inkling-2-Mini"}
MODEL_KEYS = {"flint": "flint-int4", "inkling2-mini": "inkling-2-mini", "inkling-2-mini": "inkling-2-mini"}
LOAD_LIMIT_VARS = {"flint-int4": "FLINT_LOAD_LIMIT_GB", "inkling-2-mini": "INKLING_LOAD_LIMIT_GB"}
UNSUPPORTED_MODES = {
    "inkling-2-mini": {
        "copy": "Inkling 2 Mini is text-only in this campaign and has no copy-draft path; run mode ar",
        "drafter": "Inkling 2 Mini has no served drafter in this campaign (its mtp.safetensors head is not wired); "
                   "run mode ar",
    },
}
DEFAULT_SERVE_ARGS = {"flint-int4": [], "inkling-2-mini": ["--no-drafts", "--temperature", "0"]}

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import bench_cache  # noqa: E402
import bench_metrics  # noqa: E402
import bench_stream  # noqa: E402
import bench_timing  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sources_lock() -> dict:
    return json.loads(SOURCES_LOCK.read_bytes())


def model_key(model: str) -> str:
    key = MODEL_KEYS.get(model)
    if key is None:
        raise ValueError(f"unknown MODEL {model!r}; expected one of {sorted(MODEL_KEYS)}")
    return key


def model_entries(key: str) -> list[dict]:
    return [entry for entry in sources_lock()["inputs"] if entry["model"] == key]


def model_dir(key: str) -> Path:
    entries = model_entries(key)
    if not entries:
        raise ValueError(f"no pinned inputs for {key}")
    revision = entries[0]["version"][:12]
    return Path.home() / "models" / f"{REPO_NAMES[key]}@{revision}"


CONFIG_NAMES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "processor_config.json",
                "model.safetensors.index.json", "hf_quant_config.json", "tokenizer.model")


def model_hashes(key: str) -> dict:
    entries = model_entries(key)
    rows = [{"name": entry["name"], "sha256": entry["sha256"], "size": entry["size"]} for entry in entries]
    weights = sorted((row for row in rows if row["name"].endswith(".safetensors")), key=lambda row: row["name"])
    configs = sorted((row for row in rows if row["name"].rsplit("/", 1)[-1] in CONFIG_NAMES), key=lambda row: row["name"])
    other = sorted((row for row in rows if row not in weights and row not in configs), key=lambda row: row["name"])
    aggregate = hashlib.sha256(json.dumps(
        {"revision": entries[0]["version"], "weights": weights, "configs": configs, "other": other},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"revision": entries[0]["version"], "weights": weights, "config": configs, "other": other,
            "aggregate_sha256": aggregate, "sources_lock_sha256": sha256_file(SOURCES_LOCK)}


def drafter_path() -> Path | None:
    key = "flint-dflash2"
    entries = [entry for entry in sources_lock()["inputs"] if entry["model"] == key]
    if not entries:
        return None
    path = Path.home() / "models" / f"{REPO_NAMES[key]}@{entries[0]['version'][:12]}"
    return path if (path / "config.json").is_file() and (path / "model.safetensors").is_file() else None


def load_limit_var(key: str = "flint-int4") -> str:
    if key not in LOAD_LIMIT_VARS:
        raise ValueError(f"no load-limit variable for {key!r}; expected one of {sorted(LOAD_LIMIT_VARS)}")
    return LOAD_LIMIT_VARS[key]


def load_limit_gb(key: str = "flint-int4") -> int:
    sys.path.insert(0, str(HEAVY.parent))
    import heavy
    return getattr(heavy, load_limit_var(key))


def model_mode_problem(key: str, mode: str) -> str | None:
    """The recorded reason a (model, mode) pair is refused, or None when the model serves that mode."""
    return (UNSUPPORTED_MODES.get(key) or {}).get(mode)


def default_serve_args(key: str) -> list[str]:
    """The flags every server for a model passes: Inkling 2 Mini serves greedy ar only, so its default temperature is 0 for clients (Pi) that send none."""
    return list(DEFAULT_SERVE_ARGS.get(key, []))


def model_key_for_dir(path: Path) -> str | None:
    """The locked model key whose pinned directory is ``path``, or None for a comparator's own checkpoint."""
    resolved = Path(path).expanduser().resolve()
    for key in REPO_NAMES:
        if key == "flint-dflash2":
            continue
        try:
            if model_dir(key).resolve() == resolved:
                return key
        except ValueError:
            continue
    return None


def host_summary(key: str = "flint-int4") -> dict:
    record = json.loads(HOST_RECORD.read_bytes())
    sysctl = record["sysctl"]
    return {"product": sysctl["brand"], "performance_cores": sysctl["perflevel0_physicalcpu"],
            "efficiency_cores": sysctl["perflevel1_physicalcpu"], "memory_bytes": sysctl["memsize_bytes"],
            "gpu_cores": record["gpu"]["gpu_cores"], "metal": record["gpu"]["metal_support"],
            "os": f"{record['sw_vers']['product_name']} {record['sw_vers']['build_version']}",
            "recommended_max_working_set_bytes": int(record["metal"]["recommendedMaxWorkingSetSize"]),
            "load_limit_gb": load_limit_gb(key), "context_tokens": CONTEXT_TOKENS}


def vm_pressure_level() -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True,
                                  text=True, check=True).stdout.strip())
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def free_percent() -> int | None:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    for line in out.splitlines():
        if "memory free percentage" in line:
            words = line.replace("%", " ").split()
            for word in reversed(words):
                if word.isdigit():
                    return int(word)
    return None


def therm() -> dict:
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return {"cpu_speed_limit": None, "warnings": None, "raw": None}
    speed, warnings = None, False
    for line in out.splitlines():
        if "CPU_Speed_Limit" in line:
            digits = "".join(ch for ch in line.split("=")[-1] if ch.isdigit())
            speed = int(digits) if digits else None
        if "warning level" in line and "No thermal warning level" not in line and "No performance warning level" not in line:
            warnings = True
    return {"cpu_speed_limit": speed, "warnings": warnings, "raw": out.strip()}


GPU_CLIENT = re.compile(r'"accumulatedGPUTime"=(\d+)')
GPU_CREATOR = re.compile(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"')


def gpu_time_by_pid(*, strict: bool = False) -> dict[int, tuple[str, int]]:
    """Every Metal client's accumulated GPU nanoseconds, summed per process, from IOKit."""
    out = subprocess.run(["ioreg", "-r", "-w", "0", "-l", "-c", "AGXDeviceUserClient"],
                         capture_output=True, text=True, check=strict).stdout
    totals: dict[int, tuple[str, int]] = {}
    pending = 0
    for line in out.splitlines():
        if '"AppUsage" = ' in line:
            pending = sum(int(value) for value in GPU_CLIENT.findall(line))
        elif match := GPU_CREATOR.search(line):
            pid, name = int(match.group(1)), match.group(2)
            totals[pid] = (name, totals.get(pid, (name, 0))[1] + pending)
            pending = 0
    return totals


def gpu_time_limit_pct() -> float:
    """The other-process GPU-time limit, shared with gpu-gate-ready so one setting governs both checks."""
    return float(os.environ.get("GPU_GATE_MAX_GPU_PCT", "5"))


def gpu_contention(before: dict[int, tuple[str, int]], after: dict[int, tuple[str, int]], seconds: float,
                   own: set[int], limit_pct: float | None = None) -> tuple[list[dict], list[str]]:
    """GPU time other processes used over a timed window, and the invalid reason it implies."""
    if limit_pct is None:
        limit_pct = gpu_time_limit_pct()
    users = []
    for pid, (name, ns) in after.items():
        if pid in own:
            continue
        used = ns - before.get(pid, (name, ns))[1]
        if used > 0:
            users.append({"pid": pid, "name": name, "gpu_s": used / 1e9})
    total = sum(user["gpu_s"] for user in users)
    pct = 100 * total / seconds if seconds > 0 else 0.0
    reasons = [f"other processes used {pct:.1f}% GPU time during the trial "
               f"({', '.join(f'{u['name']} pid={u['pid']}' for u in sorted(users, key=lambda u: -u['gpu_s'])[:3])})"] \
        if pct > limit_pct else []
    return users, reasons


def process_tree(root: int) -> set[int]:
    out = subprocess.run(["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    for line in out.splitlines():
        pid, ppid = (int(field) for field in line.split())
        children.setdefault(ppid, []).append(pid)
    found, stack = set(), [root]
    while stack:
        pid = stack.pop()
        found.add(pid)
        stack.extend(children.get(pid, []))
    return found


def gpu_gate(own: set[int] | None = None) -> tuple[dict, list[str]]:
    """`gpu-gate-ready -1`: other GPU users, utilization, pressure, swap and compressor. After a trial, `own`
    names the server's processes, whose GPU time and trailing utilization are the trial's own."""
    env = dict(os.environ, GPU_GATE_WAIT="0")
    if own:
        env["GPU_GATE_IGNORE_PIDS"] = " ".join(str(pid) for pid in sorted(own))
    try:
        run = subprocess.run(["gpu-gate-ready", "-1"], capture_output=True, text=True, env=env)
    except FileNotFoundError:
        return {"status": "unavailable"}, []
    detail = (run.stderr or run.stdout).strip()
    if run.returncode == 0:
        return {"status": "quiet"}, []
    if run.returncode == 75:
        return {"status": "contended", "detail": detail}, [detail.splitlines()[0]]
    return {"status": f"error {run.returncode}", "detail": detail}, []


def conditions(own: set[int] | None = None) -> dict:
    level = vm_pressure_level()
    gate, gate_reasons = gpu_gate(own)
    pressure = {"vm_pressure_level": level, "free_percent": free_percent(), "therm": therm(), "gpu_gate": gate}
    reasons = list(gate_reasons)
    if level is not None and level != 1:
        reasons.append(f"memory pressure level {level}")
    speed = pressure["therm"]["cpu_speed_limit"]
    if speed is not None and speed < 100:
        reasons.append(f"CPU speed limit {speed}%")
    if pressure["therm"]["warnings"]:
        reasons.append("thermal or performance warning recorded")
    return {"pressure": pressure, "pressure_reasons": reasons}


class LogReader:
    """One server's output: every line kept, the first line's arrival time, and the log on disk."""

    def __init__(self, stream, path: Path):
        self.stream, self.path = stream, path
        self.lines: list[str] = []
        self.first_at: float | None = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            with self.path.open("wb") as log:
                for raw in self.stream:
                    now = time.perf_counter()
                    with self.lock:
                        if self.first_at is None:
                            self.first_at = now
                        self.lines.append(raw.decode("utf-8", errors="replace").rstrip("\n"))
                    log.write(raw)
                    log.flush()
        except (OSError, ValueError):
            pass

    def matching(self, needle: str) -> str | None:
        with self.lock:
            for line in reversed(self.lines):
                if needle in line:
                    return line
        return None

    def text(self) -> str:
        with self.lock:
            return "\n".join(self.lines)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def get_json(url: str, timeout: float = 30) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError):
        return None


def parse_seconds(pattern_line: str | None, marker: str) -> float | None:
    if not pattern_line or marker not in pattern_line:
        return None
    tail = pattern_line.split(marker, 1)[1]
    number = ""
    for ch in tail:
        if ch.isdigit() or ch == ".":
            number += ch
        else:
            break
    try:
        return float(number)
    except ValueError:
        return None


def served_identity_name(key: str) -> str:
    """The model name a served endpoint should report: the repository basename without its revision."""
    return REPO_NAMES[key]


def served_model_matches(key: str, model_id: str | None) -> bool:
    """Whether a served model id names ``key``; the served name is the checkpoint directory's basename."""
    if not model_id:
        return False
    return str(model_id).lower().startswith(served_identity_name(key).lower())


def load_limit_env(model: Path) -> dict[str, str]:
    """The one load-limit variable a server for ``model`` reads, set to its campaign value."""
    key = model_key_for_dir(model)
    if key is None:
        return {load_limit_var("flint-int4"): str(load_limit_gb("flint-int4"))}
    return {load_limit_var(key): str(load_limit_gb(key))}


class Server:
    """A model server under the GPU lock: ready wait, log parsing, /memory sampling and a group-only stop."""

    def __init__(self, binary: Path, model: Path, extra_args: list[str], prompt_cache_gib: int, log_path: Path):
        if any(arg == "--context" or arg.startswith("--context=") for arg in extra_args):
            raise ValueError(f"campaign servers fix --context at {CONTEXT_TOKENS}; remove the override")
        self.binary, self.model, self.extra_args = Path(binary), Path(model), list(extra_args)
        self.prompt_cache_gib = prompt_cache_gib
        self.log_path = Path(log_path)
        self.env = dict(os.environ, **load_limit_env(self.model), TENSORFOLD_NO_UPDATE_CHECK="1")
        self.port: int | None = None
        self.base: str | None = None
        self.process: subprocess.Popen | None = None
        self.reader: LogReader | None = None
        self.launch_t = self.ready_t = None
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None

    @property
    def argv(self) -> list[str]:
        return [sys.executable, str(HEAVY), "--", str(self.binary), "serve", str(self.model), "--host", "127.0.0.1",
                "--port", str(self.port), "--parallel", "1", "--prompt-cache-gib", str(self.prompt_cache_gib),
                "--no-update-check", "--context", str(CONTEXT_TOKENS), *self.extra_args]

    def start(self, timeout: float = 1800) -> None:
        TMP.mkdir(parents=True, exist_ok=True)
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.launch_t = time.perf_counter()
            self.process = subprocess.Popen(self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            start_new_session=True, env=self.env)
            self.reader = LogReader(self.process.stdout, self.log_path)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited before ready ({self.process.returncode}): "
                                       f"{self.reader.text()[-2000:]}")
                if get_json(f"{self.base}/health", timeout=2) is not None:
                    self.ready_t = time.perf_counter()
                    self._sampler = threading.Thread(target=self._sample, daemon=True)
                    self._sampler.start()
                    return
                time.sleep(0.05)
            raise TimeoutError("server did not report healthy before the startup timeout")
        except BaseException:
            self.stop()
            raise

    def _sample(self) -> None:
        while not self._stop.is_set():
            if self.base is not None:
                value = get_json(f"{self.base}/memory", timeout=5)
                if value is not None:
                    self.samples.append({"t": time.perf_counter(), "memory": value})
            self._stop.wait(0.25)

    def snapshot(self) -> dict:
        if self.base is None:
            return {}
        return get_json(f"{self.base}/memory", timeout=10) or {}

    def serving_line(self) -> str | None:
        return self.reader.matching("serving ") if self.reader else None

    def engine_load_s(self) -> float | None:
        return parse_seconds(self.serving_line(), "loaded in ")

    def timing(self) -> dict:
        lock_wait = startup = None
        if self.launch_t is not None and self.reader is not None and self.reader.first_at is not None:
            lock_wait = self.reader.first_at - self.launch_t
        if self.reader is not None and self.reader.first_at is not None and self.ready_t is not None:
            startup = self.ready_t - self.reader.first_at
        return {"lock_wait_s": round(lock_wait, 3) if lock_wait is not None else None,
                "startup_s": round(startup, 3) if startup is not None else None,
                "engine_load_s": self.engine_load_s(),
                "spawn_to_ready_s": round(self.ready_t - self.launch_t, 3)
                if self.ready_t is not None and self.launch_t is not None else None}

    def models(self) -> list[str]:
        value = get_json(f"{self.base}/v1/models") or {}
        return [entry.get("id") for entry in value.get("data") or [] if isinstance(entry, dict)]

    def sample_mark(self) -> int:
        return len(self.samples)

    def stats(self, since: int = 0) -> dict:
        """Footprint over the samples taken since ``since``; the kernel lifetime peak covers the whole process."""
        window = list(self.samples[since:])
        physical = []
        lifetime = device_peak = 0
        for sample in window:
            process = sample["memory"].get("process") or {}
            if isinstance(process.get("physical_footprint_bytes"), int):
                physical.append(process["physical_footprint_bytes"])
                lifetime = max(lifetime, process.get("lifetime_peak_physical_footprint_bytes") or 0)
            device = sample["memory"].get("device") or {}
            if isinstance(device.get("peak_bytes"), int):
                device_peak = max(device_peak, device["peak_bytes"])
        steady = sorted(physical)
        return {"ready_bytes": physical[0] if physical else None,
                "steady_bytes": steady[len(steady) // 2] if steady else None,
                "peak_bytes": max(physical) if physical else None,
                "process_lifetime_peak_bytes": lifetime or None,
                "device_peak_bytes": device_peak or None,
                "samples": len(physical), "source": "GET /memory (proc_pid_rusage physical footprint)"}

    def stop(self) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=3)
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), 15)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), 9)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=20)
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def chat_request(base: str, model: str, content: str, *, max_tokens: int = 512, temperature: float = 0,
                 ignore_eos: bool = True, seed: int | None = 1234, timeout: float = 1800) -> dict:
    """One streamed greedy chat request, timed on the client clock, with the server's own counters kept."""
    body = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos,
            "messages": [{"role": "user", "content": content}]}
    if seed is not None and temperature == 0:
        body["seed"] = seed
    request = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    sent = time.perf_counter()
    collector = bench_stream.Collector(sent)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        collector.read(response)
    completed = time.perf_counter()
    if collector.end == "error":
        raise RuntimeError("server reported an error for the request")
    complete = collector.end == "done"
    observed = collector.evidence.finish(collector.usage, collector.runtime, complete)
    pieces = collector.pieces
    first, last = (pieces[0][0], pieces[-1][0]) if pieces else (None, None)
    out = {"sent": sent, "completed": completed, "first": first, "last": last, "pieces": pieces,
           "ttft_s": (first - sent) if complete and first is not None else None,
           "decode_s": (last - first) if complete and first is not None and last > first else None,
           "total_s": (completed - sent) if complete else None, "complete": complete,
           "completion": collector.completion(), "finish_reasons": collector.receipt()["finish_reasons"],
           "runtime": collector.runtime, "speculative": collector.speculative or {},
           **observed}
    tokens = observed["tokens"]
    decode_s = out["decode_s"]
    out["decode_tps"] = (tokens - 1) / decode_s if tokens and tokens > 1 and decode_s else None
    out.update(bench_metrics.request(out))
    out.update(bench_timing.request(out))
    return out


def request_record(run: dict, phase: str, reference: dict | None = None) -> dict:
    """One request's exported evidence (bench_cache's schema) plus the engine and timing figures it omits."""
    record = {key: value for key, value in bench_cache.receipt(run, phase, reference).items()
              if key not in ("chunk_offsets_s", "chunk_gaps_ms")}
    extra = ("decode_s", "decode_tps", "total_s", "completion", "finish_reasons", "prompt_ttft_tps",
             "itl_p50_ms", "itl_p95_ms", "tpot_ms", "round_gap_p50_ms", "round_gap_p95_ms",
             "chunk_tpot_estimate_ms", "tokens_per_chunk", "token_timing")
    record.update({key: run.get(key) for key in extra})
    record["completion_tokens"] = run.get("tokens")
    prompt, cached = run.get("prompt_tokens"), run.get("cached_tokens")
    record["uncached_tokens"] = prompt - cached if isinstance(prompt, int) and isinstance(cached, int) else None
    runtime = run.get("runtime") or {}
    record["engine_prefill_s"] = runtime.get("prefill_seconds")
    record["engine_ttft_s"] = runtime.get("time_to_first_token")
    record["engine_decode_s"] = runtime.get("seconds")
    record["engine_tokens_per_second"] = runtime.get("tokens_per_second")
    record["engine_prefill_tps"] = (prompt / runtime["prefill_seconds"]
                                    if isinstance(prompt, int) and isinstance(runtime.get("prefill_seconds"), (int, float))
                                    and runtime["prefill_seconds"] > 0 else None)
    record["synchronized"] = "client perf_counter, request sent to token arrival"
    speculative = run.get("speculative") or {}
    record["speculative"] = speculative
    rounds = speculative.get("rounds")
    record["committed_per_verification"] = (run["tokens"] / rounds
                                            if isinstance(rounds, int) and rounds > 0 and isinstance(run.get("tokens"), int)
                                            else None)
    record["acceptance_rate"] = speculative.get("acceptance_rate")
    record["drafted_tokens"] = speculative.get("drafted")
    record["accepted_tokens"] = speculative.get("accepted")
    record["verification_rounds"] = rounds
    record["tokens_per_round"] = speculative.get("tokens_per_round")
    return record
