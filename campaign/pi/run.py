"""Drive pi against the local TensorFold server: one sandboxed run per fixture, hidden checkers outside."""

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "bench"
sys.path.insert(0, str(BENCH))

import harness  # noqa: E402

PI = Path(os.environ.get("PI_BIN") or shutil.which("pi") or "/Users/josh/.local/bin/pi")
PI_FLAGS = ["--no-extensions", "--no-mcp", "--no-skills", "-nc", "--no-session", "--mode", "json"]
PI_TOOLS = "read,bash,edit,write"
SERVER_PORT = 8888
TASK_WALL_S = 600.0
SERVER_START_S = 1800.0


def metrics(base: str) -> dict[str, int]:
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=10) as response:
            text = response.read().decode("utf-8", errors="replace")
    except OSError:
        return {}
    counters = {}
    for line in text.splitlines():
        for name in ("kept_tool_calls_unoffered_total", "kept_tool_calls_malformed_total"):
            if line.startswith(f"tensorfold:{name} "):
                counters[name] = int(float(line.split()[-1]))
    return counters


def healthy(base: str) -> bool:
    return harness.get_json(f"{base}/health", timeout=3) is not None


def start_server(binary: Path, model: Path) -> tuple["harness.Server", bool]:
    base = f"http://127.0.0.1:{SERVER_PORT}"
    if healthy(base):
        return None, False
    original = harness.free_port
    harness.free_port = lambda: SERVER_PORT
    try:
        server = harness.Server(binary, model, [], 2, harness.TMP / "pi-server.log")
        server.start(timeout=SERVER_START_S)
    finally:
        harness.free_port = original
    if not healthy(base):
        server.stop()
        raise RuntimeError("server did not report healthy")
    return server, True


def load_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def summarize(events: list[dict]) -> dict:
    turns = []
    errors = []
    for event in events:
        if event.get("type") == "turn_end":
            message = event.get("message") or {}
            turns.append({"duration_ms": message.get("durationMs"), "output_tokens": (message.get("usage") or {}).get("output")})
        if event.get("type") in ("message_end", "turn_end"):
            message = event.get("message") or {}
            if message.get("errorMessage"):
                errors.append({"stop": message.get("stopReason"), "error": message.get("errorMessage")})
    completed = any(event.get("type") == "agent_end" for event in events)
    return {"completed": completed, "turns": len(turns), "turn_latency_ms": [t["duration_ms"] for t in turns],
            "turn_output_tokens": [t["output_tokens"] for t in turns], "pi_event_errors": errors}


def run_task(task_dir: Path, name: str, workspace_root: Path, agent_dir: Path, sandbox: Path,
             model_id: str) -> dict:
    workspace = workspace_root / name
    repo = workspace / "repo"
    tmp = workspace / "tmp"
    task_agent = workspace / "agent"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(task_dir / "repo", repo)
    for directory in (tmp, task_agent):
        directory.mkdir(parents=True)
    shutil.copy2(agent_dir / "models.json", task_agent / "models.json")
    (task_agent / "auth.json").write_text("{}\n", encoding="utf-8")
    (task_agent / "models-store.json").write_text("{}\n", encoding="utf-8")
    events_path = workspace / "events.jsonl"
    err_path = workspace / "pi.err.log"
    prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    argv = [str(PI), *PI_FLAGS, "--provider", "tensorfold", "--model", model_id, "--tools", PI_TOOLS]
    env = {k: v for k, v in os.environ.items() if not k.startswith("PI_")}
    env.update({"TMPDIR": str(tmp), "PI_CODING_AGENT_DIR": str(task_agent), "PI_OFFLINE": "1", "PI_TELEMETRY": "0"})
    command = ["sandbox-exec", "-f", str(sandbox), "-D", f"TASK_DIR={workspace}", "-D", f"WORK_TMP={harness.TMP}",
               "-D", f"TASKS_DIR={task_dir.parent}", *argv, "-p", prompt]
    started = time.perf_counter()
    timed_out = False
    with events_path.open("wb") as out, err_path.open("wb") as err:
        process = subprocess.Popen(command, cwd=repo, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                   start_new_session=True, env=env)
        try:
            exit_code = process.wait(timeout=TASK_WALL_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            exit_code = process.wait()
    wall_s = round(time.perf_counter() - started, 2)
    events = load_events(events_path)
    summary = summarize(events)
    checker = subprocess.run([sys.executable, str(task_dir / "check.py"), str(repo)], capture_output=True,
                             text=True, timeout=120, env={k: v for k, v in os.environ.items() if not k.startswith("PI_")})
    return {"task": name, "exit_code": exit_code, "timed_out": timed_out, "wall_s": wall_s,
            "completed": summary["completed"] and exit_code == 0 and not timed_out,
            "checker": {"pass": checker.returncode == 0, "detail": (checker.stdout + checker.stderr).strip()[-2000:]},
            "turns": summary["turns"], "turn_latency_ms": summary["turn_latency_ms"],
            "turn_output_tokens": summary["turn_output_tokens"], "pi_event_errors": summary["pi_event_errors"],
            "workspace": str(workspace)}


def count_log_parse_errors(log: Path) -> int:
    if not log.is_file():
        return 0
    return sum(1 for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
               if "malformed or unoffered tool envelopes" in line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="flint")
    parser.add_argument("--bin", default=str(harness.BASELINE_BIN))
    parser.add_argument("--only", default="", help="comma-separated task names")
    parser.add_argument("--pi-model", default="flint")
    args = parser.parse_args()

    campaign = Path(__file__).resolve().parent
    tasks_root = campaign / "tasks"
    agent_dir = campaign / "agent"
    sandbox = campaign / "sandbox.sb"
    results_dir = campaign / "results"
    model_key = harness.model_key(args.model)
    model = harness.model_dir(model_key)
    binary = Path(args.bin).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"binary {binary} is missing")

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    tasks = sorted(path for path in tasks_root.iterdir() if path.is_dir() and (not wanted or path.name in wanted))
    if not tasks:
        raise SystemExit("no tasks selected")

    server, owned = start_server(binary, model)
    base = f"http://127.0.0.1:{SERVER_PORT}"
    before = metrics(base)
    records = []
    try:
        for task_dir in tasks:
            record = run_task(task_dir, task_dir.name, harness.WORK / "pi", agent_dir, sandbox, args.pi_model)
            after = metrics(base)
            record["parse_errors"] = {
                "metrics_unoffered": after.get("kept_tool_calls_unoffered_total", 0) - before.get("kept_tool_calls_unoffered_total", 0),
                "metrics_malformed": after.get("kept_tool_calls_malformed_total", 0) - before.get("kept_tool_calls_malformed_total", 0),
                "pi_events": sum(1 for e in record["pi_event_errors"] if "parse" in (e.get("error") or "").lower()),
            }
            record["parse_errors"]["total"] = (record["parse_errors"]["metrics_unoffered"]
                                               + record["parse_errors"]["metrics_malformed"]
                                               + record["parse_errors"]["pi_events"])
            before = after
            records.append(record)
            print(f"{record['task']}: completed={record['completed']} checker={record['checker']['pass']} "
                  f"turns={record['turns']} parse_errors={record['parse_errors']['total']}", file=sys.stderr)
    finally:
        if owned and server is not None:
            server.stop()

    server_log = harness.TMP / "pi-server.log"
    totals = {"tasks": len(records), "completed": sum(1 for r in records if r["completed"]),
              "checker_passed": sum(1 for r in records if r["checker"]["pass"]),
              "parse_errors": sum(r["parse_errors"]["total"] for r in records),
              "server_log_parse_errors": count_log_parse_errors(server_log) if owned else None}
    record = {"schema": 1, "model": args.model, "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "host": harness.host_summary(),
              "pi": {"path": str(PI), "flags": PI_FLAGS, "tools": PI_TOOLS, "provider": "tensorfold", "model_id": args.pi_model},
              "server": {"base": base, "owned": owned, "argv": server.argv if server else None,
                         "model_ids": harness.get_json(f"{base}/v1/models") or None,
                         "binary": {"path": str(binary), "sha256": harness.sha256_file(binary)}},
              "model": {"key": model_key, "dir": str(model), "hashes": harness.model_hashes(model_key)},
              "agent": {"models_sha256": harness.sha256_file(agent_dir / "models.json")},
              "tasks": records, "totals": totals}
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{dt.datetime.now().astimezone().strftime('%Y-%m-%d')}-{args.model}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(totals, sort_keys=True))
    failed = [r["task"] for r in records if not r["completed"]]
    print(f"incomplete: {failed}" if failed else "all task runs completed", file=sys.stderr)
    return 0 if totals["completed"] == totals["tasks"] and totals["parse_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
