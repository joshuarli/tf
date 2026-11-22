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
TASK_WALL_S = 600.0
SERVER_START_S = 1800.0


def needs_vision(task_dir: Path) -> bool:
    """Whether a task's prompt or fixture carries an image path, i.e. content a text-only model cannot read."""
    prompt = (task_dir / "prompt.txt").read_text(encoding="utf-8", errors="replace")
    if any(marker in prompt.lower() for marker in ("image", "screenshot", ".png", ".jpg", ".jpeg")):
        return True
    for path in (task_dir / "repo").rglob("*"):
        if path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            return True
    return False


def applicable(task_dirs: list[Path], key: str) -> tuple[list[Path], list[dict]]:
    """Split Pi tasks into runnable ones and the ones a text-only model excludes with a recorded reason."""
    if key != "inkling-2-mini":
        return list(task_dirs), []
    kept, excluded = [], []
    for task_dir in task_dirs:
        if needs_vision(task_dir):
            excluded.append({"task": task_dir.name,
                             "reason": "Inkling 2 Mini is text-only in this campaign: image content parts and "
                                       "--vision are refused (zig/src/families/inkling2_loop/refuse.zig)"})
        else:
            kept.append(task_dir)
    return kept, excluded


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


def served_identity(base: str) -> dict:
    """What the live endpoint reports about itself: served model ids and their model objects."""
    document = harness.get_json(f"{base}/v1/models") or {}
    data = [entry for entry in document.get("data") or [] if isinstance(entry, dict)]
    return {"model_ids": [entry.get("id") for entry in data], "models": data}


def identity_problems(identity: dict, *, binary_sha256: str, model_key: str, expected_model: str) -> list[str]:
    """Reasons a live endpoint cannot be attributed to the requested binary and model."""
    problems = []
    if not identity.get("model_ids"):
        problems.append("endpoint listed no model id")
    entries = identity.get("models") or []
    owned = [entry.get("owned_by") for entry in entries]
    if any(value not in (None, "tensorfold") for value in owned):
        problems.append(f"endpoint is not owned by tensorfold: owned_by {owned}")
    served = identity.get("model_ids") or []
    if expected_model and served and expected_model not in served:
        problems.append(f"endpoint serves {served}, not the requested {expected_model!r}")
    if model_key and not any(harness.served_model_matches(model_key, model_id) for model_id in served):
        problems.append(f"endpoint model ids {served} do not name model {model_key!r}")
    return problems


def served_context(argv: list[str]) -> int | None:
    """The token context a server was started with, when its argv sets one."""
    for flag, value in zip(argv, argv[1:]):
        if flag == "--context":
            return int(value)
    return None


def render_models(template: Path, base: str, model_id: str, context: int | None = None) -> dict:
    """Point the Pi model template at the server; a served context bounds Pi's window and output budget."""
    document = json.loads(template.read_text(encoding="utf-8"))
    provider = document["providers"]["tensorfold"]
    provider["baseUrl"] = f"{base}/v1"
    models = provider.get("models") or []
    if models:
        models[0]["id"] = model_id
        if context is not None:
            models[0]["contextWindow"] = min(models[0].get("contextWindow", context), context)
            models[0]["maxTokens"] = min(models[0].get("maxTokens", context), context // 4)
    return document


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
    last_turn = next((event.get("message") or {} for event in reversed(events) if event.get("type") == "turn_end"), {})
    completed = any(event.get("type") == "agent_end" for event in events) and not last_turn.get("errorMessage")
    return {"completed": completed, "turns": len(turns), "turn_latency_ms": [t["duration_ms"] for t in turns],
            "turn_output_tokens": [t["output_tokens"] for t in turns], "pi_event_errors": errors}


def run_task(task_dir: Path, name: str, workspace_root: Path, models_document: dict, sandbox: Path,
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
    (task_agent / "models.json").write_text(json.dumps(models_document, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
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
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    campaign = Path(__file__).resolve().parent
    tasks_root = campaign / "tasks"
    template = campaign / "agent" / "models.json"
    sandbox = campaign / "sandbox.sb"
    results_dir = campaign / "results"
    model_key = harness.model_key(args.model)
    model = harness.model_dir(model_key)
    binary = Path(args.bin).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"binary {binary} is missing")
    binary_sha256 = harness.sha256_file(binary)

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    tasks = sorted(path for path in tasks_root.iterdir() if path.is_dir() and (not wanted or path.name in wanted))
    if not tasks:
        raise SystemExit("no tasks selected")
    tasks, excluded = applicable(tasks, model_key)
    if not tasks:
        raise SystemExit(f"every selected task needs images, which {model_key} refuses: {excluded}")

    stamp = dt.datetime.now(dt.timezone.utc)
    run_id = args.run_id or f"{args.model}-{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    log_path = harness.TMP / f"pi-server-{run_id}.log"
    server = harness.Server(binary, model, harness.default_serve_args(model_key), 2, log_path)
    server.start(timeout=SERVER_START_S)
    base = server.base
    try:
        identity = served_identity(base)
        problems = identity_problems(identity, binary_sha256=binary_sha256, model_key=model_key,
                                     expected_model=None)
        if problems:
            raise SystemExit(f"server at {base} cannot be attributed to {binary}: {'; '.join(problems)}")
        served_model = identity["model_ids"][0]
        if not harness.served_model_matches(model_key, served_model):
            raise SystemExit(f"server at {base} serves {served_model!r}, not a {harness.served_identity_name(model_key)} "
                             "endpoint; the run cannot be attributed to this model")
        models_document = render_models(template, base, served_model, served_context(server.argv))
        before = metrics(base)
        records = []
        server_exit = None
        for task_dir in tasks:
            record = run_task(task_dir, task_dir.name, harness.WORK / "pi", models_document, sandbox, served_model)
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
            server_exit = server.process.poll()
            if server_exit is not None:
                record["completed"] = False
                record["server_exit"] = server_exit
            print(f"{record['task']}: completed={record['completed']} checker={record['checker']['pass']} "
                  f"turns={record['turns']} parse_errors={record['parse_errors']['total']}", file=sys.stderr)
            if server_exit is not None:
                print(f"server exited with status {server_exit} during {record['task']} (see {log_path}); "
                      "stopping the run", file=sys.stderr)
                break
        live_identity = served_identity(base)
    finally:
        server.stop()

    totals = {"tasks": len(tasks), "completed": sum(1 for r in records if r["completed"]),
              "checker_passed": sum(1 for r in records if r["checker"]["pass"]),
              "parse_errors": sum(r["parse_errors"]["total"] for r in records),
              "server_log_parse_errors": count_log_parse_errors(log_path)}
    record = {"schema": 1, "model": args.model, "recorded_at": stamp.isoformat(), "run_id": run_id,
              "host": harness.host_summary(model_key),
              "pi": {"path": str(PI), "flags": PI_FLAGS, "tools": PI_TOOLS, "provider": "tensorfold",
                     "model_id": served_model},
              "server": {"base": base, "exit": server_exit, "port": server.port, "owned": True, "argv": server.argv,
                         "identity": live_identity,
                         "binary": {"path": str(binary), "sha256": binary_sha256}},
              "model": {"key": model_key, "dir": str(model), "hashes": harness.model_hashes(model_key)},
              "agent": {"template": str(template.relative_to(campaign)),
                        "template_sha256": harness.sha256_file(template),
                        "rendered": models_document},
              "tasks": records, "totals": totals, **({"excluded": excluded} if excluded else {}),
              "completion": {"complete": totals["completed"] == totals["tasks"],
                             "checkers_passed": totals["checker_passed"] == totals["tasks"]},
              "qualification": "run `pi-qualify --result <this file>`; completion alone does not pass checkers"}
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{run_id}.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest = results_dir / f"latest-{args.model}.json"
    latest.write_text(json.dumps({"run_id": run_id, "result": str(out.relative_to(campaign))}, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps(totals, sort_keys=True))
    failed = [r["task"] for r in records if not r["completed"]] + [t.name for t in tasks[len(records):]]
    print(f"incomplete: {failed}" if failed else "all task runs completed", file=sys.stderr)
    return 0 if totals["completed"] == totals["tasks"] else 1


if __name__ == "__main__":
    sys.exit(main())
