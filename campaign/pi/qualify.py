"""Qualify a Pi harness result: complete coverage, valid runs, protocol and checker outcomes, allowed loss."""

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ALLOWED_LOSS = 1
EXPECTED_TOOLS = "read,bash,edit,write"
EXPECTED_PROVIDER = "tensorfold"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_problems(record: dict) -> list[str]:
    tasks = record.get("tasks") or []
    totals = record.get("totals") or {}
    problems = []
    if not tasks:
        problems.append("no task records")
    names = [task.get("task") for task in tasks]
    if len(names) != len(set(names)):
        problems.append("duplicate task names")
    if totals.get("tasks") != len(tasks):
        problems.append(f"totals say {totals.get('tasks')} tasks, {len(tasks)} recorded")
    recorded_completed = sum(1 for task in tasks if task.get("completed"))
    if totals.get("completed") != recorded_completed:
        problems.append(f"totals say {totals.get('completed')} completed, {recorded_completed} recorded")
    complete = (record.get("completion") or {}).get("complete")
    if complete is not True:
        problems.append("the runner did not report complete coverage")
    return problems


def protocol_problems(record: dict) -> list[str]:
    pi = record.get("pi") or {}
    server = record.get("server") or {}
    problems = []
    if pi.get("provider") != EXPECTED_PROVIDER:
        problems.append(f"provider is {pi.get('provider')!r}, not {EXPECTED_PROVIDER!r}")
    if pi.get("tools") != EXPECTED_TOOLS:
        problems.append(f"tools are {pi.get('tools')!r}, not {EXPECTED_TOOLS!r}")
    if not pi.get("model_id"):
        problems.append("no pi model id recorded")
    if server.get("owned") is not True:
        problems.append("the server was not owned by the campaign, so its identity cannot be attributed")
    binary = (server.get("binary") or {}).get("sha256")
    if not binary:
        problems.append("no served binary hash recorded")
    identity = server.get("identity") or {}
    if not identity.get("model_ids"):
        problems.append("served identity was not captured while the process lived")
    elif pi.get("model_id") and pi["model_id"] not in identity["model_ids"]:
        problems.append(f"served identity {identity['model_ids']} does not hold pi's model {pi['model_id']!r}")
    base = server.get("base") or ""
    rendered = ((record.get("agent") or {}).get("rendered") or {}).get("providers", {}).get("tensorfold", {})
    base_url = rendered.get("baseUrl") or ""
    if base and base_url and urlsplit(base_url).netloc != urlsplit(base).netloc:
        problems.append(f"rendered models.json points at {base_url}, not the campaign server {base}")
    if totals_parse(record) != 0:
        problems.append(f"{totals_parse(record)} tool-call parse errors")
    return problems


def totals_parse(record: dict) -> int:
    return (record.get("totals") or {}).get("parse_errors") or 0


def run_problems(record: dict) -> list[str]:
    problems = []
    for task in record.get("tasks") or []:
        name = task.get("task")
        if task.get("timed_out"):
            problems.append(f"{name} timed out")
        if task.get("exit_code") not in (0, None) or task.get("completed") is not True:
            problems.append(f"{name} did not complete (exit {task.get('exit_code')}, completed {task.get('completed')})")
    return problems


def qualifies(candidate: dict, baseline: dict | None) -> dict:
    problems = coverage_problems(candidate) + protocol_problems(candidate) + run_problems(candidate)
    base_tasks = {task.get("task"): task for task in (baseline or {}).get("tasks") or []}
    cand_tasks = {task.get("task"): task for task in candidate.get("tasks") or []}
    if baseline is not None:
        if baseline.get("model") != candidate.get("model"):
            problems.append(f"baseline model {baseline.get('model')!r} is not {candidate.get('model')!r}")
        missing = sorted(set(base_tasks) - set(cand_tasks))
        extra = sorted(set(cand_tasks) - set(base_tasks))
        if missing:
            problems.append(f"tasks not run: {missing}")
        if extra:
            problems.append(f"tasks outside the baseline: {extra}")
    if problems:
        return {"qualified": False, "problems": problems, "lost": []}
    failed = sorted(name for name, task in cand_tasks.items() if not (task.get("checker") or {}).get("pass"))
    if baseline is None:
        if failed:
            problems.append(f"checkers failed without a baseline to measure loss against: {failed}")
    else:
        lost = sorted(name for name in base_tasks
                      if (base_tasks[name].get("checker") or {}).get("pass")
                      and not (cand_tasks[name].get("checker") or {}).get("pass"))
        if len(lost) > ALLOWED_LOSS:
            problems.append(f"{len(lost)} checker cases lost, at most {ALLOWED_LOSS} allowed: {lost}")
        failed = lost
    return {"qualified": not problems, "problems": problems, "lost": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="pi harness result JSON to qualify")
    parser.add_argument("--baseline", default=None, help="frozen baseline result JSON; loss is measured against it")
    args = parser.parse_args()
    candidate_path = Path(args.result)
    if not candidate_path.is_file():
        raise SystemExit(f"{candidate_path} is missing")
    baseline = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_file():
            raise SystemExit(f"{baseline_path} is missing")
        baseline = load(baseline_path)
    verdict = qualifies(load(candidate_path), baseline)
    if verdict["qualified"]:
        print(f"qualified: {candidate_path} with {len(verdict['lost'])} checker cases lost "
              f"(allowed {ALLOWED_LOSS})")
    else:
        print(f"not qualified: {candidate_path}", file=sys.stderr)
        for problem in verdict["problems"]:
            print(f"  {problem}", file=sys.stderr)
    return 0 if verdict["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
