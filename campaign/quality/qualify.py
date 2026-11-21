"""Check one quality run against Decision 10's allowed loss and the protocol, against a frozen baseline result."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOWED_LOSS = 1


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def covered(record: dict) -> tuple[bool, list[str]]:
    cases = record.get("cases") or []
    totals = record.get("totals") or {}
    problems = []
    if not cases:
        problems.append("no cases recorded")
    if totals.get("cases") != len(cases):
        problems.append(f"totals say {totals.get('cases')} cases, {len(cases)} recorded")
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        problems.append("duplicate case ids")
    if any(case.get("passed") not in (True, False) for case in cases):
        problems.append("a case has no pass/fail outcome")
    for case in cases:
        if case.get("reason", "").startswith("request failed") or case.get("reason", "").startswith("scoring failed"):
            problems.append(f"{case.get('id')} is an invalid run: {case.get('reason')}")
    suite = record.get("suite") or {}
    if suite.get("cases") != len(cases):
        problems.append(f"suite declares {suite.get('cases')} cases, {len(cases)} recorded")
    return not problems, problems


def protocol_problems(record: dict) -> list[str]:
    protocol = record.get("protocol") or {}
    problems = []
    for field, expected in (("temperature", 0), ("seed", 1234), ("max_tokens", 8192)):
        if protocol.get(field) != expected:
            problems.append(f"protocol {field} is {protocol.get(field)!r}, expected {expected!r}")
    if record.get("invalid_reasons"):
        problems.append(f"run recorded invalid reasons: {record['invalid_reasons']}")
    return problems


def qualifies(candidate: dict, baseline: dict) -> dict:
    ok, problems = covered(candidate)
    problems += protocol_problems(candidate)
    if baseline.get("set") != candidate.get("set"):
        problems.append(f"baseline set {baseline.get('set')!r} is not {candidate.get('set')!r}")
    base_cases = {case["id"]: case for case in baseline.get("cases") or []}
    cand_cases = {case["id"]: case for case in candidate.get("cases") or []}
    missing = sorted(set(base_cases) - set(cand_cases))
    extra = sorted(set(cand_cases) - set(base_cases))
    if missing:
        problems.append(f"cases not scored: {missing}")
    if extra:
        problems.append(f"cases outside the baseline: {extra}")
    if not problems:
        lost = sorted(case_id for case_id in base_cases
                      if base_cases[case_id].get("passed") and not cand_cases[case_id].get("passed"))
        if len(lost) > ALLOWED_LOSS:
            problems.append(f"{len(lost)} cases lost, at most {ALLOWED_LOSS} allowed: {lost}")
        return {"qualified": not problems, "problems": problems, "lost": lost,
                "baseline_passed": sum(1 for case in base_cases.values() if case.get("passed")),
                "baseline_cases": len(base_cases)}
    return {"qualified": False, "problems": problems, "lost": [], "baseline_passed": None, "baseline_cases": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, help="quality result JSON to qualify")
    parser.add_argument("--baseline", required=True, help="frozen baseline result JSON of the same set")
    args = parser.parse_args()
    candidate_path, baseline_path = Path(args.result), Path(args.baseline)
    for path in (candidate_path, baseline_path):
        if not path.is_file():
            raise SystemExit(f"{path} is missing")
    verdict = qualifies(load(candidate_path), load(baseline_path))
    if verdict["qualified"]:
        print(f"qualified: {candidate_path} loses {len(verdict['lost'])} of "
              f"{verdict['baseline_passed']}/{verdict['baseline_cases']} baseline passes (allowed {ALLOWED_LOSS})")
    else:
        print(f"not qualified: {candidate_path}", file=sys.stderr)
        for problem in verdict["problems"]:
            print(f"  {problem}", file=sys.stderr)
    return 0 if verdict["qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
