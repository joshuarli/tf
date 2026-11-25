"""Decision 19's gate for an activation-precision variant against the exact path, as one reproducible verdict.

usage: decision19.py --exact-quality R --variant-quality R --exact-oracle O --variant-oracle O
                     --exact-pi P --variant-pi P [--json OUT]

R is a full-set `quality/run.py` result.json; O an `omlx_gate.py compare` report (its "top-1 agreement: N/M" and
"mean KL, mean of history means: X" lines) or a literal N/M:X; P a Pi result.json. Rules (roadmap Decision 19):
  - oracle mean KL at most 1.10 times the exact path's, and top-1 no more than 0.5 points below it;
  - the full quality set loses at most one case against the exact path, every lost case listed;
  - no more runaways (replies reaching the 8,192-token cap) than the exact path, counted net (owner, 2026-11-24);
  - Pi coding tasks no worse than exact.
Both runs must cover the same cases: a case missing on one side, excluded or unsupported on one side only, fails
the gate. Cases unsupported (the engine refuses the request) or over-context on both sides are listed and not
scored. Host-condition markers are reported, not blocking: greedy outputs do not depend on host load.
"""

import argparse
import json
import re
import sys
from pathlib import Path

KL_RATIO, TOP1_POINTS, ALLOWED_LOSS = 1.10, 0.5, 1
CAP = 8192


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def oracle(value: str) -> dict:
    if m := re.fullmatch(r"(\d+)/(\d+):([0-9.eE+-]+)", value):
        return {"top1": int(m[1]), "rows": int(m[2]), "kl": float(m[3])}
    text = Path(value).read_text(encoding="utf-8")
    top = re.search(r"top-1 agreement: (\d+)/(\d+)", text)
    kl = re.search(r"mean KL, mean of history means: ([0-9.eE+-]+)", text)
    if not top or not kl:
        raise SystemExit(f"{value}: no top-1 agreement or mean KL line")
    return {"top1": int(top[1]), "rows": int(top[2]), "kl": float(kl[1])}


def outcome(case: dict) -> str:
    reason = case.get("reason") or ""
    if reason.startswith("request failed") and "HTTP Error 400" in reason:
        return "unsupported"
    if reason.startswith("request failed") or reason.startswith("scoring failed"):
        return "error"
    if reason.startswith(f"truncated at {CAP}") or case.get("finish_reason") == "length":
        return "runaway"
    return "pass" if case.get("passed") is True else "fail"


def cases(result: dict) -> dict[str, str]:
    out = {c["id"]: outcome(c) for c in result.get("cases") or []}
    for e in result.get("excluded") or []:
        out[e["id"]] = "excluded"
    return out


def pi_passed(result: dict) -> int:
    """Pi tasks whose checker passed, counted from the tasks; a task without a checker verdict does not count."""
    return sum(1 for t in result.get("tasks") or [] if (t.get("checker") or {}).get("pass") is True)


def verdict(exact_q: dict, variant_q: dict, exact_o: dict, variant_o: dict, exact_pi: dict, variant_pi: dict) -> dict:
    problems, notes = [], []
    if exact_q.get("set") != "full" or variant_q.get("set") != "full":
        problems.append("both quality runs must be the full set")
    e, v = cases(exact_q), cases(variant_q)
    for case in sorted(set(e) ^ set(v)):
        problems.append(f"{case}: recorded on one side only")
    shared = sorted(set(e) & set(v))
    unscored = {"unsupported", "excluded"}
    for case in shared:
        if (e[case] in unscored or v[case] in unscored) and e[case] != v[case]:
            problems.append(f"{case}: {e[case]} on exact, {v[case]} on the variant")
        if "error" in (e[case], v[case]):
            problems.append(f"{case}: a request or scoring error (exact {e[case]}, variant {v[case]})")
    scored = [c for c in shared if e[c] not in unscored and v[c] not in unscored]
    lost = [c for c in scored if e[c] == "pass" and v[c] != "pass"]
    gained = [c for c in scored if e[c] != "pass" and v[c] == "pass"]
    runaways = {"exact": [c for c in scored if e[c] == "runaway"], "variant": [c for c in scored if v[c] == "runaway"]}
    if len(lost) > ALLOWED_LOSS:
        problems.append(f"{len(lost)} cases lost against exact (at most {ALLOWED_LOSS})")
    if len(runaways["variant"]) > len(runaways["exact"]):
        problems.append(f"{len(runaways['variant'])} runaways against exact's {len(runaways['exact'])}")
    new_runaways = sorted(set(runaways["variant"]) - set(runaways["exact"]))
    if new_runaways:
        notes.append(f"new runaways on cases exact answers or fails short: {', '.join(new_runaways)} "
                     "(allowed by the net count; case-by-case preservation is not shown)")
    kl_ratio = variant_o["kl"] / exact_o["kl"]
    points = 100 * (variant_o["top1"] / variant_o["rows"] - exact_o["top1"] / exact_o["rows"])
    if variant_o["rows"] != exact_o["rows"]:
        problems.append(f"oracle rows differ ({exact_o['rows']} against {variant_o['rows']})")
    if kl_ratio > KL_RATIO:
        problems.append(f"oracle mean KL {kl_ratio:.3f} times exact (at most {KL_RATIO})")
    if points < -TOP1_POINTS:
        problems.append(f"oracle top-1 {points:+.2f} points (at least -{TOP1_POINTS})")
    pi = {"exact": pi_passed(exact_pi), "variant": pi_passed(variant_pi),
          "tasks": [len(exact_pi.get("tasks") or []), len(variant_pi.get("tasks") or [])]}
    if pi["tasks"][0] != pi["tasks"][1]:
        problems.append(f"Pi task counts differ {pi['tasks']}")
    if pi["variant"] < pi["exact"]:
        problems.append(f"Pi {pi['variant']} passed against exact's {pi['exact']}")
    markers = {side: r.get("invalid_reasons") or [] for side, r in (("exact", exact_q), ("variant", variant_q))}
    for side, reasons in markers.items():
        if reasons:
            notes.append(f"{side} quality run carries host-condition markers (reported, not blocking): {reasons}")
    return {"pass": not problems, "problems": problems, "notes": notes,
            "quality": {"exact_passed": sum(e[c] == "pass" for c in scored),
                        "variant_passed": sum(v[c] == "pass" for c in scored), "scored": len(scored),
                        "lost": lost, "gained": gained, "runaways": runaways,
                        "unsupported": sorted(c for c in shared if e[c] == v[c] == "unsupported"),
                        "excluded": sorted(c for c in shared if e[c] == v[c] == "excluded")},
            "oracle": {"kl_ratio": kl_ratio, "top1_points": points, "exact": exact_o, "variant": variant_o},
            "pi": pi}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for side in ("exact", "variant"):
        parser.add_argument(f"--{side}-quality", type=Path, required=True)
        parser.add_argument(f"--{side}-oracle", required=True)
        parser.add_argument(f"--{side}-pi", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = verdict(load(args.exact_quality), load(args.variant_quality), oracle(args.exact_oracle),
                     oracle(args.variant_oracle), load(args.exact_pi), load(args.variant_pi))
    q = result["quality"]
    print(f"quality: exact {q['exact_passed']}/{q['scored']}, variant {q['variant_passed']}/{q['scored']}; "
          f"lost {q['lost'] or 'none'}; gained {q['gained'] or 'none'}; runaways exact {q['runaways']['exact']} "
          f"variant {q['runaways']['variant']}; unsupported {len(q['unsupported'])}; excluded {len(q['excluded'])}")
    o = result["oracle"]
    print(f"oracle: mean KL {o['kl_ratio']:.3f} times exact; top-1 {o['top1_points']:+.2f} points")
    print(f"pi: exact {result['pi']['exact']}, variant {result['pi']['variant']} of {result['pi']['tasks'][0]}")
    for line in result["notes"]:
        print(f"note: {line}")
    for line in result["problems"]:
        print(f"FAIL: {line}")
    print("Decision 19:", "pass" if result["pass"] else "fail")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
