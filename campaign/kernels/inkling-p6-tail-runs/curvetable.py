"""curvetable.py DIR: the receipt's served-curve tables from session.sh's interleaved main/lane trials in DIR.

Client tok/s per request; a cell's figure is the median over every request of that cell in a side's trials, beside the
range of the trials' own medians. Ratios pair requests by kind, target and rep within a pair of trials, which fixes the
corpus span; a pair's figure is the geometric mean over its curve requests, and the interval is a t interval over pairs.
"""
import json
import math
import statistics
import sys
from pathlib import Path

PAIRS = [("s1-main-a", "s1-lane-a"), ("s1-main-b", "s1-lane-b"), ("s1-main-c", "s1-lane-c"),
         ("s2-main-d", "s2-lane-d"), ("s2-main-e", "s2-lane-e"), ("s2-main-f", "s2-lane-f")]
CELLS = ["1K dev", "1K held-out", "4K", "8K", "16K", "31.7K"]
LONG = {4096: "4K", 8192: "8K", 16384: "16K", 31744: "31.7K"}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def cell(record: dict) -> str:
    if record["kind"] == "first":
        return "first"
    if record["target"] == 1024:
        return "1K held-out" if "heldout" in record["source"] else "1K dev"
    return LONG[record["target"]]


def load(path: Path) -> tuple[dict, dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    requests = {(r["kind"], r["target"], r["rep"]): r for r in rows if r.get("kind") in ("first", "curve")}
    conditions = [r for r in rows if r.get("kind") == "conditions"]
    return requests, (conditions[-1] if conditions else {})


def gm(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> int:
    root = Path(sys.argv[1])
    trials = {name: load(root / f"{name}.jsonl") for pair in PAIRS for name in pair if (root / f"{name}.jsonl").is_file()}
    pairs = [(m, l) for m, l in PAIRS if m in trials and l in trials]
    sides = {"main": [m for m, _ in pairs], "lane": [l for _, l in pairs]}
    values: dict[tuple[str, str], list[float]] = {}
    medians: dict[tuple[str, str], list[float]] = {}
    for side, names in sides.items():
        for name in names:
            per: dict[str, list[float]] = {}
            for record in trials[name][0].values():
                per.setdefault(cell(record), []).append(record["client_prefill_tps"])
            for key, tps in per.items():
                values.setdefault((side, key), []).extend(tps)
                medians.setdefault((side, key), []).append(statistics.median(tps))
    ratios: dict[str, list[float]] = {}
    pair_gm = []
    for m, l in pairs:
        curve = []
        for key, record in trials[m][0].items():
            other = trials[l][0].get(key)
            if other is None or other["source"].split("[")[0] != record["source"].split("[")[0]:
                continue
            ratio = other["client_prefill_tps"] / record["client_prefill_tps"]
            ratios.setdefault(cell(record), []).append(ratio)
            if record["kind"] == "curve":
                curve.append(ratio)
        pair_gm.append(gm(curve))
    print(f"| Prompt | main tok/s, median (trial medians) | lane tok/s, median (trial medians) | lane / main per request, "
          f"median (range) | lane at 2,000 |")
    print("| --- | ---: | ---: | ---: | --- |")
    for key in CELLS:
        mv, lv = values.get(("main", key), []), values.get(("lane", key), [])
        mm, lm = medians.get(("main", key), []), medians.get(("lane", key), [])
        r = ratios.get(key, [])
        if not mv or not lv:
            continue
        print(f"| {key} | {statistics.median(mv):,.0f} ({min(mm):,.0f}-{max(mm):,.0f}) | "
              f"{statistics.median(lv):,.0f} ({min(lm):,.0f}-{max(lm):,.0f}) | "
              f"{statistics.median(r):.3f} ({min(r):.3f}-{max(r):.3f}) | "
              f"{'met' if statistics.median(lv) >= 2000 else 'missed'}, {sum(1 for x in lm if x >= 2000)} of {len(lm)} trials |")
    logs = [math.log(g) for g in pair_gm]
    n = len(logs)
    if n > 1:
        mean, sd = statistics.mean(logs), statistics.stdev(logs)
        half = T95[n - 1] * sd / math.sqrt(n)
        print(f"\nPairs' geometric means over curve requests: {', '.join(f'{g:.3f}' for g in pair_gm)}; "
              f"{math.exp(mean):.3f} across {n} pairs, 95% interval {math.exp(mean - half):.3f}-{math.exp(mean + half):.3f}")
    print("\n| Trial | first request, client s | client tok/s | spawn to ready, s | footprint ready / peak, GB | valid |")
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for side, names in sides.items():
        for name in names:
            requests, conditions = trials[name]
            first = requests.get(("first", 1024, 0), {})
            memory = conditions.get("memory") or {}
            ready, peak = memory.get("ready_bytes"), memory.get("peak_bytes")
            footprint = f"{ready / 1e9:.2f} / {peak / 1e9:.2f}" if ready and peak else "-"
            print(f"| {name} | {first.get('wall_s', float('nan')):.3f} | {first.get('client_prefill_tps', float('nan')):,.0f} | "
                  f"{conditions.get('spawn_to_ready_s', float('nan')):.1f} | {footprint} | {conditions.get('valid')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
