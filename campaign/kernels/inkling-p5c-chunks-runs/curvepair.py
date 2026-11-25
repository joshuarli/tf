"""curvepair.py MAIN.jsonl LANE.jsonl [MAIN.jsonl LANE.jsonl ...]: per-request lane/main client tok/s of interleaved pairs.

Requests pair by kind, target and rep, which fix the corpus span in every trial; each cell is the median ratio of a
length's requests in that pair, the first request reported alone. A pair's rows are split dev and held-out by source.
"""
import json
import statistics
import sys
from pathlib import Path

TARGETS = [1024, 4096, 8192, 16384, 31744]


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    reqs = {(r["kind"], r["target"], r["rep"]): r for r in rows if r.get("kind") in ("first", "curve")}
    cond = [r for r in rows if r.get("kind") == "conditions"]
    return reqs, (cond[-1].get("valid") if cond else None)


def cell(values):
    return f"{statistics.median(values):.2f}" if values else "-"


paths = sys.argv[1:]
print("| Pair | first | 1K dev | 1K held-out | 4K | 8K | 16K | 31.7K | 4K+ dev | 4K+ held-out | valid |")
print("| --- |" + " ---: |" * 9 + " --- |")
for i in range(0, len(paths) - 1, 2):
    main, mv = load(paths[i])
    lane, lv = load(paths[i + 1])
    ratios = {}
    for key, m in main.items():
        l = lane.get(key)
        if l is None or l["source"].split("[")[0] != m["source"].split("[")[0]:
            continue
        ratios[key] = (l["client_prefill_tps"] / m["client_prefill_tps"], "heldout" in m["source"])
    first = [r for (k, _, _), (r, _) in ratios.items() if k == "first"]
    row = [cell(first)]
    k1 = [(r, h) for (k, t, _), (r, h) in ratios.items() if k == "curve" and t == 1024]
    row += [cell([r for r, h in k1 if not h]), cell([r for r, h in k1 if h])]
    for t in TARGETS[1:]:
        row.append(cell([r for (k, tt, _), (r, _) in ratios.items() if k == "curve" and tt == t]))
    long = [(r, h) for (k, t, _), (r, h) in ratios.items() if k == "curve" and t > 1024]
    row += [cell([r for r, h in long if not h]), cell([r for r, h in long if h])]
    print(f"| {Path(paths[i]).stem} / {Path(paths[i + 1]).stem} | " + " | ".join(row) + f" | {mv}, {lv} |")
