"""curvetable.py: the receipt's served-curve table from the six pairs and the reuse-off references."""
import json
import statistics
import math
from pathlib import Path

D = Path("curves")
mains = ["s1-main-a", "s1-main-b", "s1-main-c", "s2-main-d", "s2-main-e", "s2-main-f"]
lanes = ["s1-lane-a", "s1-lane-b", "s1-lane-c", "s2-lane-d", "s2-lane-e", "s2-lane-f"]
offs = ["s1-lane-noreuse", "s2-lane-noreuse"]


def load(name):
    rows = [json.loads(l) for l in (D / f"{name}.jsonl").read_text().splitlines() if l.strip()]
    return {(r["kind"], r["target"], r["rep"]): r for r in rows if r.get("kind") in ("first", "curve")}


def cells(reqs):
    out = {}
    for (kind, target, rep), r in reqs.items():
        if kind == "first":
            out.setdefault("first", []).append(r["client_prefill_tps"])
            continue
        held = "heldout" in r["source"]
        key = ("1K held-out" if held else "1K dev") if target == 1024 else {4096: "4K", 8192: "8K", 16384: "16K", 31744: "31.7K"}[target]
        out.setdefault(key, []).append(r["client_prefill_tps"])
    return {k: statistics.median(v) for k, v in out.items()}


order = ["first", "1K dev", "1K held-out", "4K", "8K", "16K", "31.7K"]
M = [cells(load(n)) for n in mains]
L = [cells(load(n)) for n in lanes]
O = [cells(load(n)) for n in offs]
ratios = {k: [] for k in order}
for m, l in zip(map(load, mains), map(load, lanes)):
    for key, mr in m.items():
        lr = l.get(key)
        if not lr:
            continue
        kind, target, _ = key
        held = "heldout" in mr["source"]
        cell = "first" if kind == "first" else (("1K held-out" if held else "1K dev") if target == 1024 else {4096: "4K", 8192: "8K", 16384: "16K", 31744: "31.7K"}[target])
        ratios[cell].append(lr["client_prefill_tps"] / mr["client_prefill_tps"])
print("| Prompt | main tok/s | lane tok/s | lane reuse off | lane/main per request (median, range) | 2,000 met |")
print("| --- | ---: | ---: | ---: | ---: | --- |")
for k in order:
    mv = [x[k] for x in M if k in x]
    lv = [x[k] for x in L if k in x]
    ov = [x[k] for x in O if k in x]
    r = ratios[k]
    gm = math.exp(sum(math.log(x) for x in r) / len(r))
    met = "no" if max(lv) < 2000 else f"in {sum(1 for x in lv if x >= 2000)} of {len(lv)} trials"
    print(f"| {k} | {min(mv):,.0f}-{max(mv):,.0f} | {min(lv):,.0f}-{max(lv):,.0f} | {min(ov):,.0f}-{max(ov):,.0f} | "
          f"{statistics.median(r):.2f} ({min(r):.2f}-{max(r):.2f}), geomean {gm:.2f} | {met} |")
