"""curvesum.py RUN.jsonl...: one row a run: first request tok/s and each target's median client tok/s (1K split dev/holdout)"""
import json, statistics, sys
from pathlib import Path
targets = [1024, 4096, 8192, 16384, 31744]
print("| Run | first | 1K dev | 1K holdout | 1K all | 4K | 8K | 16K | 31.7K | valid |")
print("| --- |" + " ---: |" * 8 + " --- |")
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    cond = [r for r in rows if r.get("kind") == "conditions"]
    first = [r for r in rows if r.get("kind") == "first"]
    curve = [r for r in rows if r.get("kind") == "curve"]
    def med(xs):
        return f"{statistics.median(xs):.0f}" if xs else "-"
    cells = [f"{first[0]['client_prefill_tps']:.0f}" if first else "-"]
    k1 = [r for r in curve if r["target"] == 1024]
    dev = [r["client_prefill_tps"] for r in k1 if "heldout" not in r["source"]]
    ho = [r["client_prefill_tps"] for r in k1 if "heldout" in r["source"]]
    cells += [med(dev), med(ho), med(dev + ho)]
    for t in targets[1:]:
        cells.append(med([r["client_prefill_tps"] for r in curve if r["target"] == t]))
    valid = cond[-1].get("valid") if cond else None
    print(f"| {Path(path).stem} | " + " | ".join(cells) + f" | {valid} |")
