"""tailsum.py LOG...: per schedule, the median GPU and wall ms of a prompt's prefill and of each chunk over warm reps.

Each LOG is `tf-inkling2-prefix-check ... time` output. Rep 0 is the process's first pass over the weights and is left
out; the rest are interleaved across the schedules, so they share the GPU's clock state.
"""
import re
import statistics
import sys
from pathlib import Path

CHUNKS = re.compile(r"^time schedule=(\d+) chunks=(\S+)$")
CHUNK = re.compile(r"^time schedule=(\d+) rep=(\d+) at=(\d+) rows=(\d+) wall_ms=([\d.]+) gpu_ms=([\d.]+)$")
TOTAL = re.compile(r"^time schedule=(\d+) rep=(\d+) tokens=(\d+) chunks=(\d+) wall_ms=([\d.]+) gpu_ms=([\d.]+) greedy=(\d+)$")


def summarize(path: Path) -> None:
    names: dict[int, str] = {}
    totals: dict[int, list[tuple[float, float, int]]] = {}
    chunks: dict[tuple[int, int], list[tuple[float, float]]] = {}
    tokens = 0
    for line in path.read_text(errors="replace").splitlines():
        if m := CHUNKS.match(line):
            names[int(m.group(1))] = m.group(2)
        elif (m := CHUNK.match(line)) and int(m.group(2)) > 0:
            chunks.setdefault((int(m.group(1)), int(m.group(3))), []).append((float(m.group(5)), float(m.group(6))))
        elif (m := TOTAL.match(line)) and int(m.group(2)) > 0:
            tokens = int(m.group(3))
            totals.setdefault(int(m.group(1)), []).append((float(m.group(5)), float(m.group(6)), int(m.group(7))))
    print(f"## {path.name}: {tokens} tokens")
    print("| schedule | GPU ms | wall ms | tok/s by wall | greedy | each chunk GPU ms |")
    print("| --- | ---: | ---: | ---: | --- | --- |")
    for s, rows in sorted(totals.items()):
        gpu = statistics.median(r[1] for r in rows)
        wall = statistics.median(r[0] for r in rows)
        greedy = sorted({r[2] for r in rows})
        per = ", ".join(f"{statistics.median(v[1] for v in values):.1f}"
                        for (t, _), values in sorted(chunks.items()) if t == s)
        print(f"| {names.get(s, s)} | {gpu:.1f} | {wall:.1f} | {tokens / wall * 1e3:,.0f} | {greedy} | {per} |")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        summarize(Path(arg))
