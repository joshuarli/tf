"""chunkcurve.py LOG: GPU ms of each 1,024-row chunk of one prompt against its base, from a `time` log.

Medians over the warm reps (rep 0 left out). The fit is least squares over the chunks past the first local window
(base 1,024 and up), where every row's local attention already spans a full window.
"""
import re
import statistics
import sys
from pathlib import Path

CHUNK = re.compile(r"^time schedule=0 rep=(\d+) at=(\d+) rows=(\d+) wall_ms=([\d.]+) gpu_ms=([\d.]+)$")


def main() -> int:
    gpu: dict[int, list[float]] = {}
    rows: dict[int, int] = {}
    for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
        m = CHUNK.match(line)
        if m and int(m.group(1)) > 0:
            gpu.setdefault(int(m.group(2)), []).append(float(m.group(5)))
            rows[int(m.group(2))] = int(m.group(3))
    points = sorted((base, statistics.median(v)) for base, v in gpu.items())
    print("| base | rows | GPU ms | ms a row |")
    print("| ---: | ---: | ---: | ---: |")
    for base, ms in points:
        print(f"| {base:,} | {rows[base]} | {ms:.1f} | {ms / rows[base]:.3f} |")
    fit = [(b, ms) for b, ms in points if b >= 1024 and rows[b] == 1024]
    if len(fit) > 1:
        mx = statistics.mean(b for b, _ in fit)
        my = statistics.mean(ms for _, ms in fit)
        slope = sum((b - mx) * (ms - my) for b, ms in fit) / sum((b - mx) ** 2 for b, _ in fit)
        print(f"\nfit over {len(fit)} full chunks from base 1,024: {my - slope * mx:.1f} ms + {slope * 1024:.2f} ms per 1,024 "
              f"positions of base; total {sum(ms for _, ms in points):,.0f} ms for {sum(rows.values()):,} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
