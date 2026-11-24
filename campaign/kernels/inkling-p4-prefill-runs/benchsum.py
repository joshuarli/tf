"""benchsum.py LOG...: pass-1 (cold) and warm prefill wall times of bench.sh logs"""
import re, sys
from pathlib import Path
for path in sys.argv[1:]:
    text = Path(path).read_text()
    cold = re.search(r"inkling2_loop prefill: (\d+) tokens in ([\d.]+) ms, ([\d.]+) tok/s", text)
    warm = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r"prefill repeat \d+: \d+ tokens in ([\d.]+) ms, ([\d.]+) tok/s", text)]
    if not cold:
        print(f"{Path(path).stem}: no result")
        continue
    w = ", ".join(f"{ms:.0f} ms {tps:.0f} tok/s" for ms, tps in warm)
    print(f"{Path(path).stem}: {cold.group(1)} tokens; cold {float(cold.group(2)):.0f} ms {float(cold.group(3)):.0f} tok/s; warm {w}")
