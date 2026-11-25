"""d20jobs.py OUT_DIR: tf-inkling2-prefix-check `jobs` lines for Decision 20's captures.

(a) every oMLX oracle history under main's served schedule, the lane's, and uniform 1,152-row chunks (the lane's
largest chunk); (b) three held-out code probes (1,024, 1,033 and 1,100 tokens) under main reuse on, main reuse off and
the lane. Token files come from P5c's oracle capture directory and the held-out code fixture.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schedules import LANE_MOST, lane_schedule, main_schedule, text  # noqa: E402

ORACLE = Path("/Users/josh/repos/tf/.work/tmp/p5c/oracle")
HELDOUT = Path("/Users/josh/repos/tf/.work/tmp/p5c/ids-heldout-code.json")
PROBES = (1024, 1033, 1100)


def uniform(n: int, size: int) -> list[int]:
    return [min(size, n - at) for at in range(0, n, size)]


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    lines = []
    for path in sorted(ORACLE.glob("*.tokens.json")):
        name = path.name.removesuffix(".tokens.json")
        n = len(json.loads(path.read_text()))
        for group, chunks in (("main-served", main_schedule(n, True)), ("lane-served", lane_schedule(n)),
                              ("uniform-1152", uniform(n, LANE_MOST))):
            (out / "oracle" / group).mkdir(parents=True, exist_ok=True)
            lines.append(f"{path} {text(chunks)} {out}/oracle/{group}/{name}.bin")
    ids = json.loads(HELDOUT.read_text())
    (out / "probe").mkdir(parents=True, exist_ok=True)
    for n in PROBES:
        source = out / "probe" / f"ids-{n}.json"
        source.write_text(json.dumps(ids[:n]))
        for side, chunks in (("main-on", main_schedule(n, True)), ("main-off", main_schedule(n, False)),
                             ("lane", lane_schedule(n))):
            lines.append(f"{source} {text(chunks)} {out}/probe/{side}-{n}.bin")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
