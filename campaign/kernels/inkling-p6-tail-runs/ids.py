"""ids.py cut SRC N OUT | chunks N: the first N token ids of SRC written to OUT, or N in 1,024-row chunks and a rest."""
import json
import sys
from pathlib import Path

FULL = 1024


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "cut":
        ids = json.loads(Path(sys.argv[2]).read_text())
        n = int(sys.argv[3])
        if not 0 < n <= len(ids):
            raise SystemExit(f"{sys.argv[2]} holds {len(ids)} ids, not {n}")
        Path(sys.argv[4]).write_text(json.dumps(ids[:n]))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "chunks":
        n = int(sys.argv[2])
        print(",".join([str(FULL)] * (n // FULL) + ([str(n % FULL)] if n % FULL else [])))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
