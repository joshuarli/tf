"""Draft acceptance per sweep cell from a TensorFold served sweep's directory.

usage: mtp_acceptance.py SESSION_DIR [SESSION_DIR...]

Each SESSION_DIR is a `sweep-subagents.sh` run directory holding `results.jsonl` and `server.log`. A cell is one
repetition at one agent count; its requests are the `request_ids` the client recorded, and each request's drafted and
accepted counts come from the server's `done` line (`rounds=R accepted=A/D`). Prints one JSON object per cell.
"""

import json
import re
import sys
from pathlib import Path

DONE = re.compile(r'done (\S+) .*?tokens=(\d+).*? rounds=(\d+) accepted=(\d+)/(\d+)')


def cells(session: Path):
    done = {m[0]: (int(m[1]), int(m[2]), int(m[3]), int(m[4])) for m in DONE.findall((session / 'server.log').read_text(errors='replace'))}
    for line in (session / 'results.jsonl').read_text().splitlines():
        row = json.loads(line)
        found = [done[i] for i in row['request_ids'] if i in done]
        tokens = sum(f[0] for f in found)
        rounds = sum(f[1] for f in found)
        accepted = sum(f[2] for f in found)
        drafted = sum(f[3] for f in found)
        yield {'session': session.name, 'n': row['n'], 'repeat': row['repeat'], 'requests': len(row['request_ids']),
               'matched': len(found), 'tokens': tokens, 'rounds': rounds, 'drafted': drafted, 'accepted': accepted,
               'acceptance': accepted / drafted if drafted else None,
               'tokens_per_round': tokens / rounds if rounds else None,
               'aggregate_tokens_per_second': row['aggregate_tokens_per_second']}


def main() -> int:
    for arg in sys.argv[1:]:
        for cell in cells(Path(arg)):
            print(json.dumps(cell))
    return 0


if __name__ == '__main__':
    sys.exit(main())
