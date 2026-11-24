#!/bin/sh
set -eu
cd /Users/josh/repos/tf/.work/wt/p5c-chunks/campaign
REF=/Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref
D=/Users/josh/repos/tf/.work/tmp/p5c/oracle
for set in main-served lane-served uniform-1024; do
  ~/.local/bin/uv run --frozen --project ref python -I -B parity/inkling2-mini/omlx_gate.py compare --ref "$REF" --cand "$D/$set" \
    --json "$D/$set.json" --report "$D/$set.md" --title "oMLX oracle against the $set captures" > "$D/$set.log" 2>&1
  tail -4 "$D/$set.log"
done
~/.local/bin/uv run --frozen --project ref python -I -B parity/inkling2-mini/omlx_gate.py compare --ref "$D/main-served" --cand "$D/lane-served" \
  --json "$D/lane-vs-main-served.json" --report "$D/lane-vs-main-served.md" --title "main's served schedule against the lane's" > "$D/lane-vs-main-served.log" 2>&1
tail -4 "$D/lane-vs-main-served.log"
