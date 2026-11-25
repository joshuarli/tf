#!/bin/sh
# d20.sh BUILD: Decision 20 (a) and (b). Every oracle history and probe under main's and the lane's served schedules,
# captured in one load of BUILD's tf-inkling2-prefix-check (prefill.zig is the same on both sides), then compared.
set -eu
P=/Users/josh/repos/tf/.work/tmp/p6
D=$P/d20
here=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p6-tail-runs
spread=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p5c-chunks-runs/spread.py
REF=/Users/josh/repos/tf/.work/parity/inkling2-mini/omlx-ref
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
H="python3 /Users/josh/repos/tf/campaign/heavy.py --"
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
python3 -I -B "$here/d20jobs.py" "$D" > "$D-jobs.txt"
$H "$1/bin/tf-inkling2-prefix-check" "$M" jobs "$D-jobs.txt" > "$D-capture.log" 2>&1
tail -2 "$D-capture.log"
cd /Users/josh/repos/tf/.work/wt/p6-tail/campaign
for group in main-served lane-served uniform-1152; do
  $H uv run --frozen --project ref python -I -B parity/inkling2-mini/omlx_gate.py compare --ref "$REF" --cand "$D/oracle/$group" \
    --json "$D/oracle/$group.json" --report "$D/oracle/$group.md" --title "oMLX oracle against the $group captures" > "$D/oracle/$group.log" 2>&1
  tail -4 "$D/oracle/$group.log"
done
$H uv run --frozen --project ref python -I -B parity/inkling2-mini/omlx_gate.py compare --ref "$D/oracle/main-served" --cand "$D/oracle/lane-served" \
  --json "$D/oracle/lane-vs-main-served.json" --report "$D/oracle/lane-vs-main-served.md" --title "main's served schedule against the lane's" > "$D/oracle/lane-vs-main-served.log" 2>&1
tail -4 "$D/oracle/lane-vs-main-served.log"
for n in 1024 1033 1100; do
  echo "## $n held-out code tokens"
  echo "lane against main reuse-on: $($H uv run --frozen --project ref python -I -B "$spread" "$D/probe/main-on-$n.bin" "$D/probe/lane-$n.bin")"
  echo "main reuse-off against main reuse-on: $($H uv run --frozen --project ref python -I -B "$spread" "$D/probe/main-on-$n.bin" "$D/probe/main-off-$n.bin")"
  echo "lane against main reuse-off: $($H uv run --frozen --project ref python -I -B "$spread" "$D/probe/main-off-$n.bin" "$D/probe/lane-$n.bin")"
done > "$D/probe-spread.txt"
cat "$D/probe-spread.txt"
