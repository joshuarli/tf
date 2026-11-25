#!/bin/sh
# tail.sh BUILD: wall and GPU ms of candidate tail schedules at 1K dev lengths, held-out code tokens, one load each,
# schedules interleaved within every rep. Timed: run it inside the lane's GPU timing window only.
set -u
P=/Users/josh/repos/tf/.work/tmp/p6
T=$1/bin/tf-inkling2-prefix-check
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
H=/Users/josh/repos/tf/campaign/heavy.py
here=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p6-tail-runs
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
mkdir -p "$P/tail"
cd "$P/tail" || exit 1
for n in 1027 1033 1019; do python3 -I -B "$here/ids.py" cut "$P/ids-heldout-code.json" "$n" "ids-$n.json"; done
python3 "$H" -- "$T" "$M" time ids-1027.json 6 1024,2,1 1024,3 1026,1 1027 514,513 > tail-1027.txt 2>&1
python3 "$H" -- "$T" "$M" time ids-1033.json 6 1024,8,1 1024,9 1032,1 1033 517,516 > tail-1033.txt 2>&1
python3 "$H" -- "$T" "$M" time ids-1019.json 6 1018,1 1019 > tail-1019.txt 2>&1
python3 -I -B "$here/tailsum.py" tail-1027.txt tail-1033.txt tail-1019.txt | tee tails.md
