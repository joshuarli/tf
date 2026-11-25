#!/bin/sh
# profile.sh BUILD: GPU time of the served 1,024-row chunks to the cap (every chunk of a 31,744-token prompt), then
# split Metal 4 timelines of one chunk at bases 0, 7,168, 15,360 and 30,720 (8 warm reps, 2 stamped).
# Timed: run it inside the lane's GPU timing window only.
set -u
P=/Users/josh/repos/tf/.work/tmp/p6
T=$1/bin/tf-inkling2-prefix-check
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
H=/Users/josh/repos/tf/campaign/heavy.py
here=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p6-tail-runs
SRC=/Users/josh/repos/tf/.work/tmp/p4/fixtures/code-32768-all.json
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
mkdir -p "$P/profile"
cd "$P/profile" || exit 1
for n in 1024 8192 16384 31744; do python3 -I -B "$here/ids.py" cut "$SRC" "$n" "ids-$n.json"; done
chunks=$(python3 -I -B "$here/ids.py" chunks 31744)
python3 "$H" -- "$T" "$M" time ids-31744.json 2 "$chunks" > chunks-31744.txt 2>&1
for base in 0 7168 15360 30720; do
  n=$((base + 1024))
  chunks=$(python3 -I -B "$here/ids.py" chunks "$n")
  env TF_METAL4_TIMELINE=2 TF_INKLING_PREFILL_PROFILE_BASE=$base python3 "$H" -- "$T" "$M" profile "ids-$n.json" 10 "$chunks" > "timeline-$base.log" 2>&1
  grep -c "metal 4 timeline: inkling2-slot" "timeline-$base.log"
done
python3 -I -B "$here/buckets.py" timeline-0.log timeline-7168.log \
  timeline-15360.log timeline-30720.log > buckets.md
head -9 buckets.md
python3 -I -B "$here/chunkcurve.py" chunks-31744.txt > chunkcurve.md
tail -2 chunkcurve.md
