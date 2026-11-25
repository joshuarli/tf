#!/bin/sh
# session.sh 1|2: one timed session of three interleaved main/lane served-curve pairs (1: ABBAAB, 2: BAABBA).
set -u
here=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p6-tail-runs
P=/Users/josh/repos/tf/.work/tmp/p6
MAIN=$P/main-build/native/bin/tensorfold-native
LANE=$P/lane-build/native/bin/tensorfold-native
run() { echo "$(date +%H:%M:%S) $1"; "$here/trial.sh" "$@" 2>&1 | tail -3; }
case "$1" in
  1) run s1-main-a "$MAIN"; run s1-lane-a "$LANE"; run s1-lane-b "$LANE"; run s1-main-b "$MAIN"; run s1-main-c "$MAIN"; run s1-lane-c "$LANE" ;;
  2) run s2-lane-d "$LANE"; run s2-main-d "$MAIN"; run s2-main-e "$MAIN"; run s2-lane-e "$LANE"; run s2-lane-f "$LANE"; run s2-main-f "$MAIN" ;;
  *) echo "usage: session.sh 1|2" >&2; exit 2 ;;
esac
echo "$(date +%H:%M:%S) session $1 done"
