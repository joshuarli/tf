#!/bin/sh
# The timed block: first-request traces and six interleaved main/lane served-curve pairs over two sessions.
set -u
P=/Users/josh/repos/tf/.work/tmp/p5c
T=$P/trial.sh
MAIN=$P/cur/main-build/native/bin/tensorfold-native
LANE=$P/cur/lane-build/native/bin/tensorfold-native
TMAIN=$P/cur/trace-main-build/native/bin/tensorfold-native
TLANE=$P/cur/trace-lane-build/native/bin/tensorfold-native
TFAULT=$P/cur/fault-build/native/bin/tensorfold-native
run() { echo "$(date +%H:%M:%S) $1"; "$T" "$@" 2>&1 | tail -3; }
noreuse() { echo "$(date +%H:%M:%S) $1 (INKLING_PREFIX_REUSE=0)"; env INKLING_PREFIX_REUSE=0 "$T" "$@" 2>&1 | tail -3; }
run first-main-a "$TMAIN" --lengths 1024
run first-lane-a "$TLANE" --lengths 1024
run first-fault-a "$TFAULT" --lengths 1024
run s1-main-a "$MAIN"
run s1-lane-a "$LANE"
run s1-lane-b "$LANE"
run s1-main-b "$MAIN"
run s1-main-c "$MAIN"
run s1-lane-c "$LANE"
noreuse s1-lane-noreuse "$LANE"
run first-fault-b "$TFAULT" --lengths 1024
run first-lane-b "$TLANE" --lengths 1024
run first-main-b "$TMAIN" --lengths 1024
run s2-lane-d "$LANE"
run s2-main-d "$MAIN"
run s2-main-e "$MAIN"
run s2-lane-e "$LANE"
run s2-lane-f "$LANE"
run s2-main-f "$MAIN"
noreuse s2-lane-noreuse "$LANE"
echo "$(date +%H:%M:%S) session done"
