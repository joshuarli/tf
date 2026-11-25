#!/bin/sh
set -u
P=/Users/josh/repos/tf/.work/tmp/p5c
C=$P/cur
for spec in "probe-main-a $C/trace-main-build" "probe-lane-a $C/trace-lane-build" "probe-warm-a $C/warm-build" "probe-warm-b $C/warm-build" "probe-lane-b $C/trace-lane-build" "probe-main-b $C/trace-main-build" "probe-fault-a $C/fault-build"; do
  set -- $spec
  echo "$(date +%H:%M:%S) $1"
  "$P/probe.sh" "$1" "$2/native/bin/tensorfold-native" 2>&1 | tail -2
done
echo "$(date +%H:%M:%S) probes done"
