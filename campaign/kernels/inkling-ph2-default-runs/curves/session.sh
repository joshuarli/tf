#!/bin/sh
set -u
T=/Users/josh/repos/tf/.work/tmp/ph2c/trial.sh
B=/Users/josh/repos/tf/.work/tmp/ph2c/build/native/bin/tensorfold-native
half() { echo "$(date +%H:%M:%S) $1 (default half)"; env -u INKLING_PREFILL_HALF "$T" "$1" "$B" 2>&1 | tail -3; }
off() { echo "$(date +%H:%M:%S) $1 (INKLING_PREFILL_HALF=off)"; env INKLING_PREFILL_HALF=off "$T" "$1" "$B" 2>&1 | tail -3; }
half half-a
off off-a
off off-b
half half-b
half half-c
off off-c
echo "$(date +%H:%M:%S) session done"
