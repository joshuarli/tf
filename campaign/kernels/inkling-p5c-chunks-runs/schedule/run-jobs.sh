#!/bin/sh
set -eu
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
T=/Users/josh/repos/tf/.work/tmp/p5c/lane-build/bin/tf-inkling2-prefix-check
cd "$1"
mkdir -p main-served lane-served uniform-1024 probe
while read -r ids schedule out; do
  INKLING_LOAD_LIMIT_GB=26 "$T" "$M" "$ids" "$schedule" "$out"
done < "$2"
