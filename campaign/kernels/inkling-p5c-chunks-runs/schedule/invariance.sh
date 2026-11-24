#!/bin/sh
set -eu
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
R=/Users/josh/repos/tf/.work/tmp/p5c/main-build/bin/tf-inkling2-loop-run
cd /Users/josh/repos/tf/.work/tmp/p5c
INKLING_LOAD_LIMIT_GB=26 "$R" "$M" ids-1024.json prefill 1024 logits-c1024.bin
INKLING_LOAD_LIMIT_GB=26 "$R" "$M" ids-1024.json prefill 896 logits-c896.bin
