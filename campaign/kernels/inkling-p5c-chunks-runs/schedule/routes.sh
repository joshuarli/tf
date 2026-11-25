#!/bin/sh
set -eu
M=$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd
R=/Users/josh/repos/tf/.work/tmp/p5c/main-build/bin/tf-inkling2-loop-run
cd /Users/josh/repos/tf/.work/tmp/p5c
mkdir -p routes
INKLING_LOAD_LIMIT_GB=26 "$R" "$M" ids-1024.json prefillroutes 1024 routes/c1024 1023
INKLING_LOAD_LIMIT_GB=26 "$R" "$M" ids-1024.json prefillroutes 896 routes/c896 1023
ls -la routes
