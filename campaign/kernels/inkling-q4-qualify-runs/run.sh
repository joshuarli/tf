#!/bin/sh
# Q4's measurements on one pinned binary, in order. STEP is one of:
#   smoke      completed tasks, dev set, plain and speculative, half prefill
#   tasks      completed tasks, full set, plain and speculative, half prefill then INKLING_PREFILL_HALF=off
#   served N   fixed-window sessions N (a, b, ...): qualify then dev prompts, plain / disabled / speculative,
#              half prefill, mode order rotated per session
#   off        one fixed-window session pair (qualify, dev) with INKLING_PREFILL_HALF=off
# usage: run.sh STEP [N]   with Q4_BIN the served binary
set -eu
bin=${Q4_BIN:?set Q4_BIN to the pinned tensorfold-native}
here=$(cd "$(dirname "$0")" && pwd)
kernels=$(dirname "$here")
tasks=$here/tasks.py
served=$kernels/inkling-sv2-served-runs/run.py
curve=1024,4096,8192,16384,31744
out=../inkling-q4-qualify-runs

session() {
  python3 -I "$served" session --bin "$bin" --name "$out/served-$1-qualify" --set qualify --rotate "$2" --curve "$curve"
  python3 -I "$served" session --bin "$bin" --name "$out/served-$1-dev" --set dev --rotate "$(( ($2 + 1) % 3 ))" \
    --curve "$curve"
}

case $1 in
  smoke) python3 -I "$tasks" run --bin "$bin" --name tasks-smoke --set dev --modes ar,spec ;;
  tasks) python3 -I "$tasks" run --bin "$bin" --name tasks-half --set full --modes ar,spec
         python3 -I "$tasks" run --bin "$bin" --name tasks-off --set full --modes ar,spec --prefill off ;;
  served) case ${2:?session letter} in a) r=0 ;; b) r=1 ;; c) r=2 ;; *) r=0 ;; esac
          session "$2" "$r" ;;
  off) (export INKLING_PREFILL_HALF=off; session off 0) ;;
  *) echo "step is smoke, tasks, served N or off" >&2; exit 2 ;;
esac
