#!/bin/sh
# SC2's re-paired round sessions: DF2b's and V3's builds in their receipts' balanced order, one session a call.
# A process the host gates defer (exit 75) waits two minutes and retries, for at most two hours.
# usage: pairs.sh SESSION   (SESSION 1 runs pairs a-c, 2 runs d-f); runners under $SC2_RUNNERS
set -eu
session=$1
runners=${SC2_RUNNERS:?set SC2_RUNNERS to the directory holding run-<commit>/bin/tf-inkling2-loop-run}
out=../inkling-sc2-scorecard-runs
kernels=$(cd "$(dirname "$0")/.." && pwd)

gated() {
  tries=0
  while :; do
    status=0
    python3 -I "$@" || status=$?
    [ "$status" -ne 75 ] && return "$status"
    tries=$((tries + 1))
    [ "$tries" -ge 60 ] && return 75
    sleep 120
  done
}

case $session in
  1) df2b="main-a:chain df2b-a:tree df2b-b:tree main-b:chain main-c:chain df2b-c:tree"
     v3="main-a lane-a lane-b main-b main-c lane-c" ;;
  2) df2b="df2b-d:tree main-d:chain main-e:chain df2b-e:tree df2b-f:tree main-f:chain"
     v3="lane-d main-d main-e lane-e lane-f main-f" ;;
  *) echo "session is 1 or 2" >&2; exit 2 ;;
esac

d=$kernels/inkling-df2b-tree-runs
for item in $df2b; do
  name=${item%%:*} drafts=${item#*:}
  case $name in main-*) bin=$runners/run-50b4a681a ;; *) bin=$runners/run-d95cc3fe0 ;; esac
  gated "$d/run.py" bench "$bin/bin/tf-inkling2-loop-run" "$d/prompts.json" "$out/df2b/$name" --drafts "$drafts" --tokens 512
done
d=$kernels/inkling-v3-experts-runs
for name in $v3; do
  case $name in main-*) bin=$runners/run-53a3b8894 ;; *) bin=$runners/run-e3b30971c ;; esac
  gated "$d/run.py" bench "$bin/bin/tf-inkling2-loop-run" "$d/prompts.json" "$out/v3/$name" --drafts tree --tokens 512
done
echo "session $session done"
