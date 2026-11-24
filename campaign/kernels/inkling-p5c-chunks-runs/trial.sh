#!/bin/sh
# trial.sh NAME BINARY [serve_curve options]: one gated served-curve trial; a tripped gate keeps the try as invalid.
set -u
name=$1; bin=$2; shift 2
out=/Users/josh/repos/tf/.work/tmp/p5c/curves
mkdir -p "$out/invalid"
cd /Users/josh/repos/tf/.work/wt/p5c-chunks/campaign
export PATH="/opt/homebrew/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
try=1
while [ $try -le 3 ]; do
  file="$out/$name.jsonl"
  rm -f "$file"
  if ! GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 >/dev/null 2>&1 || ! PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready >/dev/null 2>&1; then
    echo "$name try $try: pre-trial gate busy, waiting 60 s"; sleep 60; try=$((try + 1)); continue
  fi
  flag="$out/$name.compiler"
  rm -f "$flag"
  ( while :; do pgrep -x zig >/dev/null 2>&1 && echo zig > "$flag"; pgrep -x clang >/dev/null 2>&1 && echo clang > "$flag"; pgrep -x metal >/dev/null 2>&1 && echo metal > "$flag"; sleep 1; done ) &
  watch=$!
  uv run --project . python comparators/serve_curve.py tensorfold --model-dir "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" \
    --binary "$bin" --out "$file" "$@"
  code=$?
  kill $watch 2>/dev/null; wait $watch 2>/dev/null
  log=$(ls -t /Users/josh/repos/tf/.work/tmp/curve-tensorfold-*.log 2>/dev/null | head -1)
  [ -n "$log" ] && cp "$log" "$out/$name.server.log"
  post=0; PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready -1 >/dev/null 2>&1 || post=1
  if [ $code -eq 0 ] && [ $post -eq 0 ] && [ ! -f "$flag" ]; then echo "$name: valid try $try"; exit 0; fi
  reason="exit $code"; [ $post -ne 0 ] && reason="post-trial CPU gate"; [ -f "$flag" ] && reason="compiler observed ($(cat "$flag"))"
  mv "$file" "$out/invalid/$name-try$try.jsonl" 2>/dev/null
  mv "$out/$name.server.log" "$out/invalid/$name-try$try.server.log" 2>/dev/null
  echo "$name try $try invalid: $reason"
  try=$((try + 1))
done
exit 1
