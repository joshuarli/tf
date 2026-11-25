#!/bin/sh
# probe.sh NAME BINARY: one gated firstprobe.py run (first request, six 1K prompts, load time and footprint).
set -u
name=$1; bin=$2
out=/Users/josh/repos/tf/.work/tmp/p5c/first
mkdir -p "$out/invalid"
cd /Users/josh/repos/tf/.work/wt/p5c-chunks/campaign
export PATH="/opt/homebrew/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
for try in 1 2 3; do
  if ! GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1 >/dev/null 2>&1 || ! PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready >/dev/null 2>&1; then
    echo "$name try $try: pre-trial gate busy, waiting 60 s"; sleep 60; continue
  fi
  rm -f "$out/$name.jsonl"
  flag="$out/$name.compiler"; rm -f "$flag"
  ( while :; do pgrep -x zig >/dev/null 2>&1 && echo zig > "$flag"; pgrep -x clang >/dev/null 2>&1 && echo clang > "$flag"; sleep 1; done ) &
  watch=$!
  uv run --project . python /Users/josh/repos/tf/.work/tmp/p5c/firstprobe.py --binary "$bin" \
    --model-dir "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" --out "$out/$name.jsonl" 2>&1 | tail -1
  code=$?
  kill $watch 2>/dev/null; wait $watch 2>/dev/null
  log=$(ls -t /Users/josh/repos/tf/.work/tmp/firstprobe-*.log 2>/dev/null | head -1); [ -n "$log" ] && cp "$log" "$out/$name.server.log"
  post=0; PERF_GATE_MAX_BUSY_CORES=4 perf-gate-ready -1 >/dev/null 2>&1 || post=1
  if [ $post -eq 0 ] && [ ! -f "$flag" ]; then echo "$name: valid try $try"; exit 0; fi
  mv "$out/$name.jsonl" "$out/invalid/$name-try$try.jsonl" 2>/dev/null; mv "$out/$name.server.log" "$out/invalid/$name-try$try.server.log" 2>/dev/null
  echo "$name try $try invalid (post gate $post, compiler $(cat "$flag" 2>/dev/null))"
done
exit 1
