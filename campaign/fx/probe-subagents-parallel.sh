#!/bin/sh
set -eu
export PATH=/opt/homebrew/bin:$PATH ZIG_GLOBAL_CACHE_DIR=$PWD/.work/zig-global-cache
export TMPDIR="$PWD/.work/tmp" INKLING_LOAD_LIMIT_GB=26 INKLING_COPY_DRAFTS=1 TENSORFOLD_NO_UPDATE_CHECK=1
base="$PWD/.work/tmp/subagents"
for parallel in 2 4; do
    GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 gpu-gate-ready -1
    perf-gate-ready
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    python3 campaign/heavy.py -- "$PWD/zig-out/native/bin/tensorfold-native" serve "$HOME/models/Inkling-2-Mini@ed40e7da625f" --host 127.0.0.1 --port "$port" --parallel "$parallel" --prompt-cache-gib 2 --no-update-check --context 32768 --temperature 0 >"$base/parallel-$parallel.log" 2>&1 &
    server=$!
    stop() { kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; }
    trap stop EXIT HUP INT TERM
    while :; do
        if curl -fsS "http://127.0.0.1:$port/v1/models" >"$base/parallel-$parallel-models.json" 2>/dev/null; then break; fi
        if ! kill -0 "$server" 2>/dev/null; then wait "$server" || true; tail -30 "$base/parallel-$parallel.log"; exit 1; fi
        sleep 2
    done
    perf-gate-ready
    model=$(jaq -r '.data[0].id' "$base/parallel-$parallel-models.json")
    python3 tools/bench_concurrent.py "http://127.0.0.1:$port" "$model" --levels "$parallel" --tokens 32 --reps 1 --temperatures 0 --output "$base/parallel-$parallel.json"
    perf-gate-ready -1
    stop
    trap - EXIT HUP INT TERM
done
