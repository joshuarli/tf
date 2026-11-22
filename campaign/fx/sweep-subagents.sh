#!/bin/sh
set -eu
export PATH=/opt/homebrew/bin:$PATH
export ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache
export TMPDIR="$PWD/.work/tmp"
export INKLING_LOAD_LIMIT_GB=26 INKLING_COPY_DRAFTS=1 TENSORFOLD_NO_UPDATE_CHECK=1
base="$PWD/.work/tmp/subagents"
mkdir -p "$base"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
python3 /Users/josh/repos/tensorfold-m5/campaign/heavy.py -- sh -c 'python3 -c "import sys; sys.path.insert(0, \"campaign\"); import heavy; print(heavy.wired_bytes())" > "$1"; shift; exec "$@"' sh "$base/wired" "$PWD/zig-out/native/bin/tensorfold-native" serve "$HOME/models/Inkling-2-Mini@ed40e7da625f" --host 127.0.0.1 --port "$port" --parallel 1 --prompt-cache-gib 2 --no-update-check --context 32768 --temperature 0 >"$base/server.log" 2>&1 &
server=$!
stop() { kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; }
trap stop EXIT HUP INT TERM
while :; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >"$base/models.json" 2>/dev/null; then break; fi
    if ! kill -0 "$server" 2>/dev/null; then wait "$server" || true; tail -40 "$base/server.log"; exit 1; fi
    sleep 2
done
perf-gate-ready
wired=$(cat "$base/wired")
model=$(jaq -r '.data[0].id' "$base/models.json")
python3 campaign/fx/bench_subagents.py --base "http://127.0.0.1:$port/v1" --model "$model" --log "$base/server.log" --server-pid "$server" --wired-start "$wired" >"$base/results.jsonl"
perf-gate-ready -1
perf-gate-ready
python3 tools/bench_concurrent.py "http://127.0.0.1:$port" "$model" --levels 1,2,4,8 --tokens 256 --reps 2 --temperatures 0 --alone --serial --output "$base/concurrent.json"
perf-gate-ready -1
