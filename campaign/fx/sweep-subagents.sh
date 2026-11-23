#!/bin/sh
set -eu
export PATH=/opt/homebrew/bin:$PATH
export ZIG_GLOBAL_CACHE_DIR="$(git rev-parse --git-common-dir)/../.work/zig-global-cache"
export TMPDIR="$PWD/.work/tmp"
export INKLING_LOAD_LIMIT_GB=26 INKLING_COPY_DRAFTS=1 TENSORFOLD_NO_UPDATE_CHECK=1
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
export INKLING_AGENTS="${1:-1}"
context="${2:-4096}"
base="$PWD/.work/tmp/sessions-$INKLING_AGENTS-$context"
mkdir -p "$base"
rm -f "$base/server.pid"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
python3 campaign/heavy.py -- sh -c 'python3 -c "import sys; sys.path.insert(0, \"campaign\"); import heavy; print(heavy.wired_bytes())" > "$1"; printf "%s\n" "$$" > "$2"; shift 2; exec "$@"' sh "$base/wired" "$base/server.pid" "$PWD/zig-out/native/bin/tensorfold-native" serve "$HOME/models/Inkling-2-Mini@ed40e7da625f" --host 127.0.0.1 --port "$port" --parallel "$INKLING_AGENTS" --prompt-cache-gib 2 --no-update-check --context "$context" --temperature 0 >"$base/server.log" 2>&1 &
server=$!
stop() {
    if [ -f "$base/server.pid" ]; then
        model_pid=$(cat "$base/server.pid")
        lock_pid=$(ps -o ppid= -p "$model_pid" | tr -d ' ' || true)
        if [ -n "$lock_pid" ]; then
            parent_pid=$(ps -o ppid= -p "$lock_pid" | tr -d ' ' || true)
            if [ "$parent_pid" = "$server" ]; then kill -TERM "$model_pid" 2>/dev/null || true; fi
        fi
    else
        kill -TERM "$server" 2>/dev/null || true
    fi
    wait "$server" || true
}
trap stop EXIT HUP INT TERM
while :; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >"$base/models.json" 2>/dev/null; then break; fi
    if ! kill -0 "$server" 2>/dev/null; then wait "$server" || true; tail -40 "$base/server.log"; exit 1; fi
    sleep 2
done
perf-gate-ready
wired=$(cat "$base/wired")
model=$(jaq -r '.data[0].id' "$base/models.json")
python3 -c 'import os, sys; p = "campaign/fx/bench_subagents.py"; s = open(p).read(); needle = "for n in (1, 2, 4, 8):"; assert s.count(needle) == 1; s = s.replace(needle, "for n in (int(os.environ[\"INKLING_AGENTS\"]),):"); sys.argv = [p] + sys.argv[1:]; exec(compile(s, p, "exec"), {"__name__": "__main__", "__file__": os.path.abspath(p), "os": os})' --base "http://127.0.0.1:$port/v1" --model "$model" --log "$base/server.log" --server-pid "$server" --wired-start "$wired" >"$base/results.jsonl"
perf-gate-ready -1
python3 -c 'import sys; sys.path.insert(0, "campaign/fx"); import bench_subagents; events = bench_subagents.parse_log(open(sys.argv[1]).read()); expected = int(sys.argv[2]) * 8; assert len(events["prefix"]) == expected and len(events["done"]) == expected, events; assert all(row["reused"] + row["prefilled"] + 256 <= int(sys.argv[3]) for row in events["prefix"]), events' "$base/server.log" "$INKLING_AGENTS" "$context"
jaq -e -s 'length == 2 and all(.[]; ((.cached_tokens | length) == (.n * 4)) and ([.cached_tokens | to_entries[] | select(.key % 4 != 0) | (.value != null and .value > 0)] | all))' "$base/results.jsonl"

if [ "$INKLING_AGENTS" = 4 ]; then
    perf-gate-ready
    python3 tools/bench_concurrent.py "http://127.0.0.1:$port" "$model" --levels 1,2,4 --tokens 256 --reps 2 --temperatures 0 --alone --serial --output "$base/concurrent.json"
    perf-gate-ready -1
    jaq -e '[.cells[] | .alone.unequal == 0 and .alone.failed == 0 and .serial.unequal == 0 and .serial.failed == 0 and (.streams == 1 or (.alone | has("vacuous") | not))] | all' "$base/concurrent.json" >/dev/null \
        || { echo "sweep-subagents: a concurrent reply differs from its solo reply, failed, or never overlapped another stream; see $base/concurrent.json" >&2; exit 1; }
fi
