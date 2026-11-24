#!/bin/sh
set -eu
export PATH=/opt/homebrew/bin:$PATH
export TMPDIR="$PWD/.work/tmp"
export GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60 PERF_GATE_MAX_BUSY_CORES="${PERF_GATE_MAX_BUSY_CORES:-4}"
export INKLING_AGENTS="${1:-1}"
context="${2:-4096}"
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
server_bin="${LLAMA_SERVER:-$main/.work/src/llama.cpp-upstream@af5c58347fdb/tree/build/bin/llama-server}"
gguf="${LLAMA_GGUF:-$HOME/models/Inkling-2-Mini-NVFP4-XL-GGUF@109af3770438/Inkling-2-Mini-NVFP4-XL.gguf}"
base="$PWD/.work/tmp/llama-sessions-$(basename "$gguf" .gguf)${LLAMA_TAG:+-$LLAMA_TAG}-$INKLING_AGENTS-$context"
mkdir -p "$base"
rm -f "$base/server.pid"
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
python3 campaign/heavy.py -- sh -c 'python3 -c "import sys; sys.path.insert(0, \"campaign\"); import heavy; print(heavy.wired_bytes())" > "$1"; printf "%s\n" "$$" > "$2"; shift 2; exec "$@"' sh "$base/wired" "$base/server.pid" \
    "$server_bin" -m "$gguf" --host 127.0.0.1 --port "$port" -ngl 999 --load-mode none -np "$INKLING_AGENTS" \
    -c "$((context * INKLING_AGENTS))" --cache-ram "${LLAMA_CACHE_MIB:-2048}" --jinja --temp 0 ${LLAMA_EXTRA_ARGS:-} >"$base/server.log" 2>&1 &
server=$!
stop() {
    if [ -f "$base/server.pid" ]; then kill -TERM "$(cat "$base/server.pid")" 2>/dev/null || true; else kill -TERM "$server" 2>/dev/null || true; fi
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
python3 -c 'import os, sys; p = "campaign/comparators/omlx/bench_subagents.py"; s = open(p).read(); needle = "for n in (1, 2, 4, 8):"; assert s.count(needle) == 1; s = s.replace(needle, "for n in (int(os.environ[\"INKLING_AGENTS\"]),):"); sys.argv = [p] + sys.argv[1:]; exec(compile(s, p, "exec"), {"__name__": "__main__", "__file__": os.path.abspath(p), "os": os})' --base "http://127.0.0.1:$port/v1" --model "$model" --log "$base/server.log" --server-pid "$server" --wired-start "$wired" >"$base/results.jsonl"
perf-gate-ready -1
jaq -e -s 'length == 2 and all(.[]; (.cached_tokens | length) == (.n * 4))' "$base/results.jsonl" >/dev/null

if [ "$INKLING_AGENTS" = 4 ]; then
    perf-gate-ready
    python3 tools/bench_concurrent.py "http://127.0.0.1:$port" "$model" --levels 1,2,4 --tokens 256 --reps 2 --temperatures 0 --alone --serial --output "$base/concurrent.json"
    perf-gate-ready -1
fi
