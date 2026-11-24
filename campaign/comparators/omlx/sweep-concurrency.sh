#!/bin/sh
set -eu
root=$(git rev-parse --show-toplevel)
main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
work="$root/.work"
mtp="${OMLX_MTP:-off}"
case "$mtp" in on) mtp_enabled=True ;; off) mtp_enabled=False ;; *) echo "OMLX_MTP must be on or off" >&2; exit 2 ;; esac
base="$work/tmp/omlx-concurrency-guarded-four-mtp-$mtp"
mkdir -p "$base/models" "$base/omlx" "$work/tmp"
[ -L "$base/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" ] || ln -s "$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" "$base/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
export PATH="$main/.work/venv-omlx/bin:$PATH"
export UV_CACHE_DIR="$main/.work/uv-cache" UV_PYTHON_INSTALL_DIR="$main/.work/uv-python"
export XDG_CACHE_HOME="$main/.work/xdg-cache" HF_HOME="$main/.work/hf-home" TMPDIR="$work/tmp"
export PYTHONDONTWRITEBYTECODE=1 OMLX_WITH_CUSTOM_KERNEL=0 HF_HUB_OFFLINE=1
export INKLING_LOAD_LIMIT_GB=26 GPU_GATE_MAX_GPU_PCT=60 GPU_GATE_MAX_UTIL_PCT=60
python3 - "$base/omlx/model_settings.json" "$mtp_enabled" <<'PYSETTINGS'
import json
import sys
with open(sys.argv[1], 'w') as f:
    json.dump({'version': 3, 'models': {'Inkling-2-Mini-oQ4e-mtp@3eea042751cd': {'mtp_enabled': sys.argv[2] == 'True'}}}, f)
PYSETTINGS
port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
model='Inkling-2-Mini-oQ4e-mtp@3eea042751cd'
memory_pressure > "$base/memory-start.txt"
sysctl vm.swapusage >> "$base/memory-start.txt"
python3 campaign/heavy.py -- sh -c 'python3 -c "import sys; sys.path.insert(0, \"campaign\"); import heavy; print(heavy.wired_bytes())" > "$1"; shift; until gpu-gate-ready -1; do sleep 5; done; exec "$@"' sh "$base/wired" omlx serve --model-dir "$base/models" --base-path "$base/omlx" --host 127.0.0.1 --port "$port" --no-hf-cache --memory-guard-gb 26 --max-concurrent-requests 4 > "$base/server.log" 2>&1 &
server=$!
stop() { kill -TERM "$server" 2>/dev/null || true; wait "$server" || true; }
trap stop EXIT HUP INT TERM
while :; do
    if curl -fsS "http://127.0.0.1:$port/v1/models" > "$base/models.json" 2>/dev/null; then break; fi
    if ! kill -0 "$server" 2>/dev/null; then wait "$server" || true; tail -40 "$base/server.log"; exit 1; fi
    sleep 2
done
wired=$(cat "$base/wired")
until perf-gate-ready; do sleep 10; done
python3 campaign/comparators/omlx/bench_subagents.py --base "http://127.0.0.1:$port/v1" --model "$model" --log "$base/server.log" --server-pid "$server" --wired-start "$wired" > "$base/results.jsonl"
perf-gate-ready -1
until perf-gate-ready; do sleep 10; done
python3 tools/bench_concurrent.py "http://127.0.0.1:$port" "$model" --levels 1,2,4,8 --tokens 256 --reps 2 --temperatures 0 --alone --serial --output "$base/concurrent.json" > "$base/concurrent.jsonl"
perf-gate-ready -1
memory_pressure > "$base/memory-end.txt"
sysctl vm.swapusage >> "$base/memory-end.txt"
