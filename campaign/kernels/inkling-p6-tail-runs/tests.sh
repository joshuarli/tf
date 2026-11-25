#!/bin/sh
# tests.sh: the lane's acceptance tests from its worktree. Host tests, the served host's GPU tests (store, fork,
# prefix hits, repeated prompts, speculation) and the family's capture, rollback and fork GPU tests, through heavy.py.
# Compiles: never run it inside anyone's GPU timing window.
set -u
L=/Users/josh/repos/tf/.work/tmp/p6/tests
mkdir -p "$L"
cd /Users/josh/repos/tf/.work/wt/p6-tail || exit 1
export PATH="/opt/homebrew/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tf/.work/zig-global-cache
export INKLING_LOAD_LIMIT_GB=26 TF_INKLING2_LOOP_DIR="$HOME/models/Inkling-2-Mini@ed40e7da625f"
export TF_INKLING2_LOOP_OQ4E_DIR="$HOME/models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd" TF_INKLING2_DFLASH2_DIR="$HOME/models/Inkling-2-Mini-DFlash2@4f9088ea66c2"
zig build test-inkling2-loop-host --summary all > "$L/host.log" 2>&1; echo "test-inkling2-loop-host exit $?"
python3 campaign/heavy.py -- zig build test-inkling2-loop-served -Dcpu=apple_m1 -j1 --maxrss 28000000000 --summary all > "$L/served.log" 2>&1
echo "test-inkling2-loop-served exit $?"
for filter in capture rollback fork; do
  python3 campaign/heavy.py -- zig build test-inkling2-loop-filtered "-Dinkling-filter=$filter" -Dcpu=apple_m1 -j1 \
    --maxrss 28000000000 --summary all > "$L/family-$filter.log" 2>&1
  echo "test-inkling2-loop-filtered -Dinkling-filter=$filter exit $?"
done
