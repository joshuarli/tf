#!/bin/sh
# quality.sh NAME SRC BUILD: quality dev (Decision 20 (c)) on BUILD's server, run from that build's source tree SRC so
# the run writes its results in scratch; the run directory is copied into this item's runs.
set -eu
here=/Users/josh/repos/tf/.work/wt/p6-tail/campaign/kernels/inkling-p6-tail-runs
export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp INKLING_LOAD_LIMIT_GB=26
mkdir -p "$3/native/lib" "$here/quality"
cp /Users/josh/repos/tf/zig-out/native/lib/libtfgrammar.dylib "$3/native/lib/"
cd "$2/campaign"
# run.py exits non-zero below the profile's bar; the result is still the record, so keep it and report the status.
status=0
python3 heavy.py -- env PYTHONDONTWRITEBYTECODE=1 uv run --project . python quality/run.py --model inkling2-mini-oq4e \
  --set dev --bin "$3/native/bin/tensorfold-native" || status=$?
run=
for dir in quality/results/inkling2-mini-oq4e-dev-*; do run=$dir; done
[ -d "$run" ] || { echo "no quality run directory" >&2; exit 1; }
cp -R "$run" "$here/quality/$1-$(basename "$run")"
echo "copied $run to $here/quality/$1-$(basename "$run") (quality run exit $status)"
