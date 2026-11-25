#!/bin/sh
# builds.sh MAIN_REV LANE_REV: frozen builds of the lane's base and of LANE_REV, each a `git archive` built into its own prefix.
# Compiles: never run it inside anyone's GPU timing window.
set -eu
P=/Users/josh/repos/tf/.work/tmp/p6
export PATH="/opt/homebrew/bin:$PATH" TMPDIR=/Users/josh/repos/tf/.work/tmp ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tf/.work/zig-global-cache
for side in main lane; do
  rev=$1; [ "$side" = lane ] && rev=$2
  rm -rf "$P/$side-src" "$P/$side-build"
  mkdir -p "$P/$side-src"
  git -C /Users/josh/repos/tf archive "$rev" | tar -x -C "$P/$side-src"
  (cd "$P/$side-src" && zig build native tf-inkling2-prefix-check -Dcpu=apple_m1 -Dversion=1.0.0 --prefix "$P/$side-build") > "$P/$side-build.log" 2>&1
  echo "$side $rev $(git -C /Users/josh/repos/tf rev-parse "$rev") $(sha256sum "$P/$side-build/native/bin/tensorfold-native")"
done | tee "$P/binaries.txt"
