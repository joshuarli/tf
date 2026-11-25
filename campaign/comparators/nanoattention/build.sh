#!/bin/sh
set -eu
main=$(cd "$(git -C "$(dirname "$0")" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)
export PATH="/opt/homebrew/bin:$HOME/.elan/bin:$PATH"
export TMPDIR=$main/.work/tmp
export ZIG_GLOBAL_CACHE_DIR=$main/.work/zig-global-cache
export XDG_CACHE_HOME=$main/.work/tmp/nano-xdg-cache
export CLANG_MODULE_CACHE_PATH=$main/.work/tmp/nano-clang-cache
export PYTHONDONTWRITEBYTECODE=1 ELAN_SKIP_UPDATE_CHECK=1
root=$main/.work/tmp/nanoattention-lane
source="$root/nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199"
archive=$main/.work/src/nanoattention@6f782a8dace8/nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199.tar.gz
mkdir -p "$root"
if [ ! -d "$source" ]; then tar xzf "$archive" -C "$root"; fi
(cd "$source" && zig build -Doptimize=ReleaseFast)
zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix "$root/tf-build" --cache-dir "$root/tf-cache"
sha256sum "$source/zig-out/bin/nano" "$root/tf-build/bin/tf-inkling2-loop-run"
