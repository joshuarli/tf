#!/bin/sh
set -eu
export PATH="/opt/homebrew/bin:$HOME/.elan/bin:$PATH"
export TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp
export ZIG_GLOBAL_CACHE_DIR=/Users/josh/repos/tensorfold-m5/.work/zig-global-cache
export XDG_CACHE_HOME=/Users/josh/repos/tensorfold-m5/.work/tmp/nano-xdg-cache
export CLANG_MODULE_CACHE_PATH=/Users/josh/repos/tensorfold-m5/.work/tmp/nano-clang-cache
export PYTHONDONTWRITEBYTECODE=1 ELAN_SKIP_UPDATE_CHECK=1
root=/Users/josh/repos/tensorfold-m5/.work/tmp/nanoattention-lane
source="$root/nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199"
archive=/Users/josh/repos/tensorfold-m5/.work/src/nanoattention@6f782a8dace8/nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199.tar.gz
mkdir -p "$root"
if [ ! -d "$source" ]; then tar xzf "$archive" -C "$root"; fi
(cd "$source" && zig build -Doptimize=ReleaseFast)
zig build tf-inkling2-loop-run -Dcpu=apple_m1 --prefix "$root/tf-build" --cache-dir "$root/tf-cache"
shasum -a 256 "$source/zig-out/bin/nano" "$root/tf-build/bin/tf-inkling2-loop-run"
