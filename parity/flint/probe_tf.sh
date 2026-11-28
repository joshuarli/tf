#!/bin/sh
set -x
export TMPDIR=/Users/josh/repos/tensorfold-m5/.work/tmp
cd /Users/josh/repos/tensorfold-m5/.work/wt/f2-parity
lockf /Users/josh/repos/tensorfold-m5/.work/gpu.lock ./zig-out/bin/tf-flint-run ~/models/flint-int4@3b93f9053b53 /Users/josh/repos/tensorfold-m5/.work/parity/flint/probe_ids.json logits /Users/josh/repos/tensorfold-m5/.work/parity/flint/probe_tf_logits.bin
