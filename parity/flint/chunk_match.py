"""Which rec_mlx chunking do TensorFold's decode rows match?

rec_mlx's native prefill sums each chunk at once, so a chunk boundary changes BF16 grouping (rec's own
validate_prefix_rollback.py says so). This runs the native path at several chunk sizes and the resident decode path
(one token at a time) on one history, and prints each one's difference from tf-flint-run's `logits` rows:
the pairing with the smallest residual is the arithmetic TensorFold's kernels actually implement.
"""

import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

ROOT = Path("/Users/josh/repos/tensorfold-m5/.work/wt/f2-parity")
WORK = Path("/Users/josh/repos/tensorfold-m5/.work/parity/flint")
MODEL = Path.home() / "models" / "flint-int4@3b93f9053b53"
LOCK = "/Users/josh/repos/tensorfold-m5/.work/gpu.lock"
VOCAB = 150144


def f32(a):
    return (a.astype(np.uint32) << 16).view(np.float32)


def compare(a, b):
    x, y = f32(a), f32(b)
    d = np.abs(x - y)
    ta, tb = x.argmax(1), y.argmax(1)
    return {
        "bits_equal": bool(np.array_equal(a, b)),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "top1": int((ta == tb).sum()),
        "scale": float(np.abs(x).max()),
    }


def native(model, engine, tokens, chunk):
    state = engine.prefill_state(min(len(tokens), chunk), limit=len(tokens) + 1)
    cache = model.make_cache(len(tokens), state["kv"])
    out = []
    for begin in range(0, len(tokens), chunk):
        state["kv"].grow(min(begin + chunk, len(tokens)))
        inputs = mx.array([tokens[begin : begin + chunk]], dtype=mx.int32)
        hidden = model.native.model(inputs, cache)
        logits = model.native.lm_head(hidden)
        mx.eval(logits, [entry.state for entry in cache])
        out.append(logits.astype(mx.bfloat16).view(mx.uint16))
    return np.concatenate([np.array(o).reshape(-1, VOCAB) for o in out])


def decode(engine, tokens):
    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["initial"] = 0
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    rows = []
    for token in tokens:
        logits, _trace = engine.forward(state, token)
        rows.append(logits.astype(mx.bfloat16).view(mx.uint16))
    mx.eval(rows)
    return np.stack([np.array(r).reshape(VOCAB) for r in rows])


def main() -> int:
    ids = WORK / sys.argv[1]
    tokens = json.loads(ids.read_text())
    mx.set_cache_limit(2 * 1024**3)
    tf_out = WORK / "chunk_tf.bin"
    subprocess.run(["lockf", LOCK, str(ROOT / "zig-out/bin/tf-flint-run"), str(MODEL), str(ids), "logits", str(tf_out)], check=True)
    tf = np.fromfile(tf_out, dtype=np.uint16).reshape(-1, VOCAB)

    model = QATModel(MODEL)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    print(json.dumps({"rows": len(tokens), "tf_scale": float(np.abs(f32(tf)).max())}))
    for chunk in (len(tokens), 16, 1):
        got = native(model, engine, tokens, chunk)
        print("tf vs rec-native chunk", chunk, json.dumps(compare(tf, got)))
    got = decode(engine, tokens)
    print("tf vs rec-decode        ", json.dumps(compare(tf, got)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
