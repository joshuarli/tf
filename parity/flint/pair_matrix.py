"""Pairwise teacher-forced agreement on one history, including rec_mlx against itself.

rec_mlx groups a prefill's sums by chunk, so its own chunked prefill and one-token decode differ in BF16 rounding
(rec's tests/validate_prefix_rollback.py says a different grouping changes BF16 rounding). This prints the full
pairwise matrix, so TensorFold's distance from rec can be compared with rec's distance from rec on identical inputs.
A TensorFold residual no larger than rec's own is explained by grouping; a larger one is not.
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


def stats(a, b):
    x, y = f32(a), f32(b)
    d = np.abs(x - y)
    scale = max(float(np.abs(x).max()), float(np.abs(y).max()))
    ta, tb = x.argmax(1), y.argmax(1)
    sa, sb = np.sort(x, 1), np.sort(y, 1)
    ma, mb = sa[:, -1] - sa[:, -2], sb[:, -1] - sb[:, -2]
    margin = (ma >= 0.1) & (mb >= 0.1)
    return {
        "bits_equal": bool(np.array_equal(a, b)),
        "max": float(d.max()),
        "mean": float(d.mean()),
        "mean_rel": float((d / np.maximum(np.abs(x), 1e-3)).mean()),
        "p99": float(np.percentile(d, 99)),
        "max_over_scale": float(d.max() / scale),
        "top1": int((ta == tb).sum()),
        "rows": int(len(ta)),
        "margin_0.1_rows": int(margin.sum()),
        "top1_margin_0.1": int(((ta == tb) & margin).sum()),
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
        out.append(np.array(logits.astype(mx.bfloat16).view(mx.uint16)).reshape(-1, VOCAB))
    return np.concatenate(out)


def decode(engine, tokens):
    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["initial"] = 0
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    rows = [np.array(engine.forward(state, token)[0].astype(mx.bfloat16).view(mx.uint16)).reshape(VOCAB) for token in tokens]
    return np.stack(rows)


def main() -> int:
    ids = WORK / sys.argv[1]
    tokens = json.loads(ids.read_text())
    mx.set_cache_limit(2 * 1024**3)
    tf_out = WORK / "matrix_tf.bin"
    subprocess.run(["lockf", LOCK, str(ROOT / "zig-out/bin/tf-flint-run"), str(MODEL), str(ids), "logits", str(tf_out)], check=True)
    tf = np.fromfile(tf_out, dtype=np.uint16).reshape(-1, VOCAB)

    model = QATModel(MODEL)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    paths = {
        "tf-decode": tf,
        "rec-decode": decode(engine, tokens),
        "rec-chunk16": native(model, engine, tokens, 16),
        "rec-chunk64": native(model, engine, tokens, 64),
    }
    print(json.dumps({"rows": len(tokens)}))
    names = list(paths)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"{names[i]:12s} vs {names[j]:12s} " + json.dumps(stats(paths[names[i]], paths[names[j]])))
    for name, values in paths.items():
        np.save(WORK / f"matrix_{name}.npy", values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
