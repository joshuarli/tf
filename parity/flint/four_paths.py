"""Four teacher-forced paths, correctly shaped: each engine's own two arithmetic paths and the cross pairs.

tf-flint-run `logits` writes every row (decode row kernels); `prompt` writes only the last row (prompt tiles).
rec_mlx's resident decode writes every row; its chunked native prefill writes every row. `prompt` mode's single
row is compared only with the last rows of the others, so no array is silently broadcast.
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
    x, y = f32(np.asarray(a).reshape(-1)), f32(np.asarray(b).reshape(-1))
    d = np.abs(x - y)
    sa, sb = np.sort(x), np.sort(y)
    return {
        "bits_equal": bool(np.array_equal(np.asarray(a).reshape(-1), np.asarray(b).reshape(-1))),
        "max": round(float(d.max()), 4),
        "mean": round(float(d.mean()), 4),
        "p99": round(float(np.percentile(d, 99)), 4),
        "tf_top1": int(x.argmax()),
        "other_top1": int(y.argmax()),
        "agree": bool(x.argmax() == y.argmax()),
        "tf_margin": round(float(sa[-1] - sa[-2]), 4),
    }


def tf(mode: str, ids: Path, out: Path) -> np.ndarray:
    subprocess.run(["lockf", LOCK, str(ROOT / "zig-out/bin/tf-flint-run"), str(MODEL), str(ids), mode, str(out)], check=True)
    return np.fromfile(out, dtype=np.uint16).reshape(-1, VOCAB)


def main() -> int:
    ids = WORK / sys.argv[1]
    tokens = json.loads(ids.read_text())
    mx.set_cache_limit(2 * 1024**3)
    model = QATModel(MODEL)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())

    tf_decode = tf("logits", ids, WORK / "four_tf_decode.bin")
    tf_prompt = tf("prompt", ids, WORK / "four_tf_prompt.bin")

    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["initial"] = 0
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    rec_decode = np.stack(
        [np.array(engine.forward(state, token)[0].astype(mx.bfloat16).view(mx.uint16)).reshape(VOCAB) for token in tokens]
    )

    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    cache = model.make_cache(len(tokens), state["kv"])
    rows = []
    for begin in range(0, len(tokens), 4096):
        state["kv"].grow(min(begin + 4096, len(tokens)))
        inputs = mx.array([tokens[begin : begin + 4096]], dtype=mx.int32)
        hidden = model.native.model(inputs, cache)
        logits = model.native.lm_head(hidden)
        mx.eval(logits, [entry.state for entry in cache])
        rows.append(np.array(logits.astype(mx.bfloat16).view(mx.uint16)).reshape(-1, VOCAB))
    rec_chunked = np.concatenate(rows)

    print(json.dumps({"rows": len(tokens), "shapes": {"tf_decode": tf_decode.shape, "tf_prompt": tf_prompt.shape, "rec_decode": rec_decode.shape, "rec_chunked": rec_chunked.shape}}))
    pairs = [
        ("tf-decode", tf_decode, "rec-decode", rec_decode),
        ("tf-decode", tf_decode, "rec-chunked", rec_chunked),
        ("tf-prompt-last", tf_prompt, "rec-decode-last", rec_decode[-1:]),
        ("tf-prompt-last", tf_prompt, "rec-chunked-last", rec_chunked[-1:]),
        ("tf-decode-last", tf_decode[-1:], "tf-prompt-last", tf_prompt),
        ("rec-decode", rec_decode, "rec-chunked", rec_chunked),
    ]
    for name_a, a, name_b, b in pairs:
        print(f"{name_a:16s} vs {name_b:16s} " + json.dumps(stats(a, b)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
