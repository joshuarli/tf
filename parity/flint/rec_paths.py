"""Compare rec_mlx's own two teacher-forced paths on the same tokens: chunked native prefill vs one-token resident decode.

Each row r is the distribution of token r + 1 given tokens 0..r. Prints per-row top-1, the max absolute difference and
the mean absolute difference, so a difference between the paths is visible separately from any difference against
TensorFold.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel


def bits(logits: mx.array) -> mx.array:
    return logits.astype(mx.bfloat16).view(mx.uint16)


def rows_native(model, engine, tokens, chunk):
    state = engine.prefill_state(min(len(tokens), chunk), limit=len(tokens) + 1)
    cache = model.make_cache(len(tokens), state["kv"])
    out = []
    for begin in range(0, len(tokens), chunk):
        state["kv"].grow(min(begin + chunk, len(tokens)))
        inputs = mx.array([tokens[begin : begin + chunk]], dtype=mx.int32)
        hidden = model.native.model(inputs, cache)
        mx.eval(model.native.lm_head(hidden), [entry.state for entry in cache])
        out.append(model.native.lm_head(hidden).astype(mx.bfloat16))
    return mx.concatenate(out, axis=1)


def rows_decode(model, engine, tokens):
    session = engine.make_session(max_new_tokens=len(tokens))
    out = []
    first = tokens[0]
    logits = session.prefill([first])
    out.append(logits.astype(mx.bfloat16))
    for token in tokens[1:]:
        out.append(session.forward_token(token).astype(mx.bfloat16))
    return mx.stack(out, axis=1) if out[0].ndim == 2 else mx.stack([r.reshape(1, 1, -1) for r in out], axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=4096)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    mx.set_cache_limit(2 * 1024**3)
    tokens = json.loads(args.ids.read_text())
    model = QATModel(args.model)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())

    started = time.perf_counter()
    native = rows_native(model, engine, tokens, args.chunk)
    mx.eval(native)
    middle = time.perf_counter()
    decode = rows_decode(model, engine, tokens)
    mx.eval(decode)
    finished = time.perf_counter()

    a = bits(native).reshape(len(tokens), -1)
    b = bits(decode).reshape(len(tokens), -1)
    fa = (a.astype(mx.uint32) << 16).view(mx.float32)
    fb = (b.astype(mx.uint32) << 16).view(mx.float32)
    diff = mx.abs(fa - fb)
    mx.eval(diff)
    same = mx.array_equal(a, b).item()
    print(
        json.dumps(
            {
                "rows": len(tokens),
                "bit_identical": same,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "native_seconds": middle - started,
                "decode_seconds": finished - middle,
            }
        )
    )
    ta = mx.argmax(fa, axis=1).tolist()
    tb = mx.argmax(fb, axis=1).tolist()
    print("top1 agree", sum(x == y for x, y in zip(ta, tb)), "/", len(tokens))
    for row in range(min(6, len(tokens))):
        worst = int(diff[row].argmax().item())
        print(
            row,
            "top1",
            ta[row],
            tb[row],
            "worst vocab",
            worst,
            float(fa[row, worst].item()),
            float(fb[row, worst].item()),
        )
    out = args.ids.with_suffix(".decode.npy")
    mx.save(str(out), b)
    print("decode logits bits at", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
