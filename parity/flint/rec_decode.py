"""Teacher-forced logits from rec_mlx's resident decode kernel, one token at a time from position 0.

The decode kernel keeps every layer's keys and values in KVStorage (`state["kv"]`), including the sliding layers'
513-slot rings, so it needs only a state sized for the history; the Fixed* cache objects are the native path's.
Writes bf16 bits as [tokens, vocab] little-endian uint16, which is tf-flint-run's `logits` layout.
"""

import argparse
import json
import platform
import time
from pathlib import Path

import mlx.core as mx

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

VOCAB = 150144


def bf16_bytes(bits: mx.array) -> bytes:
    words = bits.flatten().tolist()
    raw = bytearray(2 * len(words))
    for index, word in enumerate(words):
        raw[2 * index] = word & 0xFF
        raw[2 * index + 1] = (word >> 8) & 0xFF
    return bytes(raw)


def decode_rows(engine, tokens: list[int]) -> mx.array:
    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["initial"] = 0
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    rows = []
    for token in tokens:
        logits, _trace = engine.forward(state, token)
        rows.append(logits.astype(mx.bfloat16))
    out = mx.stack(rows, axis=0).reshape(1, len(tokens), VOCAB)
    mx.eval(out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    mx.set_cache_limit(2 * 1024**3)
    tokens = json.loads(args.ids.read_text())
    started = time.perf_counter()
    model = QATModel(args.model)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    loaded = time.perf_counter()
    logits = decode_rows(engine, tokens)
    finished = time.perf_counter()
    args.out.write_bytes(bf16_bytes(logits.view(mx.uint16)))
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "rows": len(tokens),
                "vocab": VOCAB,
                "dtype": "bfloat16",
                "layout": "little-endian uint16, [tokens, vocab], row-major",
                "oracle": "rec_mlx 0.1.0 e53663b262af4cb1ca498e596ca8b1a2d79e0800 resident decode",
                "tokens": len(tokens),
                "load_seconds": loaded - started,
                "forward_seconds": finished - loaded,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(tokens), "load_seconds": loaded - started, "forward_seconds": finished - loaded}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
