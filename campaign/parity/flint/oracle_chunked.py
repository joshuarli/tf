"""rec_mlx's chunked native prefill over the same corpus, as the oracle's own spread on identical inputs.

rec_mlx groups prefill sums by chunk, so its chunked prefill and one-token decode differ in BF16 rounding on the
same tokens. Comparing TensorFold with this tells apart "TensorFold's kernels differ" from "two accumulation orders
differ", which is the distinction the report has to make before calling anything noise.

Writes <out>/<name>.bin as bf16 bits, little-endian uint16, [tokens, vocab].
"""

import json
import platform
import sys
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


def chunked(model, engine, tokens: list[int], chunk: int) -> mx.array:
    state = engine.prefill_state(min(len(tokens), chunk), limit=len(tokens) + 1)
    cache = model.make_cache(len(tokens), state["kv"])
    out = []
    for begin in range(0, len(tokens), chunk):
        state["kv"].grow(min(begin + chunk, len(tokens)))
        inputs = mx.array([tokens[begin : begin + chunk]], dtype=mx.int32)
        hidden = model.native.model(inputs, cache)
        logits = model.native.lm_head(hidden)
        mx.eval(logits, [entry.state for entry in cache])
        out.append(logits.astype(mx.bfloat16))
    rows = mx.concatenate(out, axis=1)
    mx.eval(rows)
    return rows


def main() -> int:
    corpus_path, model_dir, out_dir, chunk = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
    )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    mx.set_cache_limit(2 * 1024**3)
    corpus = json.loads(corpus_path.read_text())
    started = time.perf_counter()
    model = QATModel(model_dir)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    print(json.dumps({"load_seconds": time.perf_counter() - started}), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in corpus["histories"]:
        at = time.perf_counter()
        logits = chunked(model, engine, entry["ids"], chunk)
        out_dir.joinpath(f"{entry['name']}.bin").write_bytes(bf16_bytes(logits.view(mx.uint16)))
        print(json.dumps({"history": entry["name"], "tokens": entry["tokens"], "seconds": time.perf_counter() - at}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
