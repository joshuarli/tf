"""rec_mlx teacher-forced logits for every corpus history, one process, loaded once.

Uses the path rec_mlx itself serves Flint with: the resident decoder (KVStorage in bf16) fed one token at a
time from position 0. rec's tests/validate_resident.py asserts that step equals the native model's one-token
forward bit for bit, which check_rec_equiv.py re-checks here before any history is measured.

Writes <out>/<name>.bin as bf16 bits, little-endian uint16, [tokens, vocab] -- tf-flint-run's `logits` layout.
"""

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


def rows(engine, tokens: list[int]) -> mx.array:
    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["initial"] = 0
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    out = [engine.forward(state, token)[0].astype(mx.bfloat16).reshape(VOCAB) for token in tokens]
    stacked = mx.stack(out, axis=0).reshape(1, len(tokens), VOCAB)
    mx.eval(stacked)
    return stacked


def equivalence(model, engine, tokens: list[int], row: int) -> dict:
    """rec's own claim: a resident step equals the native one-token forward over the same cache."""
    session = engine.make_session(max_new_tokens=len(tokens) + 1)
    session.prefill(tokens[:row])
    cache = session.prompt_cache.layers
    for entry, layer in zip(engine.metadata[:49], cache, strict=True):
        if not entry[1]:
            layer.grow(len(tokens) + 1)
    hidden = model.native.model.embed_tokens(mx.array([[tokens[row]]], mx.int32))
    for layer, layer_cache in zip(model.native.layers, cache, strict=True):
        hidden = layer(hidden, None, layer_cache)
    native = model.native.lm_head(model.native.model.norm(hidden)).reshape(-1).astype(mx.bfloat16)
    resident = session.forward_token(tokens[row]).astype(mx.bfloat16).reshape(-1)
    mx.eval(native, resident)
    return {
        "row": row,
        "bit_identical": bool(mx.array_equal(native.view(mx.uint16), resident.view(mx.uint16)).item()),
        "max_abs_diff": float(mx.abs(native.astype(mx.float32) - resident.astype(mx.float32)).max().item()),
    }


def main() -> int:
    import sys

    corpus_path, model_dir, out_dir, equivalence_row = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
    )
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    corpus = json.loads(corpus_path.read_text())
    mx.set_cache_limit(2 * 1024**3)
    started = time.perf_counter()
    model = QATModel(model_dir)
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    loaded = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"load_seconds": loaded - started, "oracle": "rec_mlx 0.1.0 e53663b262af4cb1ca498e596ca8b1a2d79e0800 resident decode", "histories": {}}
    report["equivalence"] = equivalence(model, engine, corpus["histories"][0]["ids"], equivalence_row)
    for entry in corpus["histories"]:
        at = time.perf_counter()
        logits = rows(engine, entry["ids"])
        out_dir.joinpath(f"{entry['name']}.bin").write_bytes(bf16_bytes(logits.view(mx.uint16)))
        report["histories"][entry["name"]] = {
            "tokens": entry["tokens"],
            "seconds": time.perf_counter() - at,
            "tokens_per_second": entry["tokens"] / (time.perf_counter() - at),
        }
        print(json.dumps({"history": entry["name"], **report["histories"][entry["name"]]}), flush=True)
    out_dir.joinpath("oracle.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"load_seconds": report["load_seconds"], "equivalence": report["equivalence"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
