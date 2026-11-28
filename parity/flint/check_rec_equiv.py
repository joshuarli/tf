"""Check this harness against rec_mlx's own assertion: a resident decode step equals a native one-token forward.

rec's tests/validate_resident.py asserts that `engine.forward(state, token)` matches the native model's one-token
forward over the same cache, bit for bit. If that fails here, the harness is wrong and no comparison against
TensorFold means anything yet.
"""

import json
import platform
import sys
from pathlib import Path

import mlx.core as mx

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

VOCAB = 150144


def main() -> int:
    ids = Path(sys.argv[1])
    row = int(sys.argv[2])
    tokens = json.loads(ids.read_text())
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    mx.set_cache_limit(2 * 1024**3)
    model = QATModel(Path.home() / "models" / "flint-int4@3b93f9053b53")
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())

    prompt = tokens[:row]
    session = engine.make_session(max_new_tokens=len(tokens) + 1)
    session.prefill(prompt)
    native_cache = session.prompt_cache.layers
    for metadata_row, cache in zip(engine.metadata[:49], native_cache, strict=True):
        if not metadata_row[1]:
            cache.grow(len(tokens) + 1)

    hidden = model.native.model.embed_tokens(mx.array([[tokens[row]]], mx.int32))
    for layer, cache in zip(model.native.layers, native_cache, strict=True):
        hidden = layer(hidden, None, cache)
    native_row = model.native.lm_head(model.native.model.norm(hidden)).reshape(-1).astype(mx.bfloat16)
    mx.eval(native_row)

    resident_row = session.forward_token(tokens[row]).astype(mx.bfloat16).reshape(-1)
    mx.eval(resident_row)

    a, b = native_row.view(mx.uint16), resident_row.view(mx.uint16)
    same = mx.array_equal(a, b).item()
    fa = native_row.astype(mx.float32)
    fb = resident_row.astype(mx.float32)
    diff = mx.abs(fa - fb)
    mx.eval(diff)
    print(
        json.dumps(
            {
                "row": row,
                "prompt": len(prompt),
                "bit_identical": same,
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
                "native_top1": int(mx.argmax(fa).item()),
                "resident_top1": int(mx.argmax(fb).item()),
                "logit_scale": float(mx.abs(fa).max().item()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
