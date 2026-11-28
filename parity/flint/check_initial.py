"""Does the resident decode kernel's `initial` state field change arithmetic?

This oracle feeds a whole history through engine.forward one token at a time from position 0 with initial=0 and
position incrementing. rec_mlx's own session sets initial=len(prompt) after QATSession.prefill and then decodes.
If initial changes any output bit, the two setups are different oracles and the choice has to be stated.
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

VOCAB = 150144


def feed(engine, tokens, initial_mode):
    state = engine.prefill_state(len(tokens), limit=len(tokens) + 1)
    state["kv"].grow(len(tokens))
    state["capacity"] = state["kv"].capacity
    rows = []
    for index, token in enumerate(tokens):
        state["initial"] = 0 if initial_mode == "zero" else index
        logits, _trace = engine.forward(state, token)
        rows.append(np.array(logits.astype(mx.bfloat16).view(mx.uint16)).reshape(VOCAB))
    mx.eval(rows)
    return np.stack(rows)


def main() -> int:
    ids = Path(sys.argv[1])
    tokens = json.loads(ids.read_text())
    mx.set_cache_limit(2 * 1024**3)
    model = QATModel(Path.home() / "models" / "flint-int4@3b93f9053b53")
    engine, _ = build_decoder(model, optimized=True)
    mx.eval(model.native.parameters())
    zero = feed(engine, tokens, "zero")
    moving = feed(engine, tokens, "moving")
    same = np.array_equal(zero, moving)
    diff = np.abs(
        (zero.astype(np.uint32) << 16).view(np.float32) - (moving.astype(np.uint32) << 16).view(np.float32)
    )
    print(
        json.dumps(
            {
                "rows": len(tokens),
                "initial_zero_equals_moving": bool(same),
                "max_abs_diff": float(diff.max()),
                "first_differing_row": int(np.nonzero(diff.max(axis=1) > 0)[0][0]) if not same else None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
