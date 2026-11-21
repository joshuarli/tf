"""Per-layer hidden states for chosen rows on both of rec_mlx's teacher-forced paths.

Both paths run the checkpoint's own modules — one token at a time from position 0 (the resident decode path
`oracle.py` measures) and as one native prefill chunk (`oracle_chunked.py`). Each path's kept rows are compared
at their logits with the bits those scripts wrote, so an instrumented path is only used where it reproduces the
oracle's own output.

Writes <out>/<history>.decode.npz, <history>.chunk.npz and <history>/oracle-trace.json.
"""

import json
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.base import create_attention_mask

from rec_mlx.kernels import build_decoder
from rec_mlx.quantized import QATModel

VOCAB = 150144
CHUNK = 4096


def numpy(x) -> np.ndarray:
    return np.array(x.astype(mx.float32)).reshape(-1)


def layer_traced(layer, x, mask, cache, rows):
    """DecoderLayer's own expressions on the whole chunk, with kept rows sliced out per sublayer."""
    h_in = layer.input_layernorm(x)
    attn = layer.self_attn(h_in, mask, cache)
    h1 = x + attn
    mlp_in = layer.post_attention_layernorm(h1)
    stages = []
    for r in rows:
        stages.append(
            {
                "attn": numpy(attn[0, r]),
                "resid_attn": numpy(h1[0, r]),
                "normed": numpy(mlp_in[0, r]),
                "input_normed": numpy(h_in[0, r]),
            }
        )
    if hasattr(layer.mlp, "gate"):
        gate = layer.mlp.gate(mlp_in).astype(mx.float32)
        scores = mx.sigmoid(gate)
        choice = scores + layer.mlp.gate.expert_bias.astype(mx.float32)
        indices = mx.stop_gradient(mx.argpartition(-choice, kth=layer.mlp.top_k - 1, axis=-1)[..., : layer.mlp.top_k])
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        slots = layer.mlp.switch_mlp(mlp_in, indices)
        routed = (slots * weights[..., None].astype(mlp_in.dtype)).sum(axis=-2)
        shared = layer.mlp.shared_expert(mlp_in)
        for r, stage in zip(rows, stages):
            stage["router"] = numpy(layer.mlp.gate(mlp_in)[0, r])
            stage["picks"] = np.array(indices[0, r]).astype(np.int32)
            stage["weights"] = numpy(weights[0, r])
            stage["slots"] = np.array(slots[0, r].astype(mx.float32)).reshape(-1)
            stage["shared"] = numpy(shared[0, r])
            stage["routed"] = numpy(routed[0, r])
            stage["mlp"] = numpy((routed + shared)[0, r])
        out = h1 + (routed + shared)
    else:
        mlp = layer.mlp(mlp_in)
        for r, stage in zip(rows, stages):
            stage["mlp"] = numpy(mlp[0, r])
        out = h1 + mlp
    for r, stage in zip(rows, stages):
        stage["resid"] = numpy(out[0, r])
    return out, stages


def blank(layers: int, hidden: int, experts: int, topk: int, rows: int) -> dict:
    z = lambda *s: np.full(s, np.nan, np.float32)
    return {
        "embed": z(rows, hidden),
        "attn": z(rows, layers, hidden),
        "resid": z(rows, layers, hidden),
        "resid_attn": z(rows, layers, hidden),
        "mlp": z(rows, layers, hidden),
        "normed": z(rows, layers, hidden),
        "input_normed": z(rows, layers, hidden),
        "router": z(rows, layers, experts),
        "picks": np.full((rows, layers, topk), -1, np.int32),
        "weights": z(rows, layers, topk),
        "slots": z(rows, layers, topk, hidden),
        "shared": z(rows, layers, hidden),
        "routed": z(rows, layers, hidden),
        "final_norm": z(rows, hidden),
        "logits": np.zeros((rows, VOCAB), np.uint16),
        "logits_verified": np.zeros(rows, bool),
    }


def store(out, slot, i, stage):
    out["attn"][slot, i] = stage["attn"]
    out["resid"][slot, i] = stage["resid"]
    out["resid_attn"][slot, i] = stage["resid_attn"]
    out["mlp"][slot, i] = stage["mlp"]
    out["normed"][slot, i] = stage["normed"]
    out["input_normed"][slot, i] = stage["input_normed"]
    if "router" in stage:
        out["router"][slot, i] = stage["router"]
        out["picks"][slot, i] = stage["picks"]
        out["weights"][slot, i] = stage["weights"]
        out["slots"][slot, i] = stage["slots"].reshape(out["slots"].shape[-2], out["slots"].shape[-1])
        out["shared"][slot, i] = stage["shared"]
        out["routed"][slot, i] = stage["routed"]


def dims(native):
    c = native.args
    return c.num_hidden_layers, c.hidden_size, c.num_experts, c.num_experts_per_tok


def decode_pass(model, native, engine, tokens, keep):
    T = len(tokens)
    layers, hidden, experts, topk = dims(native)
    state = engine.prefill_state(T + 1, limit=T + 1)
    state["initial"] = 0
    state["kv"].grow(T + 1)
    state["capacity"] = state["kv"].capacity
    cache = model.make_cache(T + 1, state["kv"])
    where = {row: i for i, row in enumerate(keep)}
    out = blank(layers, hidden, experts, topk, len(keep))
    for t, tok in enumerate(tokens):
        h = native.model.embed_tokens(mx.array([[tok]], mx.int32))
        slot = where.get(t)
        if slot is not None:
            out["embed"][slot] = numpy(h)
        for i, (layer, c) in enumerate(zip(native.layers, cache, strict=True)):
            if slot is None:
                h = layer(h, None, c)
                continue
            h, stages = layer_traced(layer, h, None, c, [0])
            store(out, slot, i, stages[0])
        mx.eval(h)
        if slot is not None:
            normed = native.model.norm(h)
            logits = native.lm_head(normed).reshape(-1).astype(mx.bfloat16)
            mx.eval(logits)
            out["final_norm"][slot] = numpy(normed)
            out["logits"][slot] = np.array(logits.view(mx.uint16))
    return out


def chunk_pass(model, native, engine, tokens, keep):
    T = len(tokens)
    layers, hidden, experts, topk = dims(native)
    state = engine.prefill_state(min(T, CHUNK), limit=T + 1)
    cache = model.make_cache(T, state["kv"])
    state["kv"].grow(T)
    inputs = mx.array([tokens], dtype=mx.int32)
    h = native.model.embed_tokens(inputs)
    full_mask = None
    if native.model.full_index is not None:
        full_mask = create_attention_mask(h, cache[native.model.full_index])
    swa_mask = None
    if native.model.swa_index is not None:
        swa_mask = create_attention_mask(h, cache[native.model.swa_index], window_size=native.model.args.sliding_window)
    out = blank(layers, hidden, experts, topk, len(keep))
    for slot, row in enumerate(keep):
        out["embed"][slot] = numpy(h[0, row])
    for i, (layer, c, kind) in enumerate(zip(native.layers, cache, native.model.layer_types, strict=False)):
        mask = full_mask if kind == "full_attention" else swa_mask
        h, stages = layer_traced(layer, h, mask, c, keep)
        for slot, stage in enumerate(stages):
            store(out, slot, i, stage)
        mx.eval(h, [entry.state for entry in cache])
    normed = native.model.norm(h)
    logits = native.lm_head(normed).astype(mx.bfloat16)
    mx.eval(normed, logits)
    for slot, row in enumerate(keep):
        out["final_norm"][slot] = numpy(normed[0, row])
        out["logits"][slot] = np.array(logits[0, row].view(mx.uint16))
    return out


def main() -> int:
    corpus_path, model_dir, out_dir, keep_path = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    bins_root = Path(sys.argv[5]) if len(sys.argv) > 5 else out_dir.parent
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("Flint requires an Apple Silicon Mac")
    mx.set_cache_limit(2 * 1024**3)
    corpus = json.loads(corpus_path.read_text())
    keep_map = json.loads(keep_path.read_text())
    model = QATModel(model_dir)
    engine, _ = build_decoder(model, optimized=True)
    native = model.native
    mx.eval(native.parameters())
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = bins_root / "oracle"
    chunk_bins = bins_root / "oracle-chunked"
    report = {}
    for entry in corpus["histories"]:
        name = entry["name"]
        keep = sorted(keep_map.get(name, []))
        if not keep:
            continue
        tokens = entry["ids"]
        started = time.perf_counter()
        decode = decode_pass(model, native, engine, tokens, keep)
        decode_seconds = time.perf_counter() - started
        mx.clear_cache()
        started = time.perf_counter()
        chunk = chunk_pass(model, native, engine, tokens, keep)
        chunk_seconds = time.perf_counter() - started
        mx.clear_cache()
        ref = np.fromfile(bins / f"{name}.bin", dtype=np.uint16).reshape(-1, VOCAB)
        ref_chunk = np.fromfile(chunk_bins / f"{name}.bin", dtype=np.uint16).reshape(-1, VOCAB)
        for slot, row in enumerate(keep):
            decode["logits_verified"][slot] = bool(np.array_equal(decode["logits"][slot], ref[row]))
            chunk["logits_verified"][slot] = bool(np.array_equal(chunk["logits"][slot], ref_chunk[row]))
        np.savez(out_dir / f"{name}.decode.npz", **decode)
        np.savez(out_dir / f"{name}.chunk.npz", **chunk)
        report[name] = {
            "tokens": len(tokens),
            "keep": keep,
            "decode_seconds": decode_seconds,
            "chunk_seconds": chunk_seconds,
            "decode_logits_match_bin": [bool(v) for v in decode["logits_verified"]],
            "chunk_logits_match_bin": [bool(v) for v in chunk["logits_verified"]],
        }
        print(json.dumps({"history": name, **report[name]}), flush=True)
    (out_dir / "oracle-trace.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
