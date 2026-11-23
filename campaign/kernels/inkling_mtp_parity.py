"""Compare TensorFold's MTP draft logits with mlx-vlm's Inkling 2 DSpark head on the same tapped target states.

usage: inkling_mtp_parity.py MODEL_DIR DUMP_PREFIX [--dtype float32|bfloat16]

DUMP_PREFIX names the files the drafter GPU test writes when TF_INKLING2_MTP_DUMP is set: `.json` (positions, anchor,
TensorFold's drafts), `.taps.f32` ([positions][taps][hidden]) and `.logits.f32` ([steps][unpadded vocab]). The
reference head (`mlx_vlm.speculative.drafters.inkling2_dspark`, adapted from vLLM) drafts from the same taps and anchor,
with its Markov chain teacher-forced on TensorFold's tokens so every step's logits are comparable. In float32 the
reference keeps TensorFold's two bf16 rounding points: the tap mixture weights and the folded unembedding.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.speculative.drafters.inkling2_dspark import dspark as ref


def config_of(model_dir: Path) -> SimpleNamespace:
    full = json.loads((model_dir / "config.json").read_text())
    text, mtp = full["text_config"], full["mtp_config"]
    return SimpleNamespace(
        num_hidden_layers=mtp["num_hidden_layers"], hidden_size=mtp["hidden_size"],
        intermediate_size=mtp["intermediate_size"], num_attention_heads=mtp["num_attention_heads"],
        num_key_value_heads=mtp["num_key_value_heads"], head_dim=mtp["head_dim"], rms_norm_eps=mtp["rms_norm_eps"],
        rope_theta=mtp["rope_theta"], sliding_window_size=mtp["sliding_window_size"],
        local_layer_ids=mtp["local_layer_ids"], markov_rank=mtp["markov_rank"], block_size=mtp["block_size"],
        target_layer_ids=mtp["target_layer_ids"], vocab_size=text["vocab_size"],
        unpadded_vocab_size=text.get("unpadded_vocab_size", text["vocab_size"]),
        final_logit_softcapping=text.get("final_logit_softcapping", 30.0))


def target_tensors(model_dir: Path, names: list[str]) -> dict:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted({index[n] for n in names}):
        loaded = mx.load(str(model_dir / shard))
        for n in names:
            if index[n] == shard:
                out[n] = loaded[n]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("dump")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = parser.parse_args()
    dtype = mx.float32 if args.dtype == "float32" else mx.bfloat16
    meta = json.loads(Path(args.dump + ".json").read_text())
    n, steps, vocab, hidden, taps = meta["positions"], meta["steps"], meta["vocab"], meta["hidden"], meta["taps"]
    tapped = np.fromfile(args.dump + ".taps.f32", dtype=np.float32).reshape(n, taps, hidden)
    ours = np.fromfile(args.dump + ".logits.f32", dtype=np.float32).reshape(steps, vocab)
    config = config_of(args.model)

    model = ref.Model(config)
    model.load_weights(list(mx.load(str(args.model / "mtp.safetensors")).items()), strict=True)
    if dtype == mx.float32:
        model.update(nn.utils.tree_map(lambda x: x.astype(mx.float32), model.parameters()))
        def mix(hidden, row, norm):
            # The reference's mixture with its weights stored at bf16, as TensorFold and a bf16 context store them.
            features = hidden.reshape(*hidden.shape[:2], -1, config.hidden_size).astype(mx.float32)
            alpha = mx.softmax(model.fuse_weight.astype(mx.float32), axis=-1).astype(mx.bfloat16).astype(mx.float32)[row]
            mixed = mx.zeros_like(features[:, :, 0])
            for i in range(features.shape[2]):
                mixed = mixed + features[:, :, i] * alpha[i]
            return norm(mixed).astype(hidden.dtype)

        model.mix = mix

    names = ["model.embed.weight", "model.embed.scales", "model.embed.biases", "model.embed_norm.weight",
             "lm_head.weight", "lm_head.scales", "lm_head.biases"]
    t = target_tensors(args.model, names)
    embed = nn.QuantizedEmbedding(config.vocab_size, hidden, group_size=64, bits=8)
    embed.update({"weight": t["model.embed.weight"], "scales": t["model.embed.scales"], "biases": t["model.embed.biases"]})
    embed_norm = nn.RMSNorm(hidden, eps=config.rms_norm_eps)
    embed_norm.weight = t["model.embed_norm.weight"]
    quant_head = nn.QuantizedLinear(hidden, config.vocab_size, bias=False, group_size=64, bits=8)
    quant_head.update({"weight": t["lm_head.weight"], "scales": t["lm_head.scales"], "biases": t["lm_head.biases"]})
    model.markov_head.fold(quant_head)
    if dtype == mx.float32:
        dense = mx.dequantize(t["lm_head.weight"], t["lm_head.scales"], t["lm_head.biases"], group_size=64, bits=8).astype(mx.float32)
        head = nn.Linear(hidden, config.vocab_size, bias=False)
        head.weight = dense
        markov = model.markov_head
        markov._fold_weight = markov._fold_weight.astype(mx.bfloat16).astype(mx.float32)
        markov._fold_bias = markov._fold_bias.astype(mx.bfloat16).astype(mx.float32)
        embed_norm.weight = embed_norm.weight.astype(mx.float32)

        def embed_fn(ids):
            return embed_norm(embed(ids).astype(mx.float32))
    else:
        head = quant_head

        def embed_fn(ids):
            return embed_norm(embed(ids))

    model._target = SimpleNamespace(model=SimpleNamespace(embed=embed_fn), lm_head=head)
    hidden_states = mx.array(tapped.reshape(1, n, taps * hidden)).astype(dtype)

    def run(tokens):
        seen = []

        def sampler(logits):
            seen.append(np.array(logits.astype(mx.float32))[0])
            if tokens is None:
                return mx.argmax(logits, axis=-1).astype(mx.int32)
            return mx.array([tokens[len(seen) - 1]], dtype=mx.int32)

        drafted = model.draft_block(meta["anchor"], hidden_states, model.make_cache(), config.block_size, sampler, mx.int32)
        mx.eval(drafted)
        return [int(x) for x in np.array(drafted)[0]], seen

    forced, theirs = run(meta["drafts"])
    free, _ = run(None)
    report = {"dtype": args.dtype, "positions": n, "anchor": meta["anchor"], "ours": meta["drafts"], "reference": free,
              "equal_prefix": next((i for i, (a, b) in enumerate(zip(meta["drafts"], free)) if a != b), steps), "steps": []}
    for s in range(steps):
        a, b = ours[s], theirs[s][:vocab]
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        top_ours, top_ref = int(np.argmax(a)), int(np.argmax(b))
        order = np.argsort(-b)
        report["steps"].append({
            "step": s, "max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "scale": float(np.abs(b).max()),
            "argmax_equal": top_ours == top_ref, "ours": top_ours, "reference": top_ref,
            "reference_rank_of_ours": int(np.nonzero(order == top_ours)[0][0]),
            "reference_margin": float(b[order[0]] - b[order[1]]),
            "top5_overlap": len(set(np.argsort(-a)[:5].tolist()) & set(order[:5].tolist())),
        })
    json.dump(report, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
