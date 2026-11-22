import argparse
import os
import struct
from pathlib import Path

# MLX enables TF32 by default (mlx/utils.h:225-227); disable it for f32 reference arithmetic.
os.environ['MLX_ENABLE_TF32'] = '0'

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from mlx_vlm import load
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.speculative.drafters import load_drafter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('output')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model, _ = load(args.model)
    draft, kind = load_drafter(args.model)
    assert kind == 'dflash' and draft.config.block_size == 8
    records = []
    original = draft.draft_block

    def capture(anchor, hidden, cache, width, sampler, token_dtype=mx.int32):
        mx.eval(hidden)
        records.append((int(anchor), np.array(hidden.astype(mx.float32)), int(width)))
        return original(anchor, hidden, cache, width, sampler, token_dtype)

    draft.draft_block = capture
    starts = []
    for name in ('code', 'prose', 'structured'):
        for width in (4, 8):
            starts.append(len(records))
            prompt = Path(f'campaign/prompts/dev/{name}-1024.txt').read_text()
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=True
            )['input_ids']
            generation = generate_step(
                mx.array([ids]), model, None, None, max_tokens=32, temperature=0,
                prefill_step_size=128, draft_model=draft, draft_kind=kind,
                draft_block_size=width,
            )
            list(generation)
            print(name, width, len(records) - starts[-1], flush=True)
    starts.append(len(records))
    arrays = {'starts': np.array(starts, dtype=np.int32)}
    for i, (anchor, hidden, width) in enumerate(records):
        arrays[f'anchor_{i}'] = np.array(anchor, dtype=np.int32)
        arrays[f'hidden_{i}'] = hidden
        arrays[f'width_{i}'] = np.array(width, dtype=np.int32)
    print('captured', len(records), flush=True)
    target = model.language_model
    target.model.layers = []
    mx.clear_cache()
    draft.set_dtype(mx.float32)
    target.model.embed_tokens.set_dtype(mx.float32)
    target.model.embed_norm.set_dtype(mx.float32)
    target.lm_head.set_dtype(mx.float32)
    mx.eval(draft.parameters(), target.model.embed_tokens.parameters(),
            target.model.embed_norm.parameters(), target.lm_head.parameters())
    draft.markov_head.fold(target.lm_head)
    for group in range(len(starts) - 1):
        cache = draft.make_cache()
        for i in range(starts[group], starts[group + 1]):
            anchor, hidden, width = records[i]
            logits = []

            def sample(values):
                mx.eval(values)
                logits.append(np.array(values))
                return mx.argmax(values, axis=-1)

            proposals = original(anchor, mx.array(hidden), cache, width, sample)
            mx.eval(proposals)
            arrays[f'proposals_{i}'] = np.array(proposals)
            arrays[f'logits_{i}'] = np.concatenate(logits, axis=0)
        print('replayed', group, flush=True)
    with open(args.output, 'wb') as output:
        output.write(b'DSPARK01')
        output.write(struct.pack('<I', len(starts) - 1))
        for group in range(len(starts) - 1):
            output.write(struct.pack('<I', starts[group + 1] - starts[group]))
            for i in range(starts[group], starts[group + 1]):
                anchor, hidden, width = records[i]
                logits = arrays[f'logits_{i}']
                proposals = arrays[f'proposals_{i}']
                output.write(struct.pack('<IIII', anchor, width, hidden.shape[1], logits.shape[1]))
                output.write(np.asarray(hidden, dtype='<f4').tobytes())
                output.write(np.asarray(logits, dtype='<f4').tobytes())
                output.write(np.asarray(proposals, dtype='<i4').tobytes())
    print('saved', args.output, flush=True)


if __name__ == '__main__':
    main()
