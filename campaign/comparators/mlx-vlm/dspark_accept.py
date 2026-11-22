import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer
from mlx_vlm import load
from mlx_vlm.generate.ar import generate_step
from mlx_vlm.speculative.drafters import load_drafter

model_path = Path('/Users/josh/models/Inkling-2-Mini@ed40e7da625f')
name = sys.argv[1]
text = Path(f'campaign/prompts/dev/{name}-1024.txt').read_text()
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
ids = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, tokenize=True)['input_ids']
model, _ = load(str(model_path))
draft, kind = load_drafter(str(model_path))
assert kind == 'dflash' and draft.config.block_size == 8


def run(width):
    prompt = mx.array([ids])
    mx.eval(prompt)
    kwargs = {} if width == 0 else {'draft_model': draft, 'draft_kind': kind, 'draft_block_size': width}
    gen = generate_step(prompt, model, None, None, max_tokens=256, temperature=0, prefill_step_size=128, **kwargs)
    first, _ = next(gen)
    start = time.perf_counter()
    tokens = [first] + [token for token, _ in gen]
    elapsed = time.perf_counter() - start
    assert len(tokens) == 256, len(tokens)
    result = {'prompt': name, 'prompt_tokens': len(ids), 'width': width, 'tokens': len(tokens), 'decode_tps': 255 / elapsed, 'hash': hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}
    if width:
        lengths = list(draft.accept_lens)
        sizes = list(draft.draft_lens)
        full = [a for a, d in zip(lengths, sizes) if d == width - 1]
        assert full and all(0 <= a < width for a in full)
        result.update(rounds=len(full), accepted_histogram=dict(sorted(collections.Counter(full).items())), committed_per_round=1 + sum(full) / len(full), terminal_short_rounds=len(lengths) - len(full))
    print(json.dumps(result), flush=True)
    return result


baseline = run(0)
records = [baseline]
for width in (8, 4, 2):
    record = run(width)
    record['equal_ar'] = record['hash'] == baseline['hash']
    records.append(record)
with Path('campaign/comparators/mlx-vlm/runs/dspark-accept.jsonl').open('a') as output:
    for record in records:
        output.write(json.dumps(record, sort_keys=True) + '\n')
