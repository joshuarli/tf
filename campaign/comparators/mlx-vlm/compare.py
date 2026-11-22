import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate.ar import generate_step

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('prompt')
parser.add_argument('--steps', type=int, default=16)
args = parser.parse_args()
ids = json.loads(Path(args.prompt).read_text())
if not ids or args.steps < 2:
    raise ValueError('expected a nonempty prompt and at least two output tokens')
model, _ = load(args.model)
prompt = mx.array([ids])
mx.eval(prompt)
tokens = generate_step(prompt, model, None, None, max_tokens=args.steps, temperature=0, prefill_step_size=128)
start = time.perf_counter()
first, _ = next(tokens)
first_at = time.perf_counter()
out = [first]
for _ in range(args.steps - 1):
    token, _ = next(tokens)
    out.append(token)
end = time.perf_counter()
print(json.dumps({'prompt_tokens': len(ids), 'generated_ids': out, 'ttft_s': first_at - start, 'prefill_tps': len(ids) / (first_at - start), 'decode_tps': (len(out) - 1) / (end - first_at)}), flush=True)
