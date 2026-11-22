import json
from pathlib import Path
import re
import statistics

records = [json.loads(line) for line in Path('campaign/comparators/nanoattention/runs/pairs.jsonl').read_text().splitlines()]
accepted = [r for r in records if r['exit'] == 0]

def sample(record):
    text = record['output']
    charged = float(re.search(r'heavy: done: peak charged ([0-9.]+) GB', text).group(1))
    if record['engine'] == 'tensorfold':
        prefill = float(re.search(r'prefill: \d+ tokens in [0-9.]+ ms, ([0-9.]+) tok/s chunk 512', text).group(1))
        millis = float(re.search(r'bench at \d+: ([0-9.]+) ms a step wall', text).group(1))
        return prefill, 1000 / millis, charged
    value = next(json.loads(line) for line in text.splitlines() if line.startswith('{"type":"nano_bench"'))
    return value['prefill_tokens_per_second'], value['decode_tokens_per_second'], charged

def display(values):
    return f'{statistics.median(values):.2f} ({min(values):.2f}–{max(values):.2f})'

print('| Prompt | Configuration | Engine | Context capacity | Prefill tok/s | Decode tok/s | Peak charged GB | First differing index |')
print('| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |')
for size in (512, 1024):
    for config in ('reference', 'optimized'):
        tf = [r for r in accepted if r['prompt_tokens'] == size and r['config'] == config and r['engine'] == 'tensorfold']
        nano = [r for r in accepted if r['prompt_tokens'] == size and r['config'] == config and r['engine'] == 'nano']
        assert sorted(r['pair'] for r in tf) == [1, 2, 3]
        assert sorted(r['pair'] for r in nano) == [1, 2, 3]
        assert len({tuple(r['predictions']) for r in tf}) == 1
        assert len({tuple(r['predictions']) for r in nano}) == 1
        a, b = tf[0]['predictions'], nano[0]['predictions']
        assert len(a) == len(b) == 65
        difference = next((str(i) for i, (x, y) in enumerate(zip(a, b)) if x != y), 'none in 65')
        for name, runs, capacity in [('TensorFold', tf, size + 65), ('nano', nano, size + 64)]:
            samples = [sample(r) for r in runs]
            print(f'| {size} | {config} | {name} | {capacity} | {display([s[0] for s in samples])} | {display([s[1] for s in samples])} | {max(s[2] for s in samples):.1f} | {difference} |')
