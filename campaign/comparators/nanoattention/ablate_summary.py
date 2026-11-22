import ast
import json
from pathlib import Path
from statistics import median

source = Path('campaign/comparators/nanoattention/ablate.py').read_text()
assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'knobs' for target in node.targets))
knobs = ast.literal_eval(assignment.value)

records = [json.loads(line) for line in Path('campaign/comparators/nanoattention/runs/ablation.jsonl').read_text().splitlines()]
accepted = {}
for row in records:
    if row['exit'] == 0:
        key = row['candidate'], row['round'], row['position']
        if key in accepted:
            raise ValueError(f'duplicate accepted run: {key}')
        accepted[key] = row

print('| Removed | Control prefill | Candidate prefill | Prefill ratio | Control decode | Candidate decode | Decode ratio |')
print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
for entry in knobs:
    name = entry.split('=')[0]
    candidates = []
    controls = []
    ratios = {'prefill': [], 'decode': []}
    for rnd in range(1, 4):
        before, candidate, after = [accepted[name, rnd, position] for position in ('before', 'candidate', 'after')]
        candidates.append(candidate)
        controls.extend((before, after))
        for metric in ratios:
            ratios[metric].append(candidate[metric] / ((before[metric] + after[metric]) / 2))
    print(f"| {name} | {median(x['prefill'] for x in controls):.2f} | {median(x['prefill'] for x in candidates):.2f} | {median(ratios['prefill']):.3f} | {median(x['decode'] for x in controls):.2f} | {median(x['decode'] for x in candidates):.2f} | {median(ratios['decode']):.3f} |")
