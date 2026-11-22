import json
import math
import re
from pathlib import Path

CONTEXT = 32768
VOCAB = 200058
MODEL_NAMES = frozenset({'Inkling-2-Mini-oQ4e-mtp@3eea042751cd', 'Inkling-2-Mini@ed40e7da625f'})


def validate_model_path(path):
    path = Path(path).expanduser()
    if path.parent != Path.home() / 'models' or path.name not in MODEL_NAMES:
        raise ValueError('model path does not identify a pinned checkpoint revision')
    return path


def redact_other_processes(output):
    return '\n'.join('[unrelated process details removed]'
                     if any(marker in line for marker in (' command=', 'GPU time users:', 'pid=', 'bundle_id=')) else line
                     for line in output.splitlines())


def read_prompt(path, steps):
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or len(ids) < 2 or any(type(i) is not int or not 0 <= i < VOCAB for i in ids):
        raise ValueError('prompt must contain Inkling token IDs')
    if type(steps) is not int or steps < 2 or len(ids) + steps > CONTEXT:
        raise ValueError('prompt and completion exceed the 32K context or have invalid length')
    return ids


def parse_result(output, prompt_tokens, steps):
    rows = []
    for line in output.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get('type') == 'omlx_comparator':
            rows.append(row)
    if len(rows) != 1:
        raise ValueError('expected one oMLX result')
    row = rows[0]
    ids = row.get('generated_ids')
    if row.get('prompt_tokens') != prompt_tokens or not isinstance(ids, list) or len(ids) != steps:
        raise ValueError('oMLX returned the wrong prompt or completion length')
    if any(type(i) is not int or not 0 <= i < VOCAB for i in ids):
        raise ValueError('oMLX returned invalid token IDs')
    if type(row.get('mlx_peak_bytes')) is not int or row['mlx_peak_bytes'] <= 0:
        raise ValueError('oMLX returned invalid MLX peak memory')
    for key in ('ttft_seconds', 'decode_seconds'):
        value = row.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid {key}')
    return row | {'prefill_tokens_per_second': prompt_tokens / row['ttft_seconds'],
                  'decode_tokens_per_second': (steps - 1) / row['decode_seconds']}


def mtp_cycles(stderr):
    cycles = [int(value) for value in re.findall(r'MTP\[[^]]+\] finish=\S+ tokens=\d+ cycles=(\d+)', stderr)]
    if not cycles or not sum(cycles):
        raise ValueError('Lightning MTP reported no draft/verify cycles')
    return sum(cycles)
