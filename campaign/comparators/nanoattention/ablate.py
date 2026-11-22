import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

root = Path.cwd()
work = Path('/Users/josh/repos/tensorfold-m5/.work/tmp/nanoattention-lane')
nano = work / 'nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199/zig-out/bin/nano'
model = '/Users/josh/models/Inkling-2-Mini@ed40e7da625f'
heavy = '/Users/josh/repos/tensorfold-m5/campaign/heavy.py'
receipt = root / 'campaign/comparators/nanoattention/runs/ablation.jsonl'
knobs = ['batch=128', 'expert_simd=true', 'expert_wide_dot=true', 'expert_gate_pair=true', 'expert_prefill=true', 'expert_wide_k=true', 'expert_bf16_tiles=true', 'expert_mlx_scale=true', 'bf16_linear_tile=true', 'bf16_prefill_tile=8', 'bf16_prefill_matrix=true', 'bf16_decode_vec=true', 'fuse_bf16_cast=true', 'concurrent_scratch=true', 'relative_split=true', 'relative_split_4=true', 'relative_tile_prefill=true']
groups = {'expert_prefill': {'expert_prefill', 'expert_wide_k', 'expert_bf16_tiles', 'expert_mlx_scale'}, 'relative_split': {'relative_split', 'relative_split_4'}}
env = os.environ.copy()
env.update({'PATH': '/opt/homebrew/bin:' + env['PATH'], 'TMPDIR': '/Users/josh/repos/tensorfold-m5/.work/tmp', 'XDG_CACHE_HOME': str(work / 'xdg-cache'), 'ZIG_GLOBAL_CACHE_DIR': '/Users/josh/repos/tensorfold-m5/.work/zig-global-cache', 'PYTHONDONTWRITEBYTECODE': '1', 'GPU_GATE_MAX_GPU_PCT': '35', 'GPU_GATE_MAX_UTIL_PCT': '35'})
ids = json.loads((root / 'campaign/prompts/inkling/code-1024.json').read_text())
workload = work / 'workload-1024.json'
assert json.loads(workload.read_text()) == {'bos': ids[0], 'code': ids[1:], 'decode': [0] * 64}
completed = set()
if receipt.exists():
    for line in receipt.read_text().splitlines():
        item = json.loads(line)
        if item['exit'] == 0:
            completed.add((item['candidate'], item['round'], item['position']))

for entry in knobs:
    name = entry.split('=')[0]
    removed = groups.get(name, {name})
    for round_number in range(1, 4):
        for position in ('before', 'candidate', 'after'):
            key = (name, round_number, position)
            if key in completed:
                continue
            selected = [k for k in knobs if position != 'candidate' or k.split('=')[0] not in removed]
            command = [str(nano), 'bench', model, '--context', '1088', '--prompt-tokens', '1024', '--decode-tokens', '64', '--repeats', '1', '--decode-mode', 'greedy', '--bench-tokens', str(workload)]
            for knob in selected:
                command.extend(['--knob', knob])
            body = 'gpu-gate-ready -1 || exit 75; exec ' + shlex.join(command)
            wrapped = ['python3', heavy, '--', 'sh', '-c', body]
            attempt = 0
            while True:
                attempt += 1
                print('start', key, 'attempt', attempt, flush=True)
                result = subprocess.run(wrapped, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                output = result.stdout
                item = {'candidate': name, 'removed': sorted(removed) if position == 'candidate' else [], 'round': round_number, 'position': position, 'attempt': attempt, 'exit': result.returncode}
                if result.returncode == 0:
                    lines = [line for line in output.splitlines() if line.startswith('{"type":"nano_bench"')]
                    if len(lines) != 1:
                        raise ValueError('expected exactly one nano result')
                    data = json.loads(lines[0])
                    if data['prompt_tokens'] != 1024 or data['decode_tokens'] != 64 or len(data['predictions']) != 65:
                        raise ValueError('unexpected nano workload')
                    item.update({'batch': data['batch'], 'prefill': data['prefill_tokens_per_second'], 'decode': data['decode_tokens_per_second'], 'prediction_checksum': data['prediction_checksum']})
                    m = re.search(r'peak charged ([0-9.]+) GB', output)
                    if not m:
                        raise ValueError('missing heavy receipt')
                    item['peak_gb'] = float(m.group(1))
                with receipt.open('a') as f:
                    f.write(json.dumps(item, separators=(',', ':')) + '\n')
                print('end', key, 'exit', result.returncode, 'peak', item.get('peak_gb'), 'prefill', item.get('prefill'), 'decode', item.get('decode'), flush=True)
                if result.returncode == 0:
                    break
                if result.returncode == 75:
                    time.sleep(7)
                    continue
                print(output, flush=True)
                if result.returncode in (3, 4):
                    print('STOPPED_ON: heavy.py refusing on memory', flush=True)
                elif re.search(r'unknown knob|invalid knob|InvalidKnob', output, re.I):
                    print('STOPPED_ON: nano knob names failing to parse', flush=True)
                sys.exit(result.returncode)
