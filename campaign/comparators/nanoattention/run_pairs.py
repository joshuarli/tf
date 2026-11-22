import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys
import time

root = Path.cwd()
work = Path('/Users/josh/repos/tensorfold-m5/.work/tmp/nanoattention-lane')
model = '/Users/josh/models/Inkling-2-Mini@ed40e7da625f'
heavy = '/Users/josh/repos/tensorfold-m5/campaign/heavy.py'
nano = str(work / 'nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199/zig-out/bin/nano')
tf = str(work / 'tf-build/bin/tf-inkling2-loop-run')
receipt = root / 'campaign/comparators/nanoattention/runs/pairs.jsonl'
env = os.environ.copy()
env.update({'PATH': '/opt/homebrew/bin:' + str(Path.home() / '.elan/bin') + ':' + env['PATH'], 'TMPDIR': '/Users/josh/repos/tensorfold-m5/.work/tmp', 'ZIG_GLOBAL_CACHE_DIR': '/Users/josh/repos/tensorfold-m5/.work/zig-global-cache', 'XDG_CACHE_HOME': str(work / 'xdg-cache'), 'PYTHONDONTWRITEBYTECODE': '1', 'GPU_GATE_MAX_GPU_PCT': '35', 'GPU_GATE_MAX_UTIL_PCT': '35'})
optimized = ['batch=128', 'expert_simd=true', 'expert_wide_dot=true', 'expert_gate_pair=true', 'expert_prefill=true', 'expert_wide_k=true', 'expert_bf16_tiles=true', 'expert_mlx_scale=true', 'bf16_linear_tile=true', 'bf16_prefill_tile=8', 'bf16_prefill_matrix=true', 'bf16_decode_vec=true', 'fuse_bf16_cast=true', 'concurrent_scratch=true', 'relative_split=true', 'relative_split_4=true', 'relative_tile_prefill=true']
done = set()
if receipt.exists():
    for line in receipt.read_text().splitlines():
        item = json.loads(line)
        if item['exit'] == 0:
            done.add((item['prompt_tokens'], item['config'], item['pair'], item['engine']))

critical = 0
for size in (512, 1024):
    ids = json.loads((root / f'campaign/prompts/inkling/code-{size}.json').read_text())
    assert len(ids) == size
    workload = work / f'workload-{size}.json'
    workload.write_text(json.dumps({'bos': ids[0], 'code': ids[1:], 'decode': [0] * 64}))
    for config in ('reference', 'optimized'):
        for pair in range(1, 4):
            order = ('tensorfold', 'nano') if (pair + (size == 1024) + (config == 'optimized')) % 2 else ('nano', 'tensorfold')
            for engine in order:
                key = (size, config, pair, engine)
                if key in done:
                    continue
                prompt = f'campaign/prompts/inkling/code-{size}.json'
                output = work / f'tf-{size}-{config}-{pair}.bin'
                if engine == 'tensorfold':
                    bench = [tf, model, prompt, 'bench', '64', '0', '512']
                    greedy = [tf, model, prompt, 'greedy-decode', '65', str(output)]
                    body = shlex.join(bench) + ' || exit $?; perf-gate-ready -1 || exit 75; GPU_GATE_IGNORE_PIDS=$$ gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; gpu-gate-ready -1 || exit 75; ' + shlex.join(greedy) + ' || exit $?; perf-gate-ready -1 || exit 75'
                else:
                    command = [nano, 'bench', model, '--context', str(size + 64), '--prompt-tokens', str(size), '--decode-tokens', '64', '--repeats', '1', '--decode-mode', 'greedy', '--bench-tokens', str(workload)]
                    if config == 'optimized':
                        for knob in optimized:
                            command += ['--knob', knob]
                    body = shlex.join(command) + ' || exit $?; perf-gate-ready -1 || exit 75'
                shell = 'perf-gate-ready || exit 75; gpu-gate-ready -1 || exit 75; ' + body + '; GPU_GATE_IGNORE_PIDS=$$ gpu-gate-ready -1 || exit 75'
                command = ['python3', heavy, '--', 'sh', '-c', shell]
                attempt = 0
                while True:
                    while subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip() != '1':
                        time.sleep(5)
                    attempt += 1
                    print(f'start {key} attempt={attempt}', flush=True)
                    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    output_text = '\n'.join('[unrelated process details removed]' if ' command=' in line else line for line in result.stdout.splitlines()) + '\n'
                    item = {'prompt_tokens': size, 'config': config, 'pair': pair, 'engine': engine, 'attempt': attempt, 'command': shlex.join(command), 'exit': result.returncode, 'output': output_text}
                    if result.returncode == 0 and engine == 'tensorfold':
                        data = output.read_bytes()
                        if len(data) != 65 * 4:
                            raise ValueError('unexpected TensorFold output length')
                        item['predictions'] = list(struct.unpack('<65I', data))
                    if result.returncode == 0 and engine == 'nano':
                        parsed = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"type":"nano_bench"'))
                        if len(parsed['predictions']) != 65 or parsed['prompt_tokens'] != size or parsed['decode_tokens'] != 64:
                            raise ValueError('unexpected nano workload')
                        item['predictions'] = parsed['predictions']
                    with receipt.open('a') as f:
                        f.write(json.dumps(item) + '\n')
                        f.flush()
                    print(f'end {key} exit={result.returncode}\n{result.stdout}', flush=True)
                    if result.returncode == 3:
                        print("STOPPED_ON: a run exceeds heavy.py's 28 GB cap", flush=True)
                        sys.exit(3)
                    if result.returncode == 4:
                        critical += 1
                        if critical >= 3:
                            print('STOPPED_ON: memory pressure turns critical on 3 consecutive attempts', flush=True)
                            sys.exit(4)
                    else:
                        critical = 0
                    if result.returncode == 0:
                        break
                    if result.returncode not in (4, 75):
                        print('MODEL_OR_COMMAND_FAILURE', flush=True)
                        sys.exit(result.returncode)
                    time.sleep(5)
