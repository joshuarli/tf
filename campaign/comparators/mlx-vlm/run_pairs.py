import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
model = '/Users/josh/models/Inkling-2-Mini@ed40e7da625f'
heavy = '/Users/josh/repos/tensorfold-m5/campaign/heavy.py'
mlx = '/Users/josh/repos/tensorfold-m5/.work/venv-mlx-vlm/bin/python'
env = os.environ.copy()
env.update({'PATH': '/opt/homebrew/bin:' + env['PATH'], 'TMPDIR': '/Users/josh/repos/tensorfold-m5/.work/tmp', 'ZIG_GLOBAL_CACHE_DIR': '/Users/josh/repos/tensorfold-m5/.work/zig-global-cache', 'HF_HOME': '/Users/josh/repos/tensorfold-m5/.work/hf-home', 'HF_HUB_OFFLINE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'GPU_GATE_MAX_GPU_PCT': '35', 'GPU_GATE_MAX_UTIL_PCT': '35'})
sequence = [(512, 1, 'mlx'), (512, 1, 'native'), (512, 2, 'native'), (512, 2, 'mlx'), (512, 3, 'mlx'), (512, 3, 'native'), (1024, 1, 'native'), (1024, 1, 'mlx'), (1024, 2, 'mlx'), (1024, 2, 'native'), (1024, 3, 'native'), (1024, 3, 'mlx')]
receipt = root / 'campaign/runs/inkling-mlxvlm-pairs.jsonl'
critical = 0
for size, pair, engine in sequence:
    prompt = f'campaign/prompts/inkling/code-{size}.json'
    run = [mlx, 'campaign/comparators/mlx-vlm/compare.py', model, prompt, '--steps', '16'] if engine == 'mlx' else ['campaign/comparators/mlx-vlm/build/bin/native-compare', model, prompt, '16']
    shell = 'perf-gate-ready || exit 75; gpu-gate-ready -1 || exit 75; ' + shlex.join(run) + '; result=$?; perf-gate-ready -1 || exit 75; GPU_GATE_IGNORE_PIDS=$$ gpu-gate-ready -1 || exit 75; exit $result'
    command = ['python3', heavy, '--', 'sh', '-c', shell]
    attempt = 0
    while True:
        while subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip() != '1':
            time.sleep(5)
        attempt += 1
        print(f'start {size} pair={pair} engine={engine} attempt={attempt}', flush=True)
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        record = {'prompt_tokens': size, 'pair': pair, 'engine': engine, 'attempt': attempt, 'command': shlex.join(command), 'exit': result.returncode, 'output': result.stdout}
        with receipt.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(f'end {size} pair={pair} engine={engine} exit={result.returncode}\n{result.stdout}', flush=True)
        if result.returncode == 3:
            print('STOPPED_ON: a run exceeds heavy.py\'s 28 GB cap', flush=True)
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
            print('UNEXPECTED_FAILURE', flush=True)
            sys.exit(result.returncode)
