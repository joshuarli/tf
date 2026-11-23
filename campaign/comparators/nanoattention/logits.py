import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time

MAIN = Path(subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', '--path-format=absolute', '--git-common-dir'], check=True, capture_output=True, text=True).stdout.strip()).parent

work = MAIN / '.work/tmp/nanoattention-lane'
model = '/Users/josh/models/Inkling-2-Mini@ed40e7da625f'
heavy = str(MAIN / 'campaign/heavy.py')
nano = str(work / 'nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199/zig-out/bin/nano')
tf = str(work / 'tf-build/bin/tf-inkling2-loop-run')
receipt = Path('campaign/comparators/nanoattention/runs/logits.jsonl')
env = os.environ.copy()
env.update({'PATH': '/opt/homebrew/bin:' + str(Path.home() / '.elan/bin') + ':' + env['PATH'], 'TMPDIR': str(MAIN / '.work/tmp'), 'ZIG_GLOBAL_CACHE_DIR': str(MAIN / '.work/zig-global-cache'), 'XDG_CACHE_HOME': str(work / 'xdg-cache'), 'PYTHONDONTWRITEBYTECODE': '1', 'GPU_GATE_MAX_GPU_PCT': '35', 'GPU_GATE_MAX_UTIL_PCT': '35'})
optimized = ['batch=128', 'expert_simd=true', 'expert_wide_dot=true', 'expert_gate_pair=true', 'expert_prefill=true', 'expert_wide_k=true', 'expert_bf16_tiles=true', 'expert_mlx_scale=true', 'bf16_linear_tile=true', 'bf16_prefill_tile=8', 'bf16_prefill_matrix=true', 'bf16_decode_vec=true', 'fuse_bf16_cast=true', 'concurrent_scratch=true', 'relative_split=true', 'relative_split_4=true', 'relative_tile_prefill=true']
critical = 0
differences = []
for size in (512, 1024):
    prompt = f'campaign/prompts/inkling/code-{size}.json'
    for engine in ('tensorfold', 'reference', 'optimized'):
        output = work / f'logits-{size}-{engine}.f32'
        if engine == 'tensorfold':
            run = [tf, model, prompt, 'prefill', '512', str(output)]
        else:
            batch = '128' if engine == 'optimized' else '64'
            run = [nano, 'eval', model, '--context', str(size), '--tokens-file', prompt, '--feed', batch, '--positions', str(size - 1), '--logits-out', str(output)]
            if engine == 'optimized':
                for knob in optimized:
                    run += ['--knob', knob]
        shell = 'perf-gate-ready || exit 75; gpu-gate-ready -1 || exit 75; ' + shlex.join(run) + ' || exit $?; perf-gate-ready -1 || exit 75; GPU_GATE_IGNORE_PIDS=$$ gpu-gate-ready -1 || exit 75'
        command = ['python3', heavy, '--', 'sh', '-c', shell]
        attempt = 0
        while True:
            while subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip() != '1':
                time.sleep(5)
            attempt += 1
            print(f'start logits size={size} engine={engine} attempt={attempt}', flush=True)
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            sanitized = '\n'.join('[unrelated process details removed]' if ' command=' in line else line for line in result.stdout.splitlines()) + '\n'
            item = {'prompt_tokens': size, 'engine': engine, 'attempt': attempt, 'command': shlex.join(command), 'exit': result.returncode, 'output': sanitized}
            with receipt.open('a') as f:
                f.write(json.dumps(item) + '\n')
                f.flush()
            print(f'end logits size={size} engine={engine} exit={result.returncode}', flush=True)
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
                sys.exit(result.returncode)
            time.sleep(5)
    vocab = 200058
    rows = []
    for engine in ('tensorfold', 'reference', 'optimized'):
        p = work / f'logits-{size}-{engine}.f32'
        assert p.stat().st_size == vocab * 4 * (size if engine == 'tensorfold' else 1)
        with p.open('rb') as f:
            f.seek(-vocab * 4, os.SEEK_END)
            row = struct.unpack(f'<{vocab}f', f.read(vocab * 4))
            assert all(map(math.isfinite, row))
            rows.append(row)
    for engine, row in zip(('reference', 'optimized'), rows[1:]):
        delta = max(abs(a - b) for a, b in zip(rows[0], row))
        difference = {'prompt_tokens': size, 'engine': engine, 'max_abs_difference': delta}
        differences.append(difference)
        print(json.dumps(difference), flush=True)
Path('campaign/comparators/nanoattention/runs/logits-summary.json').write_text(json.dumps(differences, indent=2) + '\n')
