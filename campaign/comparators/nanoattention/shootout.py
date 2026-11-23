import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time

import numpy as np

from inkling import OPTIMIZED, harness, output_agreement, parse_result, workload
from prompt_ids import prompt_ids

MAIN = Path(subprocess.run(['git', '-C', str(Path(__file__).resolve().parent), 'rev-parse', '--path-format=absolute', '--git-common-dir'], check=True, capture_output=True, text=True).stdout.strip()).parent

ROOT = Path(__file__).resolve().parents[3]
TEMP = ROOT / '.work/tmp/n5-shootout'
MODEL = Path.home() / 'models/Inkling-2-Mini@ed40e7da625f'
NANO = MAIN / '.work/tmp/nanoattention-lane/nanoattention-6f782a8dace846e4e64a5a12cbdf5c0252166199/zig-out/bin/nano'
TF = ROOT / '.work/tmp/native-build/native-compare'
HEAVY = MAIN / 'campaign/heavy.py'
RESULTS = ROOT / 'campaign/comparators/nanoattention/runs/shootout-20261122-gated.jsonl'
ORACLE = MAIN / '.work/parity/inkling2-mini/ref'
KNOBS = ['--knob', 'batch=128'] + [item for key, value in OPTIMIZED.items() for item in ('--knob', f'{key}={value}')]
ENV = dict(os.environ, PATH='/opt/homebrew/bin:' + os.environ['PATH'], INKLING_LOAD_LIMIT_GB='26',
           GPU_GATE_MAX_GPU_PCT='60', GPU_GATE_MAX_UTIL_PCT='60', TMPDIR=str(TEMP),
           XDG_CACHE_HOME=str(TEMP / 'xdg'), PYTHONDONTWRITEBYTECODE='1')


def record(item):
    with RESULTS.open('a') as f:
        f.write(json.dumps(item, sort_keys=True) + '\n')
        f.flush()


def clean_diagnostics(text):
    return '\n'.join('[unrelated process details removed]' if ' command=' in line else line
                     for line in text.splitlines())


def prelaunch_memory_failure(item):
    return (item.get('exit') in (3, 4) and not item.get('worker')
            and 'host wired growth' in item.get('output', '')
            and 'footprint 0.0 GB' in item.get('output', ''))


def worker(path):
    command = json.loads(Path(path).read_text())
    own = {os.getpid()}
    reasons = []
    before_perf = subprocess.run(['perf-gate-ready'], text=True, capture_output=True)
    before = harness.conditions(own)
    if before['pressure_reasons']:
        reasons.append('pre-run host conditions failed')
    if before['pressure']['gpu_gate'].get('status') != 'quiet':
        reasons.append('pre-run GPU gate did not report quiet')
    before['pressure_reasons'] = []
    before['pressure']['gpu_gate'].pop('detail', None)
    if before_perf.returncode:
        reasons.append('pre-run performance gate failed')
    if reasons:
        print(json.dumps({'kind': 'worker', 'before': before, 'reasons': reasons,
                          'perf_before': clean_diagnostics(before_perf.stdout + before_perf.stderr)}), flush=True)
        return 75
    gpu_before = harness.gpu_time_by_pid(strict=True)
    started = time.monotonic()
    child = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    own.add(child.pid)
    stdout, stderr = child.communicate()
    after_perf = subprocess.run(['perf-gate-ready', '-1'], text=True, capture_output=True)
    elapsed = time.monotonic() - started
    gpu_after = harness.gpu_time_by_pid(strict=True)
    after = harness.conditions(own)
    users, contention = harness.gpu_contention(gpu_before, gpu_after, elapsed, own)
    if after['pressure_reasons']:
        reasons.append('post-run host conditions failed')
    if after['pressure']['gpu_gate'].get('status') != 'quiet':
        reasons.append('post-run GPU gate did not report quiet')
    after['pressure_reasons'] = []
    after['pressure']['gpu_gate'].pop('detail', None)
    if contention:
        reasons.append('other GPU clients exceeded the 60 percent limit')
    if not gpu_before or not gpu_after:
        reasons.append('GPU client sample was empty')
    if (set(gpu_before) - own) != (set(gpu_after) - own):
        reasons.append('GPU client set changed during the model process')
    if any(gpu_after[pid][1] < gpu_before[pid][1] for pid in set(gpu_before) & set(gpu_after) - own):
        reasons.append('GPU time counter reset')
    if after_perf.returncode:
        reasons.append('post-run performance gate failed')
    print(json.dumps({'kind': 'worker', 'before': before, 'after': after,
                      'other_gpu_seconds': sum(user['gpu_s'] for user in users), 'other_gpu_clients': len(users),
                      'contention_window': 'whole engine process, including load and warmup',
                      'reasons': reasons, 'stdout': stdout, 'stderr': stderr, 'child_exit': child.returncode,
                      'perf_before': clean_diagnostics(before_perf.stdout + before_perf.stderr),
                      'perf_after': clean_diagnostics(after_perf.stdout + after_perf.stderr)}), flush=True)
    return child.returncode if child.returncode else 75 if reasons else 0


def gated(command, label):
    job = TEMP / 'job.json'
    job.write_text(json.dumps(command))
    outer = ['python3', str(HEAVY), '--', sys.executable, str(Path(__file__).resolve()), 'worker', str(job)]
    busy_start = time.monotonic()
    cap_retry = RESULTS.exists() and any(
        entry['label'] == label and prelaunch_memory_failure(entry)
        for entry in (json.loads(line) for line in RESULTS.read_text().splitlines())
    )
    while True:
        print(f'{label}: starting', flush=True)
        result = subprocess.run(outer, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = '\n'.join('[unrelated process details removed]' if ' command=' in line and not line.startswith('{"kind": "worker"') else line for line in result.stdout.splitlines())
        worker_rows = [json.loads(line) for line in output.splitlines() if line.startswith('{"kind": "worker"')]
        worker_row = worker_rows[0] if len(worker_rows) == 1 else {}
        pressure = re.search(r'host pressure worst (\d+)', output)
        status = result.returncode
        if status == 0 and (pressure is None or int(pressure.group(1)) != 1):
            status = 76
        item = {'label': label, 'command': shlex.join(outer), 'engine_command': shlex.join(command),
                'exit': status, 'output': output, 'worker': worker_row}
        record(item)
        print(f'{label}: exit={status}', flush=True)
        if prelaunch_memory_failure(item) and not cap_retry:
            cap_retry = True
            time.sleep(10)
            continue
        if status not in (75, 76):
            return item
        if time.monotonic() - busy_start > 10800:
            raise TimeoutError('host busy more than 3 hours of retries')
        time.sleep(10)


def prepare():
    TEMP.mkdir(parents=True, exist_ok=True)
    for size in (1024, 4096, 16384):
        for topic in ('code', 'prose', 'structured'):
            source_size = 32768 if size == 16384 else size
            source = ROOT / f'campaign/prompts/dev/{topic}-{source_size}.txt'
            ids = prompt_ids(MODEL, source.read_text())
            if size == 16384:
                ids = ids[:16380] + ids[-4:]
            if len(ids) > 32256:
                raise ValueError('a prompt beyond the 32K cap')
            (TEMP / f'{topic}-{size}.json').write_text(json.dumps(ids))
            print(topic, size, len(ids), hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(), flush=True)


def timings():
    failures = []
    entries = [json.loads(line) for line in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    completed = {entry['label']: entry['result'] for entry in entries if entry['label'].endswith('/parsed')}
    comparisons = {entry['label'] for entry in entries if entry['label'].endswith('/comparison')}
    failed_cells = {entry['label'].split('/')[0] for entry in entries
                    if 'exit' in entry and entry['exit'] not in (0, 75, 76) and not prelaunch_memory_failure(entry)}
    cap_failures = {}
    for entry in entries:
        if prelaunch_memory_failure(entry):
            cap_failures[entry['label']] = cap_failures.get(entry['label'], 0) + 1
    failed_cells.update(label.split('/')[0] for label, count in cap_failures.items() if count >= 2)
    for size in (1024, 4096, 16384):
        for topic in ('code', 'prose', 'structured'):
            name = f'{topic}-{size}'
            if name in failed_cells:
                failures.append((name, 'previous model failure'))
                continue
            path = TEMP / f'{name}.json'
            ids = json.loads(path.read_text())
            token_file = TEMP / f'{name}-nano.json'
            token_file.write_text(json.dumps(workload(ids, 512)))
            for pair in range(1, 4):
                rows = {}
                for engine in (('tf', 'nano') if pair % 2 else ('nano', 'tf')):
                    parsed_label = f'{name}/{pair}/{engine}/parsed'
                    if parsed_label in completed:
                        rows[engine] = completed[parsed_label]
                        continue
                    if engine == 'tf':
                        command = [str(TF), str(MODEL), str(path), '512', '1']
                    else:
                        command = [str(NANO), 'bench', str(MODEL), '--context', str(len(ids) + 512), '--prompt-tokens', str(len(ids)), '--decode-tokens', '511', '--decode-mode', 'greedy', '--bench-tokens', str(token_file), '--repeats', '1'] + KNOBS
                    row = gated(command, f'{name}/{pair}/{engine}')
                    if row['exit']:
                        failures.append((name, pair, engine, row['exit']))
                        break
                    rows[engine] = parse_result(row['worker']['stdout'], row['worker']['stderr'], engine, len(ids), 512)
                    record({'label': parsed_label, 'result': rows[engine]})
                if len(rows) == 2:
                    label = f'{name}/{pair}/comparison'
                    if label not in comparisons:
                        record({'label': label, 'prompt_tokens': len(ids), 'agreement': output_agreement(rows['tf']['predictions'], rows['nano']['predictions'])})
                else:
                    break
    return failures


def fidelity_row(reference, candidate):
    ref = np.asarray(reference, dtype=np.float64).copy()
    got = np.asarray(candidate, dtype=np.float64).copy()
    if ref.shape != got.shape or ref.ndim != 1 or ref.size < 2:
        raise ValueError('logit rows must have equal one-dimensional shapes')
    if not np.isfinite(ref).all() or not np.isfinite(got).all():
        raise ValueError('nonfinite logits')
    best = np.argpartition(ref, -2)[-2:]
    confident = float(ref[best].max() - ref[best].min()) >= 0.1
    agree = ref.argmax() == got.argmax()
    ref -= ref.max()
    got -= got.max()
    log_ref = ref - np.log(np.exp(ref).sum())
    log_got = got - np.log(np.exp(got).sum())
    return confident, bool(agree), float(np.sum(np.exp(log_ref) * (log_ref - log_got)))


def fidelity():
    histories = json.loads((ROOT / 'campaign/parity/inkling2-mini/histories.json').read_text())
    total_kl = 0.0
    total_rows = confident = agree = 0
    for history in histories:
        name, ids = history['name'], history['token_ids']
        token_file = TEMP / f'{name}-history.json'
        output = TEMP / f'{name}-logits.f32'
        token_file.write_text(json.dumps(ids))
        command = [str(NANO), 'eval', str(MODEL), '--context', str(max(128, len(ids))), '--tokens-file', str(token_file), '--feed', '128', '--positions', 'all', '--logits-out', str(output)] + KNOBS
        row = gated(command, f'oracle/{name}')
        if row['exit']:
            return row['exit']
        reference = np.load(ORACLE / f'{name}.npy', mmap_mode='r')
        if reference.shape != (len(ids), 200058):
            raise ValueError(f'unexpected oracle shape: {name} {reference.shape}')
        if output.stat().st_size != len(ids) * 200058 * 4:
            raise ValueError(f'unexpected candidate logits size: {name} {output.stat().st_size}')
        candidate = np.memmap(output, dtype='<f4', mode='r', shape=reference.shape)
        for i in range(len(ids)):
            is_confident, is_agree, kl = fidelity_row(reference[i], candidate[i])
            if is_confident:
                confident += 1
                agree += int(is_agree)
            total_kl += kl
            total_rows += 1
        record({'label': f'oracle/{name}/compared', 'positions': len(ids)})
        del candidate
        output.unlink()
    record({'label': 'oracle/summary', 'histories': len(histories), 'positions': total_rows, 'confident': confident, 'agree': agree, 'confident_top1': agree / confident, 'mean_kl': total_kl / total_rows})
    return 0


def summary():
    entries = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    for size in (1024, 4096, 16384):
        for topic in ('code', 'prose', 'structured'):
            name = f'{topic}-{size}'
            comparisons = [x for x in entries if re.fullmatch(rf'{name}/\d+/comparison', x['label'])]
            print(name, 'differences', [x['agreement']['first_different_generated_index'] for x in comparisons])
            for engine in ('tf', 'nano'):
                parsed = [x['result'] for x in entries if re.fullmatch(rf'{name}/\d+/{engine}/parsed', x['label'])]
                runs = [x for x in entries if re.fullmatch(rf'{name}/\d+/{engine}', x['label']) and x['exit'] == 0
                        and 'stdout' in x.get('worker', {})]
                if len(parsed) != 3 or len(runs) != 3:
                    print(engine, 'incomplete', len(parsed), len(runs))
                    continue
                def metric(key):
                    values = [r[key] for r in parsed]
                    return f'{statistics.median(values):.3f} ({min(values):.3f}–{max(values):.3f})'
                peaks = [float(re.search(r'peak charged ([\d.]+) GB', run['output']).group(1)) for run in runs]
                print(engine, 'ttft', metric('prefill_seconds'), 'prefill', metric('prefill_tokens_per_second'),
                      'decode', metric('decode_tokens_per_second'), 'peak GB', max(peaks),
                      'outputs stable', all(r['predictions'] == parsed[0]['predictions'] for r in parsed))
    for entry in entries:
        if entry['label'] == 'oracle/summary':
            print(json.dumps(entry, sort_keys=True))


if __name__ == '__main__':
    TEMP.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == 'prepare':
        prepare()
    elif sys.argv[1] == 'timings':
        failures = timings()
        print(f'failures={failures}', flush=True)
        sys.exit(bool(failures))
    elif sys.argv[1] == 'fidelity':
        sys.exit(fidelity())
    elif sys.argv[1] == 'summary':
        summary()
    elif sys.argv[1] == 'worker':
        sys.exit(worker(sys.argv[2]))
