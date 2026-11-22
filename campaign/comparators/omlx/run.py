import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from adapter import mtp_cycles, parse_result, read_prompt, validate_model_path
from setup import model_entries, source_entry

CAMPAIGN = Path(__file__).resolve().parents[2]
ROOT = Path('/Users/josh/repos/tensorfold-m5')
WORK = ROOT / '.work'
MODEL = Path.home() / 'models/Inkling-2-Mini-oQ4e-mtp@3eea042751cd'
HEAVY = ROOT / 'campaign/heavy.py'
PYTHON = WORK / 'venv-omlx/bin/python'


async def worker(model, ids, steps, mtp):
    validate_model_path(model)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('omlx.patches.mlx_lm_mtp.batch_generator').setLevel(logging.INFO)
    import mlx.core as mx
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.engine_core import get_mlx_executor
    from omlx.model_settings import ModelSettings
    from omlx.patches.inkling2.dspark import load_inkling2_dspark_drafter
    from omlx.request import SamplingParams
    from omlx.scheduler import SchedulerConfig

    settings = ModelSettings(mtp_enabled=mtp)
    engine = VLMBatchedEngine(str(model), model_settings=settings, scheduler_config=SchedulerConfig())
    await engine.start()
    try:
        draft = None
        if mtp:
            target = engine.prepare_dflash_target()
            drafter = await asyncio.get_running_loop().run_in_executor(
                get_mlx_executor(), load_inkling2_dspark_drafter, str(model), target, None, False)
            if drafter is None:
                raise RuntimeError('Lightning MTP drafter did not load')
            engine.set_dflash_drafter(drafter)
            if engine.dflash_drafter is None:
                raise RuntimeError('Lightning MTP drafter was not attached')
            draft = {'block_size': drafter.block_size, 'maximum_depth': drafter.depth,
                     'adaptive': drafter.adaptive}
        mx.reset_peak_memory()
        start = time.perf_counter()
        request = await engine._engine.add_request(
            prompt=ids, sampling_params=SamplingParams(max_tokens=steps, temperature=0, ignore_eos=True),
            skip_cache_store=True)
        first = last = None
        generated = []
        async for output in engine._engine.stream_outputs(request):
            if output.error:
                raise RuntimeError(output.error)
            if output.new_token_ids:
                if first is None:
                    first = output.generated_at
                last = output.generated_until
                generated.extend(output.new_token_ids)
        if first is None or last is None or len(generated) != steps:
            raise RuntimeError(f'incomplete generation: {len(generated)} tokens')
        row = {'type': 'omlx_comparator', 'prompt_tokens': len(ids), 'generated_ids': generated,
               'mtp': mtp, 'temperature': 0, 'ttft_seconds': first - start,
               'decode_seconds': last - first, 'cache_quantization': settings.turboquant_kv_enabled,
               'draft': draft, 'mlx_peak_bytes': mx.get_peak_memory()}
        print(json.dumps(row), flush=True)
    finally:
        await engine.stop()


def attempt(model, prompt, steps, mtp):
    command = [str(PYTHON), str(Path(__file__).resolve()), '--worker', '--model', str(model),
               '--prompt', str(prompt), '--steps', str(steps), '--mtp', 'on' if mtp else 'off']
    script = ('gpu-gate-ready -1 || exit 75; perf-gate-ready || exit 75; ' + shlex.join(command)
              + '; result=$?; perf-gate-ready -1 || exit 75; exit "$result"')
    outer = ['python3', str(HEAVY), '--cap-gb', '26', '--', 'sh', '-c', script]
    deadline = time.monotonic() + 7200
    history = []
    while True:
        result = subprocess.run(outer, env=os.environ.copy(), text=True, capture_output=True)
        stderr = '\n'.join('[unrelated process details removed]' if ' command=' in line else line
                           for line in result.stderr.splitlines())
        row = {'command': shlex.join(outer), 'exit': result.returncode,
               'stdout': result.stdout, 'stderr': stderr}
        history.append(row)
        if result.returncode != 75:
            return row | {'attempts': history}
        if time.monotonic() >= deadline:
            print(json.dumps({'attempts': history}), flush=True)
            raise RuntimeError('host busy more than 2 hours of retries')
        print('host busy; retrying admission', flush=True)
        time.sleep(15)


def run(args):
    validate_model_path(args.model)
    entries = json.loads((CAMPAIGN / 'sources.lock.json').read_text())['inputs']
    source_entry(entries)
    model_entries(entries)
    prompt = Path(args.prompt)
    if not prompt.is_file():
        prompt = CAMPAIGN / 'prompts/inkling' / f'{args.prompt}.json'
    ids = read_prompt(prompt, args.steps)
    rows = []
    output = CAMPAIGN / 'comparators/omlx/results' / f'{prompt.stem}-{time.time_ns()}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    for pair in range(args.pairs):
        modes = (False, True) if args.mode == 'both' else (args.mode == 'on',)
        if pair % 2:
            modes = tuple(reversed(modes))
        for mtp in modes:
            print(f'Running {prompt.stem} pair {pair + 1} Lightning MTP {"on" if mtp else "off"}', flush=True)
            row = attempt(args.model, prompt, args.steps, mtp)
            row['pair'] = pair + 1
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + '\n')
            if row['exit'] != 0:
                raise RuntimeError(f'engine failed; receipt: {output}')
            row['result'] = parse_result(row['stdout'], len(ids), args.steps)
            if row['result'].get('mtp') is not mtp:
                raise ValueError('oMLX returned the wrong speculation mode')
            if mtp:
                row['mtp_cycles'] = mtp_cycles(row['stderr'])
            match = re.search(r'peak charged ([\d.]+) GB', row['stderr'])
            if not match:
                raise ValueError('heavy.py did not report peak charged memory')
            row['peak_charged_gb'] = float(match.group(1))
            row['prompt_ids_sha256'] = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
            output.write_text(json.dumps(rows, indent=2) + '\n')
    if args.mode == 'both':
        comparisons = []
        for pair in range(1, args.pairs + 1):
            pair_rows = [row for row in rows if row['pair'] == pair]
            off, on = (next(row['result']['generated_ids'] for row in pair_rows if row['result']['mtp'] is mode)
                       for mode in (False, True))
            first = next((index for index, (a, b) in enumerate(zip(off, on)) if a != b), None)
            comparisons.append({'pair': pair, 'identical': first is None, 'first_difference': first})
        output.write_text(json.dumps({'trials': rows, 'comparisons': comparisons}, indent=2) + '\n')
        print(json.dumps(comparisons), flush=True)
    print(f'Receipt: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--prompt', default='code-1024')
    parser.add_argument('--steps', type=int, default=512)
    parser.add_argument('--mtp', choices=('on', 'off'), default='off')
    parser.add_argument('--mode', choices=('on', 'off', 'both'), default='both')
    parser.add_argument('--pairs', type=int, default=1)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error('pairs must be positive')
    if args.worker:
        asyncio.run(worker(args.model, read_prompt(args.prompt, args.steps), args.steps, args.mtp == 'on'))
    else:
        run(args)


if __name__ == '__main__':
    main()
