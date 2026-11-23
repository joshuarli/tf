import argparse
import concurrent.futures
import json
import importlib.util
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PREFIX = re.compile(r'inkling prefix: reused (\d+) of (\d+) prompt tokens')
DONE = re.compile(r'done (\S+) .*?tokens=(\d+).*?ttft=([\d.]+).*?elapsed=([\d.]+) queued=([\d.]+)')
CONTEXT = '\n'.join(f'def transform_{i}(value): return (value * {i + 1} + {i}) % 997' for i in range(115))


def parse_log(log):
    return {'prefix': [{'reused': int(a), 'prefilled': int(b) - int(a)} for a, b in PREFIX.findall(log)],
            'done': [{'id': a, 'tokens': int(b), 'ttft': float(c), 'elapsed': float(d), 'queued': float(e)}
                     for a, b, c, d, e in DONE.findall(log)]}


def aggregate(agents, wall):
    counts = [sum(turn['tokens'] for turn in agent['turns']) for agent in agents]
    return {'wall_seconds': wall, 'aggregate_tokens_per_second': sum(counts) / wall,
            'agent_tokens_per_second': [n / agent['seconds'] for n, agent in zip(counts, agents)],
            'ttft_seconds': [turn['ttft'] for agent in agents for turn in agent['turns']],
            'output_tokens': sum(counts),
            'cached_tokens': [turn['cached_tokens'] for agent in agents for turn in agent['turns']],
            'request_ids': [turn['id'] for agent in agents for turn in agent['turns']]}


def agent(base, model, turns, barrier, agent_id):
    messages = [{'role': 'system', 'content': 'Review this code context:\n' + CONTEXT},
                {'role': 'user', 'content': f'Workspace {agent_id}: what does transform_0(42) return? Explain in one sentence.'}]
    records = []
    barrier.wait()
    started = time.monotonic()
    for _ in range(turns):
        body = {'model': model, 'messages': messages, 'max_tokens': 256, 'temperature': 0,
                'stream': True, 'stream_options': {'include_usage': True}, 'reasoning_effort': 'minimal'}
        req = urllib.request.Request(base + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        sent = time.monotonic()
        first = None
        content = []
        reasoning = []
        tokens = None
        finished = False
        cached_tokens = None
        request_id = None
        with urllib.request.urlopen(req, timeout=800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                data = line[5:].strip()
                if data == '[DONE]':
                    finished = True
                    break
                event = json.loads(data)
                request_id = event.get('id', request_id)
                if event.get('usage'):
                    tokens = event['usage']['completion_tokens']
                    cached_tokens = (event['usage'].get('prompt_tokens_details') or {}).get('cached_tokens')
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    piece = delta.get('content') or delta.get('reasoning_content') or ''
                    if piece and first is None:
                        first = time.monotonic()
                    content.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning_content') or '')
        answer = ''.join(content)
        if not finished or tokens is None or first is None or not answer:
            raise ValueError(f'incomplete streamed response: done={finished}, tokens={tokens}, '
                             f'first_output={first is not None}, answer_chars={len(answer)}')
        records.append({'tokens': tokens, 'ttft': first - sent, 'cached_tokens': cached_tokens, 'id': request_id})
        reply = {'role': 'assistant', 'content': answer}
        if ''.join(reasoning):
            reply = {'role': 'assistant', 'reasoning_content': ''.join(reasoning), 'content': answer}
        messages.extend([reply,
                         {'role': 'user', 'content': 'What does transform_1(42) return? Explain in one sentence.'}])
    return {'turns': records, 'seconds': time.monotonic() - started}


def memory_sampler(pid, wired_start, stop, samples):
    spec = importlib.util.spec_from_file_location('heavy', Path(__file__).resolve().parents[1] / 'heavy.py')
    heavy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(heavy)
    while not stop.is_set():
        members = heavy.tree(pid)
        samples.append(heavy.charge([heavy.usage(member) for member in members],
                                    max(0, heavy.wired_bytes() - wired_start)))
        stop.wait(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--turns', type=int, default=4)
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--wired-start', type=int, required=True)
    args = parser.parse_args()
    for repeat in range(2):
        for n in (1, 2, 4, 8):
            subprocess.run(["perf-gate-ready"], check=True)
            offset = args.log.stat().st_size
            barrier = threading.Barrier(n + 1)
            stop = threading.Event()
            samples = []
            sampler = threading.Thread(target=memory_sampler, args=(args.server_pid, args.wired_start, stop, samples))
            sampler.start()
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                    tasks = [pool.submit(agent, args.base.rstrip('/'), args.model, args.turns, barrier, i) for i in range(n)]
                    barrier.wait()
                    started = time.monotonic()
                    agents = [task.result() for task in tasks]
                    wall = time.monotonic() - started
                    subprocess.run(["perf-gate-ready", "-1"], check=True)
            finally:
                stop.set()
                sampler.join()
            if not samples:
                raise ValueError('no memory samples')
            with args.log.open('rb') as log:
                log.seek(offset)
                events = parse_log(log.read().decode(errors='replace'))
            if len(events['prefix']) != n * args.turns or len(events['done']) != n * args.turns:
                raise ValueError(f'incomplete server log for {n}: {events}')
            print(json.dumps({'n': n, 'repeat': repeat + 1, 'peak_charged_bytes': max(samples),
                              **aggregate(agents, wall), **events}), flush=True)


if __name__ == '__main__':
    main()
