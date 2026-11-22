import io
import json
import threading
import unittest
from unittest.mock import patch

from bench_subagents import agent, aggregate, parse_log


class BenchmarkTests(unittest.TestCase):
    def test_log_records_prefix_and_queue(self):
        log = ('inkling prefix: reused 12 of 42 prompt tokens\n'
               'done chat-1 prompt=42 cached=12 thinking=False effort=none tokens=8 sha=a finish=stop '
               'rounds=1 accepted=0/0 ttft=1.200 decode=2.000 elapsed=3.000 queued=0.500 tok/s=4.0\n')
        result = parse_log(log)
        self.assertEqual(result['prefix'], [{'reused': 12, 'prefilled': 30}])
        self.assertEqual(result['done'][0]['queued'], 0.5)
        self.assertEqual(result['done'][0]['tokens'], 8)

    def test_followup_retains_the_previous_reply(self):
        bodies = []

        def reply(request, timeout):
            bodies.append(json.loads(request.data))
            return io.BytesIO(b'data: {"choices":[{"delta":{"content":"result"}}]}\n'
                              b'data: {"choices":[],"usage":{"completion_tokens":3}}\n'
                              b'data: [DONE]\n')

        with patch('bench_subagents.urllib.request.urlopen', side_effect=reply):
            result = agent('http://example/v1', 'model', 2, threading.Barrier(1), 3)
        self.assertEqual([turn['tokens'] for turn in result['turns']], [3, 3])
        self.assertEqual(bodies[1]['messages'][:2], bodies[0]['messages'])
        self.assertEqual(bodies[1]['messages'][2:], [
            {'role': 'assistant', 'content': 'result'},
            {'role': 'user', 'content': 'What does transform_1(42) return? Explain in one sentence.'},
        ])

    def test_hidden_output_does_not_count_as_a_completed_turn(self):
        reply = io.BytesIO(b'data: {"choices":[],"usage":{"completion_tokens":256}}\n'
                           b'data: [DONE]\n')
        with patch('bench_subagents.urllib.request.urlopen', return_value=reply):
            with self.assertRaisesRegex(ValueError, 'first_output=False, answer_chars=0'):
                agent('http://example/v1', 'model', 1, threading.Barrier(1), 0)

    def test_aggregate_counts_all_committed_output(self):
        agents = [{'seconds': 4, 'turns': [{'tokens': 8, 'ttft': 1}]},
                  {'seconds': 2, 'turns': [{'tokens': 4, 'ttft': 2}]}]
        result = aggregate(agents, 6)
        self.assertEqual(result['aggregate_tokens_per_second'], 2)
        self.assertEqual(result['agent_tokens_per_second'], [2, 2])
        self.assertEqual(result['ttft_seconds'], [1, 2])


if __name__ == '__main__':
    unittest.main()
