import io
import json
import threading
import unittest
from unittest.mock import patch

from bench_subagents import agent, aggregate


class AggregateTests(unittest.TestCase):
    def test_reports_each_turn_cache_and_output(self):
        agents = [
            {'seconds': 2, 'turns': [{'tokens': 4, 'ttft': 0.5, 'cached_tokens': 12},
                                     {'tokens': 6, 'ttft': 0.2, 'cached_tokens': None}]},
            {'seconds': 4, 'turns': [{'tokens': 8, 'ttft': 0.3, 'cached_tokens': 0}]},
        ]
        result = aggregate(agents, 4)
        self.assertEqual(result['output_tokens'], 18)
        self.assertEqual(result['aggregate_tokens_per_second'], 4.5)
        self.assertEqual(result['agent_tokens_per_second'], [5, 2])
        self.assertEqual(result['cached_tokens'], [12, None, 0])


class FollowupTests(unittest.TestCase):
    def test_followup_retains_the_previous_reply_and_its_reasoning(self):
        bodies = []

        def reply(request, timeout):
            bodies.append(json.loads(request.data))
            return io.BytesIO(b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n'
                              b'data: {"choices":[{"delta":{"content":"result"}}]}\n'
                              b'data: {"choices":[],"usage":{"completion_tokens":3}}\n'
                              b'data: [DONE]\n')

        with patch('bench_subagents.urllib.request.urlopen', side_effect=reply):
            agent('http://example/v1', 'model', 2, threading.Barrier(1), 3)
        self.assertNotIn('chat_template_kwargs', bodies[0])
        self.assertEqual(bodies[1]['messages'][2],
                         {'role': 'assistant', 'reasoning_content': 'think', 'content': 'result'})


if __name__ == '__main__':
    unittest.main()
