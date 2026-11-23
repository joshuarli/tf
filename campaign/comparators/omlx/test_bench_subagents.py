import unittest

from bench_subagents import aggregate


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


if __name__ == '__main__':
    unittest.main()
