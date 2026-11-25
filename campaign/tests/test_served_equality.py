"""The served equality check fails when any output evidence is changed or missing, and passes the recorded sessions."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent.parent
RUNS = CAMPAIGN / "kernels" / "inkling-sv2-served-runs"
SESSION = CAMPAIGN / "kernels" / "inkling-sc2-scorecard-runs" / "served" / "qualify-a"

spec = importlib.util.spec_from_file_location("sv2_run", RUNS / "run.py")
sv2 = importlib.util.module_from_spec(spec)
sys.modules["sv2_run"] = sv2
spec.loader.exec_module(sv2)


def rows(path: Path, kind: str) -> list[dict]:
    return [r for line in path.read_text().splitlines() if (r := json.loads(line)).get("kind") == kind]


class ServedEquality(unittest.TestCase):
    trials = rows(SESSION.with_suffix(".jsonl"), "trial")
    replies = rows(SESSION.parent / "qualify-a.checks.jsonl", "reply")

    def mismatches(self, trials: list[dict], replies: list[dict]) -> list[str]:
        return [p for p in sv2.differences(trials, replies, "s") if ": invalid: " not in p]

    def spec_cold(self, trials: list[dict]) -> dict:
        return next(t for t in trials if t["mode"] == "spec" and t["valid"])["requests"]["cold"]

    def test_the_recorded_session_is_equal(self):
        self.assertEqual(self.mismatches(self.trials, self.replies), [])

    def test_a_changed_or_missing_stream_field_fails(self):
        for field, value in (("token_sha256", "0" * 64), ("output_sha256", "0" * 64), ("completion_tokens", 511),
                             ("finish_reasons", ["stop"]), ("tool_sha256", "0" * 64), ("token_sha256", None),
                             ("output_sha256", None), ("complete", False)):
            with self.subTest(field=field, value=value):
                trials = copy.deepcopy(self.trials)
                self.spec_cold(trials)[field] = value
                self.assertTrue(self.mismatches(trials, self.replies))

    def test_the_same_status_with_different_output_fails(self):
        trials = copy.deepcopy(self.trials)
        t = next(t for t in trials if t["mode"] == "spec" and t["valid"])
        for phase in ("cold", "warm"):
            t["requests"][phase]["token_sha256"] = "1" * 64
        t["natural_termination"]["output_sha256"] = "2" * 64
        self.assertEqual(t["requests"]["cold"]["completion"], "truncated")
        found = self.mismatches(trials, self.replies)
        self.assertTrue(any(" cold: " in p for p in found) and any(" warm: " in p for p in found)
                        and any(" natural: " in p for p in found))

    def test_a_changed_reply_token_fails(self):
        replies = copy.deepcopy(self.replies)
        r = next(r for r in replies if r["mode"] == "spec" and r["phase"] == "fork")
        r["token_ids"] = r["token_ids"][:-1] + [r["token_ids"][-1] + 1]
        self.assertTrue(self.mismatches(self.trials, replies))

    def test_a_trial_without_its_ar_reference_fails(self):
        trials = [t for t in self.trials if not (t["mode"] == "ar" and t["prompt"]["name"] == "qualify-code-1024")]
        self.assertTrue(any("no ar trial" in p for p in self.mismatches(trials, self.replies)))


if __name__ == "__main__":
    unittest.main()
