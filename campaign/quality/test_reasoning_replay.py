"""CPU checks for the reasoning-replay experiment: its tasks, both arms' histories, scoring and statistics."""

import io
import json
import re
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import reasoning_replay as rr  # noqa: E402


def reply(content: str, reasoning: str = "", finish: str = "stop") -> dict:
    return {"content": content, "reasoning": reasoning, "finish_reason": finish, "prompt_tokens": 10,
            "completion_tokens": 5, "reasoning_tokens": 2 if reasoning else 0, "cached_tokens": 0}


def fenced(source: str) -> str:
    return "Here it is.\n```python\n" + source + "```\n"


def quiet(argv: list) -> int:
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = io.StringIO()
    try:
        return rr.main(argv)
    finally:
        sys.stdout, sys.stderr = stdout, stderr


def guarded(observe, namespace, case):
    try:
        return observe(namespace, case)
    except BaseException as exc:
        return ("crash", type(exc).__name__)


class Tasks(unittest.TestCase):
    def test_transform_matches_the_workload_context(self):
        namespace = {}
        exec(rr.CONTEXT, namespace)
        for index in (0, 1, 57, 114):
            for value in (0, 1, 500, 996):
                self.assertEqual(namespace[f"transform_{index}"](value), rr.transform(index, value))
        self.assertNotIn(f"transform_{rr.TRANSFORMS}", namespace)

    def test_families_are_deterministic_distinct_and_five_turns_long(self):
        for family in rr.FAMILIES:
            first = rr.conversations(family)
            self.assertEqual(first, rr.conversations(family))
            self.assertEqual(len(first), 40)
            self.assertEqual(len({conversation["id"] for conversation in first}), 40)
            self.assertTrue(all(len(conversation["turns"]) == 5 for conversation in first))
            self.assertEqual(len({conversation["turns"][0]["user"] for conversation in first}), 40)
        self.assertEqual(rr.conversations("chain", 2, turns=6)[1]["turns"][:5], rr.conversations("chain", 2)[1]["turns"])
        with self.assertRaises(ValueError):
            rr.conversations("code", 41)

    def test_chain_expectations_follow_the_stated_transforms(self):
        for conversation in rr.conversations("chain", 5, turns=6):
            value = conversation["start"]
            self.assertIn(f"Start with the value {value}.", conversation["turns"][0]["user"])
            for turn in conversation["turns"]:
                (op,) = map(int, re.findall(r"transform_(\d+)", turn["user"]))
                value = rr.transform(op, value)
                self.assertEqual(turn["expect"], value)
                self.assertEqual(turn["ops"], [op])
                self.assertIn(rr.VALUE_LINE, turn["user"])

    def test_recall_asks_for_the_intermediates_of_earlier_turns(self):
        for conversation in rr.conversations("recall", 10):
            turns = conversation["turns"]
            first = [int(n) for n in re.findall(r"transform_(\d+)", turns[0]["user"])]
            second = [int(n) for n in re.findall(r"transform_(\d+)", turns[1]["user"])]
            values, value = [], conversation["start"]
            for op in first + second:
                value = rr.transform(op, value)
                values.append(value)
            self.assertEqual([turn["expect"] for turn in turns], [values[2], values[4], values[0], values[1], values[3]])
            self.assertEqual([turn["recall"] for turn in turns], [False, False, True, True, True])
            self.assertEqual(turns[2]["user"].count(f"transform_{first[0]}?"), 1)
            self.assertEqual(turns[4]["user"].count(f"transform_{second[0]}?"), 1)

    def test_every_code_turn_rejects_the_previous_turns_reference(self):
        observe = rr.observer()
        for conversation in rr.conversations("code"):
            for turn in range(2, 6):
                previous = {"__name__": "previous"}
                exec(rr.reference(conversation, turn - 1), previous)
                tests = rr.code_tests(conversation, turn)
                self.assertTrue(any(guarded(observe, previous, case) != expected for case, expected in tests),
                                (conversation["id"], turn))
            self.assertIn(conversation["turns"][0]["user"].split("`")[1].split("(")[0], rr.reference(conversation, 1))


class Sandbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.checker = rr.CodeChecker(Path(cls.directory.name), wall_s=5, cpu_s=3)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_references_pass_and_the_previous_turn_fails_in_the_sandbox(self):
        for conversation in rr.conversations("code", len(rr.PROJECTS)):
            for turn in range(1, 6):
                verdict = self.checker.check(conversation, turn, fenced(rr.reference(conversation, turn)))
                self.assertTrue(verdict["passed"], (conversation["id"], turn, verdict))
                if turn > 1:
                    verdict = self.checker.check(conversation, turn, fenced(rr.reference(conversation, turn - 1)))
                    self.assertFalse(verdict["passed"], (conversation["id"], turn))
                    self.assertLess(verdict["detail"]["passed"], verdict["detail"]["total"])

    def test_replies_that_dodge_the_tests_fail(self):
        conversation = rr.conversations("code", 1)[0]
        for content in ("I would write a slugify function.", fenced("raise SystemExit(0)\n"),
                        fenced("import os\nos._exit(0)\n"), fenced("def slugify(title):\n    return 1 / 0\n"),
                        fenced("while True:\n    pass\n"), "```python\ndef slugify(title):\n    return title"):
            self.assertFalse(self.checker.check(conversation, 1, content)["passed"], content)

    def test_a_reply_split_over_blocks_runs_as_one_program(self):
        conversation = rr.conversations("code", 1)[0]
        content = ("```python\ndef _clean(title):\n    return title.strip().lower()\n```\nand\n"
                   "```python\ndef slugify(title):\n    return _clean(title)\n```")
        self.assertTrue(self.checker.check(conversation, 1, content)["passed"])


class Histories(unittest.TestCase):
    class Scripted:
        def __init__(self):
            self.bodies = []

        def complete(self, body):
            self.bodies.append(body)
            number = sum(1 for message in body["messages"] if message["role"] == "user")
            return reply(f"Value: {number}", f"thought {number}" if number != 2 else "")

    def test_arms_share_turn_one_and_differ_only_in_replayed_reasoning(self):
        conversation = rr.conversations("chain", 1)[0]
        replay, stripped = self.Scripted(), self.Scripted()
        rr.converse(replay, "model", conversation, "minimal", "replay", 64)
        rr.converse(stripped, "model", conversation, "minimal", "stripped", 64)
        self.assertEqual(replay.bodies[0], stripped.bodies[0])
        self.assertEqual(replay.bodies[0]["messages"], [{"role": "system", "content": rr.SYSTEM},
                                                         {"role": "user", "content": conversation["turns"][0]["user"]}])
        last_replay, last_stripped = replay.bodies[-1]["messages"], stripped.bodies[-1]["messages"]
        self.assertEqual([m for m in last_replay if m["role"] != "assistant"],
                         [m for m in last_stripped if m["role"] != "assistant"])
        self.assertEqual([m for m in last_replay if m["role"] == "assistant"][:2], [
            {"role": "assistant", "reasoning_content": "thought 1", "content": "Value: 1"},
            {"role": "assistant", "content": "Value: 2"}])
        self.assertEqual([m for m in last_stripped if m["role"] == "assistant"],
                         [{"role": "assistant", "content": f"Value: {n}"} for n in range(1, 5)])
        for body in replay.bodies + stripped.bodies:
            self.assertEqual((body["temperature"], body["max_tokens"], body["reasoning_effort"], body["stream"]),
                             (0, 64, "minimal", False))

    def test_identical_bodies_are_sent_once_and_reload_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            client = rr.Client(path, "http://example/v1", log=io.StringIO())
            sent, gate = [], threading.Event()

            def send(body):
                sent.append(body)
                gate.wait(5)
                return reply("Value: 1", "r")

            client.send = send
            body = rr.request_body("model", [{"role": "user", "content": "hi"}], "minimal", 8)
            threads = [threading.Thread(target=client.complete, args=(body,)) for _ in range(4)]
            for thread in threads:
                thread.start()
            time.sleep(0.2)
            gate.set()
            for thread in threads:
                thread.join()
            self.assertEqual(len(sent), 1)
            offline = rr.Client(path)
            self.assertEqual(offline.complete(body)["content"], "Value: 1")
            with self.assertRaises(rr.Missing):
                offline.complete(rr.request_body("model", [{"role": "user", "content": "hi"}], "high", 8))


class Sending(unittest.TestCase):
    def failure(self, code: int, text: str):
        return urllib.error.HTTPError("http://example/v1/chat/completions", code, "error", {}, io.BytesIO(text.encode()))

    def test_a_context_refusal_is_a_failed_turn_and_other_errors_stop_the_run(self):
        client = rr.Client(Path("unused.jsonl"), "http://example/v1", log=io.StringIO())
        body = rr.request_body("model", [{"role": "user", "content": "hi"}], "high", 4096)
        refusal = self.failure(400, '{"error": {"code": "context_length_exceeded", "message": "too long"}}')
        with mock.patch.object(rr.urllib.request, "urlopen", side_effect=refusal):
            got = client.send(body)
        self.assertEqual((got["finish_reason"], got["content"], got["reasoning"]), ("context_length", "", ""))
        rows = rr.score_turns(rr.conversations("chain", 1)[0], [got])
        self.assertFalse(rows[0]["correct"])
        with mock.patch.object(rr.urllib.request, "urlopen", side_effect=self.failure(400, '{"error": "bad"}')):
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                client.send(body)

    def test_server_errors_are_retried(self):
        client = rr.Client(Path("unused.jsonl"), "http://example/v1", log=io.StringIO())
        body = rr.request_body("model", [{"role": "user", "content": "hi"}], "minimal", 8)
        answer = mock.MagicMock()
        answer.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "Value: 3", "reasoning_content": "r"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 4, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}},
             "tensorfold": {"reasoning_effort": "minimal"}}).encode()
        with mock.patch.object(rr.urllib.request, "urlopen", side_effect=[self.failure(503, "busy"), answer]), \
                mock.patch.object(rr.time, "sleep"):
            got = client.send(body)
        self.assertEqual((got["content"], got["reasoning_tokens"], got["reasoning_effort"]), ("Value: 3", 1, "minimal"))


class Scoring(unittest.TestCase):
    def test_values_are_read_from_the_value_line_first(self):
        expected = {"Value: 412": (412, True), "**Value:** 77": (77, True), "value = 5\nValue: 6": (6, True),
                    "The result is 42.": (42, False), "": (None, False), "Value: -3": (-3, True),
                    "transform_12 gives 345": (345, False)}
        for content, parsed in expected.items():
            self.assertEqual(rr.parse_value(content), parsed, content)

    def test_code_blocks_are_extracted_in_order(self):
        self.assertIsNone(rr.extract_code("no fence"))
        self.assertIsNone(rr.extract_code("```bash\nls\n```"))
        self.assertEqual(rr.extract_code("x\n```python\na = 1\n```\ntext\n```py\nb = 2\n```"), "a = 1\n\n\nb = 2\n")
        self.assertEqual(rr.extract_code("```\nc = 3\n```"), "c = 3\n")

    def test_chain_scoring_separates_absolute_and_step_errors(self):
        conversation = rr.conversations("chain", 1)[0]
        turns = conversation["turns"]
        wrong = (turns[0]["expect"] + 1) % rr.MODULUS
        follow = rr.transform(turns[1]["ops"][0], wrong)
        rows = rr.score_turns(conversation, [reply(f"Value: {wrong}"), reply(f"Value: {follow}"), reply("no idea")])
        self.assertEqual([row["correct"] for row in rows], [False, follow == turns[1]["expect"], False])
        self.assertEqual([row["step"] for row in rows], [False, True, False])
        self.assertEqual(rows[2]["answer"], None)

    def test_mcnemar_is_exact(self):
        self.assertEqual(rr.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(rr.mcnemar(0, 5), 0.0625)
        self.assertAlmostEqual(rr.mcnemar(1, 9), 2 * 11 / 1024)
        self.assertEqual(rr.mcnemar(3, 3), 1.0)

    def test_bootstrap_brackets_the_mean(self):
        self.assertEqual(rr.bootstrap([1.0] * 10), (1.0, 1.0, 1.0))
        mean, low, high = rr.bootstrap([0.0, 1.0] * 20)
        self.assertEqual(mean, 0.5)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


class EndToEnd(unittest.TestCase):
    """A scripted server answers the replay arm correctly and the stripped arm wrongly after turn 1."""

    def test_scoring_attributes_the_scripted_difference_to_the_stripped_arm(self):
        selected = (rr.conversations("chain", 3) + rr.conversations("recall", 3) + rr.conversations("code", 2))
        by_first = {conversation["turns"][0]["user"]: conversation for conversation in selected}

        def send(body):
            users = [m["content"] for m in body["messages"] if m["role"] == "user"]
            conversation, number = by_first[users[0]], len(users)
            replayed = any("reasoning_content" in m for m in body["messages"])
            right = number == 1 or replayed
            if conversation["family"] == "code":
                content = fenced(rr.reference(conversation, number if right else number - 1))
            else:
                expect = conversation["turns"][number - 1]["expect"]
                content = f"Value: {expect if right else (expect + 1) % rr.MODULUS}"
            reasoning = f"working on turn {number}"
            shown = sum(len(m["content"]) + len(m.get("reasoning_content", "")) for m in body["messages"])
            return {**reply(content, reasoning), "prompt_tokens": shown, "reasoning_tokens": len(reasoning),
                    "completion_tokens": len(reasoning) + 3}

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            client = rr.Client(out / "responses.jsonl", "http://example/v1", log=io.StringIO())
            client.send = send
            rr.run_effort(client, "model", selected, "minimal", 128, 4)
            (out / "runs.jsonl").write_text(json.dumps({"event": "start", "model": "model"}) + "\n")
            status = quiet(["score", "--out", str(out), "--efforts", "minimal", "--conversations", "3",
                            "--max-tokens", "128"])
            self.assertEqual(status, 1)
            status = quiet(["score", "--out", str(out), "--efforts", "minimal", "--conversations", "3",
                            "--max-tokens", "128", "--families", "chain,recall"])
            self.assertEqual(status, 0)
            summary = json.loads((out / "summary.json").read_text())["summary"]
            for name in ("chain/minimal", "recall/minimal"):
                block = summary[name]
                self.assertEqual(block["per_turn"][0]["replay"], block["per_turn"][0]["stripped"])
                self.assertEqual((block["later_turns"]["replay"], block["later_turns"]["stripped"]), (1.0, 0.0))
                self.assertEqual(block["later_turns"]["diff"], 1.0)
                self.assertEqual(block["later_turns"]["replay_better"], 3)
                self.assertEqual(block["replayed_reasoning_render"]["extra_prompt_tokens_minus_reasoning_tokens"],
                                 [0, 0])
            self.assertEqual(summary["recall/minimal"]["recall_turns"]["stripped"], 0.0)
            self.assertEqual(summary["chain/minimal"]["later_steps"]["stripped"], 0.0)

            status = quiet(["score", "--out", str(out), "--efforts", "minimal", "--conversations", "2",
                            "--max-tokens", "128", "--families", "code"])
            self.assertEqual(status, 0)
            code = json.loads((out / "summary.json").read_text())["summary"]["code/minimal"]
            self.assertEqual([entry["replay"] for entry in code["per_turn"]], [1.0] * 5)
            self.assertEqual([entry["stripped"] for entry in code["per_turn"]], [1.0, 0.0, 0.0, 0.0, 0.0])
            examples = json.loads((out / "summary.json").read_text())["examples"]["code/minimal"]
            self.assertEqual({example["winner"] for example in examples}, {"replay"})


if __name__ == "__main__":
    unittest.main()
