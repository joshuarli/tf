"""A context limit includes output headroom and must exclude oversized fixtures before inference."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN / "bench"))

import bench
import context_policy
import harness


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, CAMPAIGN / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


quality = load("quality/run.py", "context_quality")
qualify = load("quality/qualify.py", "context_qualify")


class ContextBudget(unittest.TestCase):
    def test_reserved_completion_is_part_of_the_32k_limit(self):
        for output in (512, 8192):
            self.assertIsNone(harness.context_problem(32768 - output, output))
            self.assertIn("budget exceeded", harness.context_problem(32769 - output, output))
        self.assertIn("fixture was not truncated", harness.context_problem(32768, 512))

    def test_invalid_counts_cannot_bypass_the_limit(self):
        for prompt, output in ((-1, 512), (32768, -512), (True, 512), (1, 2.5)):
            with self.assertRaises(ValueError):
                harness.context_problem(prompt, output)

    def test_nominal_32k_prompt_is_refused_before_server_start(self):
        prompts = {32768: [{"name": "code-32768", "expected_prompt_tokens": 32768}]}
        with self.assertRaisesRegex(SystemExit, "32768 rendered prompt tokens.*512 reply tokens"):
            bench.verify_context(prompts)
        self.assertEqual(prompts[32768][0]["expected_prompt_tokens"], 32768)

    def test_server_and_pairing_record_the_same_fixed_context(self):
        with mock.patch.object(harness, "load_limit_env", return_value={}):
            server = harness.Server(Path("native"), Path("model"), ["--no-drafts"], 0, Path("log"))
        self.assertEqual(server.argv[server.argv.index("--context") + 1], "32768")
        contract = bench.contract(key="inkling-2-mini", hashes={}, host={}, mode="ar", flags=["--no-drafts"],
                                  cell=1024, cache_gib=0, prompts=[], copy_drafts=False)
        self.assertEqual(contract["context_policy"]["max_tokens"], 32768)
        for args in (["--context", "262144"], ["--context=262144"]):
            with self.assertRaisesRegex(ValueError, "fix --context at 32768"):
                harness.Server(Path("native"), Path("model"), args, 0, Path("log"))

    def test_preflight_preserves_template_and_tools_in_the_tokenizer_request(self):
        body = {"messages": [{"role": "user", "content": "hello"}], "tools": [{"type": "function"}],
                "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 512}
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"count": 32500}).encode()
        with mock.patch.object(context_policy.urllib.request, "urlopen", return_value=response) as post:
            count = harness.rendered_prompt_tokens("http://127.0.0.1:9999", body)
        request = post.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:9999/v1/tokenize")
        self.assertEqual(json.loads(request.data), body)
        self.assertIn("32500 rendered prompt tokens", harness.context_problem(count, 512))

    def test_pi_reserves_room_for_its_output(self):
        document = json.loads((CAMPAIGN / "pi/agent/models.json").read_text())
        model = document["providers"]["tensorfold"]["models"][0]
        self.assertEqual(model["contextWindow"], 32768)
        self.assertEqual(model["maxTokens"], 8192)
        self.assertGreater(model["contextWindow"] - model["maxTokens"], 0)

    def test_quality_exclusion_is_not_a_model_failure_or_a_qualified_smaller_suite(self):
        class Server:
            base, argv = "http://127.0.0.1:9999", ["serve", "--context", "32768"]

            def __init__(self, *args):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def models(self):
                return ["Inkling-2-Mini@test"]

            def timing(self):
                return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "native").write_bytes(b"binary")
            (root / "prompt.txt").write_text("unchanged fixture")
            cases = [{"id": "retrieval-long", "kind": "retrieval", "sets": ["dev"], "prompt_file": "prompt.txt"}]
            args = SimpleNamespace(model="inkling2-mini", set="dev", bin=str(root / "native"), run_id="cap-test")
            with mock.patch.object(quality, "REPO", root), \
                    mock.patch.object(quality, "HERE", root), \
                    mock.patch.object(quality, "RESULTS", root / "results"), \
                    mock.patch.object(quality, "load_suite", return_value=({}, {})), \
                    mock.patch.object(quality, "select", return_value=cases), \
                    mock.patch.object(harness, "model_dir", return_value=root), \
                    mock.patch.object(harness, "model_hashes", return_value={}), \
                    mock.patch.object(harness, "host_summary", return_value={}), \
                    mock.patch.object(harness, "sha256_file", return_value="sha"), \
                    mock.patch.object(harness, "conditions", return_value={"pressure": {}, "pressure_reasons": []}), \
                    mock.patch.object(harness, "Server", Server), \
                    mock.patch.object(harness, "rendered_prompt_tokens", return_value=25000), \
                    mock.patch.object(quality, "send") as send, \
                    mock.patch("sys.stdout", new_callable=io.StringIO), \
                    mock.patch("sys.stderr", new_callable=io.StringIO):
                self.assertEqual(quality.run(args), 1)
            send.assert_not_called()
            result = json.loads((root / "results/cap-test/result.json").read_text())
            self.assertEqual(result["totals"]["failed"], 0)
            self.assertEqual(result["completion"]["context_excluded"], 1)
            self.assertFalse(result["completion"]["complete"])
            self.assertEqual(result["excluded"][0]["id"], "retrieval-long")
            self.assertFalse(qualify.covered(result)[0])
            self.assertEqual((root / "prompt.txt").read_text(), "unchanged fixture")


if __name__ == "__main__":
    unittest.main()
