"""CPU-only protocol checks: exact inputs, output accounting, timing and invalid-attempt handling."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location("nano_inkling", Path(__file__).with_name("inkling.py"))
inkling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inkling)


def receipt(**changes):
    return {"type": "nano_bench", "version": 1, "model_type": "inkling2_loop", "prompt_tokens": 4,
            "decode_tokens": 2, "repeats": 1, "warmup_runs": 1,
            "decode_mode": "free-running greedy (fixed length)", "predictions": [1, 2, 3],
            "prefill_seconds": [2.0], "decode_seconds": [0.5]} | changes


class ProtocolTests(unittest.TestCase):
    def test_workload_uses_exact_ids_once_and_reserves_all_generated_tokens(self):
        ids = [12, 23, 34, 45]
        work = inkling.workload(ids, 512)
        self.assertEqual([work["bos"], *work["code"]], ids)
        self.assertEqual(len(work["decode"]), 511)
        inkling.workload([0] * (32768 - 512), 512)
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            inkling.workload([0] * (32769 - 512), 512)

    def test_bad_inputs_are_refused_before_launch(self):
        for ids, steps in (([], 2), ([0], 2), ([True, 1], 2), ([-1, 1], 2),
                           ([200058, 1], 2), ([1, 2], 1), ([1, 2], 2.5)):
            with self.subTest(ids=ids, steps=steps), self.assertRaises(ValueError):
                inkling.workload(ids, steps)

    def test_cli_selects_inkling_cap_fixed_greedy_and_expected_knobs(self):
        command = inkling.nano_command("model", "tokens.json", 1024, 512, "optimized")
        for flag, expected in (("--context", "32768"), ("--prompt-tokens", "1024"),
                               ("--decode-tokens", "511"), ("--decode-mode", "greedy"), ("--repeats", "1")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        knobs = [command[i + 1] for i, token in enumerate(command) if token == "--knob"]
        self.assertIn("batch=128", knobs)
        self.assertIn("expert_mlx_scale=true", knobs)
        self.assertIn("relative_tile_prefill=true", knobs)
        reference = inkling.nano_command("model", "tokens.json", 1024, 512, "reference")
        self.assertEqual(reference.count("--knob"), 1)

    def test_rates_charge_only_decode_predictions_after_first_prefill_token(self):
        result = inkling.parse_result(json.dumps(receipt()), "diagnostic", "nanoattention", 4, 3)
        self.assertEqual(result["prefill_tokens_per_second"], 2)
        self.assertEqual(result["decode_tokens_per_second"], 4)
        native = receipt(type="tf_inkling_bench", prefill_seconds=2.0, decode_seconds=0.5,
                         generated_tokens=3, context=32768, cache_capacity_tokens=32768, warmup_trial_match=True)
        self.assertEqual(inkling.parse_result("", json.dumps(native), "tf", 4, 3)["predictions"], [1, 2, 3])

    def test_wrong_protocol_incomplete_outputs_and_nonfinite_times_are_invalid(self):
        changes = ({"model_type": "flint"}, {"decode_mode": "teacher-forced"}, {"prompt_tokens": 5},
                   {"decode_tokens": 3}, {"warmup_runs": 0}, {"repeats": 2}, {"predictions": [1, 2]},
                   {"predictions": [1, 2, True]}, {"prefill_seconds": [float("nan")]},
                   {"decode_seconds": [float("inf")]}, {"prefill_seconds": [0]},
                   {"decode_seconds": [-1]}, {"decode_seconds": [0.5, 0.6]})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                inkling.parse_result(json.dumps(receipt(**change)), "", "nanoattention", 4, 3)
        with self.assertRaises(ValueError):
            inkling.parse_result(json.dumps(receipt()) + "\n" + json.dumps(receipt()), "", "nanoattention", 4, 3)

    def test_divergence_is_named_and_equal_greedy_output_is_not_logit_qualification(self):
        diff = inkling.output_agreement([1, 2, 3], [1, 9, 3])
        self.assertEqual(diff["first_different_generated_index"], 1)
        self.assertEqual(diff["matching_prefix_tokens"], 1)
        self.assertFalse(diff["identical"])
        same = inkling.output_agreement([1, 2], [1, 2])
        self.assertTrue(same["identical"])
        self.assertIn("not established", same["qualification"])

    def test_unavailable_or_contended_gpu_gate_preserves_attempt_without_running_engine(self):
        for status in ("unavailable", "contended", "error 1"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "job.json"
                path.write_text(json.dumps({"command": ["unused"], "timeout": 10}))
                conditions = {"pressure": {"gpu_gate": {"status": status}}, "pressure_reasons": []}
                with mock.patch.object(inkling.harness, "conditions", return_value=conditions), \
                        mock.patch.object(inkling.subprocess, "Popen") as launch, \
                        contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(inkling.worker(path), 75)
                launch.assert_not_called()
                self.assertIn("pre-run GPU gate", json.loads(stdout.getvalue())["invalid_reasons"][0])

    def test_cancellation_stops_owned_wrapper_and_marks_saved_attempt(self):
        child = mock.Mock(pid=912, returncode=130)
        child.communicate.side_effect = [KeyboardInterrupt(), ("", "heavy: cancelled")]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(inkling.subprocess, "Popen", return_value=child), \
                mock.patch.object(inkling.harness, "process_tree", return_value={912}), \
                mock.patch.object(inkling.os, "getpgid", return_value=100), \
                mock.patch.object(inkling.os, "getpgrp", return_value=100):
            row = inkling.attempt(["unused"], Path(directory) / "trial", 10)
            self.assertTrue(row["interrupted"])
            self.assertIn("interrupted by user", row["invalid_reasons"])
            self.assertEqual(Path(row["stderr_log"]).read_text(), "heavy: cancelled")
        child.terminate.assert_called_once()

    def test_native_receipt_requires_the_shared_context_and_verified_warmup(self):
        native = receipt(type="tf_inkling_bench", generated_tokens=3, context=32768,
                         cache_capacity_tokens=32768, warmup_trial_match=True)
        for change in ({"context": 262144}, {"cache_capacity_tokens": 7}, {"warmup_trial_match": False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                inkling.parse_result("", json.dumps(native | change), "tf", 4, 3)

    def test_strict_gpu_sampling_fails_closed_and_changing_clients_invalidate(self):
        error = inkling.subprocess.CalledProcessError(1, ["ioreg"])
        with mock.patch.object(inkling.harness.subprocess, "run", side_effect=error) as sample:
            with self.assertRaises(inkling.subprocess.CalledProcessError):
                inkling.harness.gpu_time_by_pid(strict=True)
            self.assertTrue(sample.call_args.kwargs["check"])
        child = mock.Mock(pid=912, returncode=0)
        child.communicate.return_value = ("", "")
        quiet = {"pressure": {"gpu_gate": {"status": "quiet"}}, "pressure_reasons": []}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.json"
            path.write_text(json.dumps({"command": ["unused"], "timeout": 10}))
            with mock.patch.object(inkling.harness, "conditions", return_value=quiet), \
                    mock.patch.object(inkling.harness, "gpu_time_by_pid", side_effect=[{}, {999: ("other", 1)}]), \
                    mock.patch.object(inkling.subprocess, "Popen", return_value=child), \
                    contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(inkling.worker(path), 75)
            self.assertIn("GPU client set changed", json.loads(stdout.getvalue())["invalid_reasons"][0])


if __name__ == "__main__":
    unittest.main()
