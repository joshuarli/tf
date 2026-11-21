"""Negative gate for the Inkling 2 Mini parity check: the contract must fail loudly.

Each fixture drives the real entry point (compare.py main) with a tiny
vocabulary and asserts the verdict. A same-argmax corruption with a large
logit error, a NaN row, a wrong row count, a stale runner or token-history
receipt, and a missing receipt must all fail without claiming qualification;
a close pair must pass. A recorded explanation never turns a failure into a
pass and never applies to another history.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import provenance  # noqa: E402

COMPARE = HERE / "compare.py"
VOCAB = 8
TOKENS = [5, 7, 11]
RUNNER_SHA = "a" * 64
MODEL_ID = "model-fixture-identity"
POLICY = "test-policy"


def base_ref() -> np.ndarray:
    rng = np.random.default_rng(7)
    ref = rng.normal(0, 3, size=(len(TOKENS), VOCAB)).astype(np.float32)
    ref[np.arange(len(TOKENS)), [0, 1, 2]] += 10
    return ref


def corrupt_pair() -> tuple[np.ndarray, np.ndarray]:
    ref, cand = base_ref(), base_ref()
    ref[0, :3] = np.float32([2, 1, 0])
    ref[0, 3:] = np.float32(-5)
    cand[0, :3] = np.float32([200, 100, 0])
    cand[0, 3:] = np.float32(-5)
    return ref, cand


def write_case(directory: Path, ref: np.ndarray, cand: np.ndarray, *,
               tokens: list[int] = TOKENS, receipt_runner: str = RUNNER_SHA,
               receipt_tokens: list[int] | None = None, write_receipt: bool = True,
               exceptions: list[dict] | None = None) -> dict[str, Path]:
    histories = directory / "histories.json"
    histories.write_text(json.dumps([{"name": "tiny", "token_ids": tokens}], indent=2))
    ref_dir, metal_dir = directory / "ref", directory / "metal"
    ref_dir.mkdir(exist_ok=True)
    metal_dir.mkdir(exist_ok=True)
    np.save(ref_dir / "tiny.npy", ref.astype(np.float32))
    out = metal_dir / "tiny.bin"
    cand.astype(np.float32).tofile(out)
    if write_receipt:
        provenance.write_receipt(out, runner_sha256=receipt_runner,
                                 model_identity_digest=MODEL_ID,
                                 token_sha256=provenance.token_history_sha(
                                     receipt_tokens if receipt_tokens is not None else TOKENS),
                                 policy=POLICY, vocab=VOCAB,
                                 rows=cand.shape[0])
    exc_path = directory / "exceptions.json"
    exc_path.write_text(json.dumps({"schema": 1, "exceptions": exceptions or []}, indent=2))
    return {"histories": histories, "ref_dir": ref_dir, "metal_dir": metal_dir,
            "report": directory / "report.md", "exceptions": exc_path}


def run_compare(paths: dict[str, Path]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(COMPARE),
         "--histories", str(paths["histories"]),
         "--ref-dir", str(paths["ref_dir"]),
         "--metal-dir", str(paths["metal_dir"]),
         "--report", str(paths["report"]),
         "--vocab", str(VOCAB),
         "--contract", "f32-vs-f32",
         "--exceptions", str(paths["exceptions"]),
         "--arithmetic-policy", POLICY,
         "--runner-sha256", RUNNER_SHA,
         "--model-identity", MODEL_ID],
        capture_output=True, text=True)


def console(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


class ParityGate(unittest.TestCase):
    def test_same_argmax_large_error_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = corrupt_pair()
            result = run_compare(write_case(Path(directory), ref, cand))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertNotIn("parity: met", console(result))
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("not qualified", report)
            self.assertIn("max_abs", report)

    def test_nan_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = base_ref(), base_ref()
            cand[1, 3] = np.nan
            result = run_compare(write_case(Path(directory), ref, cand))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertIn("non-finite", console(result))

    def test_wrong_row_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref = base_ref()
            cand = np.zeros((len(TOKENS) + 1, VOCAB), dtype=np.float32)
            result = run_compare(write_case(Path(directory), ref, cand))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertIn("names 3 tokens", console(result))

    def test_runner_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref = base_ref()
            result = run_compare(write_case(Path(directory), ref, base_ref(),
                                            receipt_runner="b" * 64))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertIn("runner_sha256", console(result))

    def test_token_history_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref = base_ref()
            result = run_compare(write_case(Path(directory), ref, base_ref(),
                                            tokens=[5, 7, 12]))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertIn("token_sha256", console(result))

    def test_missing_receipt_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_compare(write_case(Path(directory), base_ref(), base_ref(),
                                            write_receipt=False))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            self.assertIn("no receipt", console(result))

    def test_clean_close_case_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref = base_ref()
            result = run_compare(write_case(Path(directory), ref, ref + np.float32(2e-5)))
            self.assertEqual(result.returncode, 0, console(result))
            self.assertIn("parity: qualified", result.stdout)
            self.assertIn("parity: qualified", (Path(directory) / "report.md").read_text())

    def test_explained_failure_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = corrupt_pair()
            exc = [{"history": "tiny", "onset_position": 0,
                    "mechanism": "fixture mechanism", "evidence": "fixture doc"}]
            result = run_compare(write_case(Path(directory), ref, cand, exceptions=exc))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("parity: qualified", console(result))
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("1 explained divergences", report)
            self.assertIn("not qualified", report)

    def test_exception_for_another_history_does_not_apply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ref, cand = corrupt_pair()
            exc = [{"history": "someone-else", "onset_position": 0,
                    "mechanism": "fixture mechanism", "evidence": "fixture doc"}]
            result = run_compare(write_case(Path(directory), ref, cand, exceptions=exc))
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("qualified", console(result))
            report = (Path(directory) / "report.md").read_text()
            self.assertIn("1 unexplained", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
