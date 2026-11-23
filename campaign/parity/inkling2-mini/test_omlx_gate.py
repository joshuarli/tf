"""CPU fixtures for the oMLX oracle: the streamed metrics, the gate's refusals and thresholds, and the capture's
receipts. Tiny vocabularies stand in for the 200,058-token one; no model or GPU is touched."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import omlx_capture  # noqa: E402
import omlx_gate  # noqa: E402
import provenance  # noqa: E402

VOCAB = 7
POLICY = "f32-activations+prefill-512"


def logits(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 3.0, size=(rows, VOCAB)).astype(np.float32)


class Metrics(unittest.TestCase):
    def test_identical_captures_agree_everywhere(self):
        a = logits(10, 1)
        m = omlx_gate.history_metrics(a, a.copy(), 0.1, 3)
        self.assertEqual((m["top1"], m["rows"]), (10, 10))
        self.assertEqual((m["mean_kl"], m["max_kl"], m["max_abs"]), (0.0, 0.0, 0.0))

    def test_kl_is_reference_to_candidate_in_nats_across_blocks(self):
        ref = np.zeros((5, 2), dtype=np.float32)
        cand = np.tile(np.array([math.log(3.0), 0.0], dtype=np.float32), (5, 1))
        m = omlx_gate.history_metrics(ref, cand, 0.1, 2)
        self.assertAlmostEqual(m["mean_kl"], 0.5 * math.log(4.0 / 3.0), places=6)
        self.assertAlmostEqual(m["max_kl"], 0.5 * math.log(4.0 / 3.0), places=6)

    def test_confident_positions_follow_the_reference_margin(self):
        ref = np.array([[2.0, 0.0, 0.0], [1.0, 0.95, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
        cand = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        m = omlx_gate.history_metrics(ref, cand, 0.1, 2)
        self.assertEqual((m["top1"], m["confident"], m["confident_top1"]), (1, 2, 1))
        self.assertEqual(m["disagreements"], [1, 2])
        self.assertEqual(m["max_kl_position"], 2)

    def test_a_pick_among_tied_maxima_agrees(self):
        ref = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]], dtype=np.float32)
        cand = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [2.0, 2.0, 0.0]], dtype=np.float32)
        m = omlx_gate.history_metrics(ref, cand, 0.1, 2)
        self.assertEqual((m["top1"], m["reference_ties"], m["disagreements"]), (2, 2, [1]))

    def test_non_finite_candidate_logits_are_counted(self):
        a = logits(4, 2)
        b = a.copy()
        b[3, 1] = np.nan
        m = omlx_gate.history_metrics(a, b, 0.1, 4)
        self.assertEqual((m["nonfinite_candidate"], m["nonfinite_kl_rows"]), (1, 1))

    def test_pooled_mean_kl_is_the_mean_of_history_means(self):
        rows = [{"name": "a", "rows": 1, "top1": 1, "confident": 1, "confident_top1": 1, "reference_ties": 0,
                 "mean_kl": 0.4,
                 "kl_sum": 0.4, "max_kl": 0.4, "max_kl_position": 0, "max_abs": 1.0, "nonfinite_reference": 0,
                 "nonfinite_candidate": 0, "nonfinite_kl_rows": 0},
                {"name": "b", "rows": 3, "top1": 2, "confident": 0, "confident_top1": 0, "reference_ties": 1,
                 "mean_kl": 0.0,
                 "kl_sum": 0.0, "max_kl": 0.0, "max_kl_position": 0, "max_abs": 0.0, "nonfinite_reference": 0,
                 "nonfinite_candidate": 0, "nonfinite_kl_rows": 0}]
        s = omlx_gate.summarize(rows)
        self.assertAlmostEqual(s["mean_kl"], 0.2)
        self.assertAlmostEqual(s["row_mean_kl"], 0.1)
        self.assertAlmostEqual(s["top1_percent"], 75.0)
        self.assertEqual(s["max_kl_at"], {"history": "a", "position": 0})


class Captures(unittest.TestCase):
    def test_npy_and_raw_rows_open_and_wrong_shapes_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            np.save(folder / "a.npy", logits(3, 3))
            logits(2, 4).tofile(folder / "b.bin")
            (folder / "c.bin").write_bytes(b"\0" * (4 * VOCAB + 4))
            np.save(folder / "d.npy", logits(3, 5).astype(np.float64))
            self.assertEqual(omlx_gate.open_capture(folder, "a", VOCAB)[0].shape, (3, VOCAB))
            self.assertEqual(omlx_gate.open_capture(folder, "b", VOCAB)[0].shape, (2, VOCAB))
            for name in ("c", "d", "missing"):
                data, why = omlx_gate.open_capture(folder, name, VOCAB)
                self.assertIsNone(data)
                self.assertTrue(why)
            self.assertIsNone(omlx_gate.open_capture(folder, "a", VOCAB, (".bin",))[0])
            self.assertIsNone(omlx_gate.open_capture(folder, "b", VOCAB, (".npy",))[0])


class Gate(unittest.TestCase):
    """A two-history oracle, a runner capture of it and a gate file whose baseline is the clean comparison."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.histories = root / "histories.json"
        entries = [{"name": "short", "token_ids": [1, 2, 3, 4]}, {"name": "long", "token_ids": list(range(1, 13))}]
        self.histories.write_text(json.dumps(entries))
        self.oracle, self.cand, self.model = root / "oracle", root / "cand", root / "model"
        for folder in (self.oracle, self.cand, self.model):
            folder.mkdir()
        (self.model / "config.json").write_text("{}")
        self.runner = root / "runner"
        self.runner.write_bytes(b"runner")
        outputs = {}
        for i, entry in enumerate(entries):
            name, tokens = entry["name"], entry["token_ids"]
            ref = logits(len(tokens), 10 + i)
            np.save(self.oracle / f"{name}.npy", ref)
            outputs[name] = {"rows": len(tokens), "token_sha256": provenance.token_history_sha(tokens),
                             "output_sha256": provenance.sha256_file(self.oracle / f"{name}.npy"), "nonfinite": 0}
            self.write_candidate(name, tokens, ref + np.float32(1e-3) * logits(len(tokens), 20 + i))
        digest = provenance.capture_digest({n: o["output_sha256"] for n, o in outputs.items()})
        self.manifest = {"complete": True, "histories": {"sha256": provenance.sha256_file(self.histories)},
                         "layout": {"vocab": VOCAB}, "outputs": outputs, "digest": digest}
        (self.oracle / "manifest.json").write_text(json.dumps(self.manifest))
        self.gate = root / "gate.json"
        self.write_gate(digest)
        self.entries = entries

    def tearDown(self):
        self.tmp.cleanup()

    def write_candidate(self, name, tokens, rows):
        path = self.cand / f"{name}.bin"
        rows.astype(np.float32).tofile(path)
        provenance.write_receipt(path, runner_sha256=provenance.sha256_file(self.runner),
                                 model_identity_digest=provenance.model_identity(self.model),
                                 token_sha256=provenance.token_history_sha(tokens), policy=POLICY,
                                 vocab=VOCAB, rows=len(tokens))

    def write_gate(self, digest, top1=16, mean_kl=1e-5, row_mean_kl=1e-5):
        self.gate.write_text(json.dumps({
            "oracle": {"digest": digest, "engine": "fixture"},
            "candidate": {"runner_mode": "prefill", "prefill_chunk": 512, "arithmetic_policy": POLICY},
            "baseline": {"revision": "fixture", "positions": 16, "top1": top1, "mean_kl": mean_kl,
                         "row_mean_kl": row_mean_kl, "max_kl": 1e-4},
            "margin": {"top1_points": 0.5, "mean_kl_factor": 1.5}}))

    def gate_exit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.run_gate()

    def run_gate(self):
        return omlx_gate.main(["gate", "--histories", str(self.histories), "--oracle", str(self.oracle),
                               "--cand", str(self.cand), "--gate", str(self.gate), "--runner", str(self.runner),
                               "--model", str(self.model), "--vocab", str(VOCAB), "--block", "5",
                               "--json", str(self.cand / "gate.json")])

    def verdict(self):
        return json.loads((self.cand / "gate.json").read_text())

    def test_a_faithful_candidate_passes(self):
        self.assertEqual(self.gate_exit(), 0)
        self.assertEqual(self.verdict()["verdict"], "pass")

    def test_lost_top1_agreement_fails(self):
        tokens = self.entries[1]["token_ids"]
        ref = np.load(self.oracle / "long.npy")
        flipped = ref.copy()
        flipped[:, 0] = ref.max(axis=1) + 5.0
        self.write_candidate("long", tokens, flipped)
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("top-1 agreement" in f for f in self.verdict()["failures"]))

    def test_mean_kl_above_its_bound_fails(self):
        self.write_gate(self.manifest["digest"], mean_kl=1e-9)
        self.assertEqual(self.gate_exit(), 1)
        failures = self.verdict()["failures"]
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0].startswith("mean KL "))

    def test_row_weighted_mean_kl_above_its_bound_fails(self):
        self.write_gate(self.manifest["digest"], row_mean_kl=1e-9)
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(all(f.startswith("row-weighted mean KL") for f in self.verdict()["failures"]))

    def test_a_candidate_is_read_only_from_the_runner_output_its_receipt_covers(self):
        np.save(self.cand / "short.npy", np.zeros((4, VOCAB), dtype=np.float32))
        self.assertEqual(self.gate_exit(), 0)

    def test_an_unreadable_oracle_manifest_is_refused(self):
        (self.oracle / "manifest.json").write_text("{")
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("unreadable manifest" in f for f in self.verdict()["failures"]))

    def test_another_oracle_is_refused(self):
        self.write_gate("0" * 64)
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("baseline was measured against" in f for f in self.verdict()["failures"]))

    def test_an_oracle_file_changed_after_its_manifest_is_refused(self):
        np.save(self.oracle / "short.npy", logits(4, 99))
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("is not the file the manifest records" in f for f in self.verdict()["failures"]))

    def test_a_capture_from_another_runner_is_refused(self):
        self.runner.write_bytes(b"rebuilt runner")
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("runner_sha256" in f for f in self.verdict()["failures"]))

    def test_a_missing_runner_is_a_problem_not_a_crash(self):
        self.runner.unlink()
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("no runner at" in f for f in self.verdict()["failures"]))

    def test_a_non_finite_candidate_logit_fails(self):
        tokens = self.entries[0]["token_ids"]
        rows = np.load(self.oracle / "short.npy").copy()
        rows[2, 3] = np.inf
        self.write_candidate("short", tokens, rows)
        self.assertEqual(self.gate_exit(), 1)
        self.assertTrue(any("non-finite" in f for f in self.verdict()["failures"]))

    def test_thresholds_derive_from_the_baseline_and_margins(self):
        bounds = omlx_gate.thresholds({"baseline": {"positions": 3015, "top1": 2900, "mean_kl": 0.02,
                                                    "row_mean_kl": 0.06, "max_kl": 4.0},
                                       "margin": {"top1_points": 0.5, "mean_kl_factor": 1.5}})
        self.assertAlmostEqual(bounds["min_top1_percent"], 100.0 * 2900 / 3015 - 0.5)
        self.assertAlmostEqual(bounds["max_mean_kl"], 0.03)
        self.assertAlmostEqual(bounds["max_row_mean_kl"], 0.09)


class CaptureReceipts(unittest.TestCase):
    def test_a_capture_is_kept_only_under_the_same_inputs_and_digest(self):
        engine = {"packages": {"omlx": "0.8.3"}, "omlx_commit": "e72868dac12d", "python": "3.13"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "greeting.npy"
            np.save(out, logits(3, 7))
            want = omlx_capture.wanted_receipt("greeting", [1, 2, 3], 2048, engine, "identity")
            receipt = want | {"output_sha256": provenance.sha256_file(out), "nonfinite": 0}
            omlx_capture.receipt_path(out).write_text(json.dumps(receipt))
            self.assertIsNotNone(omlx_capture.held_receipt(out, want))
            for changed in ({"chunk": 512}, {"token_sha256": "0" * 64},
                            {"engine": engine | {"omlx_commit": "other"}}, {"model_identity": "other"}):
                self.assertIsNone(omlx_capture.held_receipt(out, want | changed))
            omlx_capture.receipt_path(out).write_text(json.dumps(receipt | {"nonfinite": 1}))
            self.assertIsNone(omlx_capture.held_receipt(out, want))
            omlx_capture.receipt_path(out).write_text(json.dumps(receipt))
            np.save(out, logits(3, 8))
            self.assertIsNone(omlx_capture.held_receipt(out, want))

    def test_checkpoint_files_must_match_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text("{}")
            good = {"name": "m/config.json", "size": 2, "sha256": provenance.sha256_file(model / "config.json")}
            self.assertEqual(omlx_capture.check_checkpoint(model, [good], hash_weights=True)[0]["sha256"],
                             good["sha256"])
            for bad in (good | {"size": 3}, good | {"sha256": "0" * 64}, good | {"name": "m/missing.json"}):
                with self.assertRaises(SystemExit):
                    omlx_capture.check_checkpoint(model, [bad], hash_weights=True)

    def test_the_checkpoint_directory_must_name_the_pinned_revision(self):
        entries = omlx_capture.lock_entries(omlx_capture.CHECKPOINT)
        self.assertTrue(entries and all(e["model"] == omlx_capture.LOCK_MODEL for e in entries))
        with self.assertRaises(SystemExit):
            omlx_capture.lock_entries(Path.home() / "models" / "Inkling-2-Mini-oQ4e-mtp@000000000000")


if __name__ == "__main__":
    unittest.main()
