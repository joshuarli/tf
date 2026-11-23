"""CPU fixtures for the campaign suite: skipped Zig tests fail a run, and selection keeps the full tier manual."""

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CAMPAIGN = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("campaign_suite", CAMPAIGN / "suite" / "suite.py")
suite = importlib.util.module_from_spec(spec)
sys.modules["campaign_suite"] = suite
spec.loader.exec_module(suite)


def args(tier="edit", names="", areas="", skip="", full=False):
    return argparse.Namespace(tier=tier, names=names, areas=areas, skip=skip, full_requested=full)


class SkippedTests(unittest.TestCase):
    def test_a_skipped_checkpoint_test_fails_the_run(self):
        log = "Build Summary: 4/4 steps succeeded; 20/24 tests passed (4 skipped)\n"
        self.assertEqual(suite.skipped_tests(log), "4 tests skipped")

    def test_every_test_run_and_passed_is_clean(self):
        self.assertIsNone(suite.skipped_tests("noise\nBuild Summary: 4/4 steps succeeded; 24/24 tests passed\n"))
        self.assertIsNone(suite.skipped_tests("Build Summary: 4/4 steps succeeded; 24/24 tests passed (0 skipped)\n"))

    def test_a_summary_without_tests_proves_nothing(self):
        self.assertIn("proved nothing", suite.skipped_tests("Build Summary: 3/3 steps succeeded\n"))
        self.assertIn("proved nothing", suite.skipped_tests(""))

    def test_the_last_summary_decides(self):
        log = ("Build Summary: 1/1 steps succeeded; 2/3 tests passed (1 skipped)\n"
               "Build Summary: 1/1 steps succeeded; 3/3 tests passed\n")
        self.assertIsNone(suite.skipped_tests(log))


class Selection(unittest.TestCase):
    def test_checkpoint_tests_refuse_skips_and_name_the_checkpoint(self):
        by_name = {w.name: w for w in suite.WORKLOADS}
        gpu = by_name["inkling-gpu"]
        self.assertTrue(gpu.no_skips and by_name["inkling-host"].no_skips)
        self.assertEqual(dict(gpu.env)["TF_INKLING2_LOOP_DIR"], str(suite.INKLING))
        self.assertEqual(dict(gpu.env)["TF_INKLING2_LOOP_OQ4E_DIR"], str(suite.INKLING_OQ4E))
        self.assertIn("inkling-weights", gpu.needs)
        self.assertIn("--summary", gpu.command)

    def test_change_ceiling_takes_the_gpu_suite_but_not_the_agent_sweep(self):
        names = {w.name for w in suite.select(args(tier="change", areas="inkling"))}
        self.assertIn("inkling-gpu", names)
        self.assertNotIn("inkling-agents-4", names)
        self.assertNotIn("inkling-oracles", names)

    def test_full_tier_and_unknown_or_empty_selections_are_refused(self):
        for bad in (args(tier="full"), args(names="inkling-oracles"), args(names="nope"),
                    args(names="inkling-gpu", skip="inkling-gpu")):
            with self.assertRaises(SystemExit):
                suite.select(bad)

    def test_the_omlx_oracle_gate_runs_in_the_scenario_tier_only(self):
        gate = {w.name: w for w in suite.WORKLOADS}["inkling-omlx-oracle"]
        self.assertEqual((gate.tier, gate.area), ("scenario", "inkling"))
        self.assertEqual(gate.command, suite.make("omlx-oracle-gate"))
        self.assertTrue({"omlx-oracle", "inkling-oq4e-weights", "target:omlx-oracle-gate"} <= set(gate.needs))
        for tier in ("edit", "change"):
            self.assertNotIn(gate.name, {w.name for w in suite.select(args(tier=tier))})
        self.assertIn(gate.name, {w.name for w in suite.select(args(tier="scenario", areas="inkling"))})
        self.assertEqual([w.name for w in suite.select(args(names=gate.name))], [gate.name])

    def test_the_omlx_oracle_preflight_wants_a_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(suite, "OMLX_ORACLE", Path(tmp)):
            self.assertIn("make -C campaign omlx-oracle-capture", suite.requirement("omlx-oracle", set(), False))
            for held in ('{"complete": false}', "{"):
                (Path(tmp) / "manifest.json").write_text(held)
                self.assertIn("incomplete", suite.requirement("omlx-oracle", set(), False))
            (Path(tmp) / "manifest.json").write_text('{"complete": true}')
            self.assertIsNone(suite.requirement("omlx-oracle", set(), False))

    def test_inkling_binary_freshness_watches_shared_core_and_metal_sources(self):
        self.assertIn(suite.REPO / "zig/src/core", suite.INKLING_SOURCES)
        self.assertIn(suite.REPO / "zig/src/metal", suite.INKLING_SOURCES)


if __name__ == "__main__":
    unittest.main()
