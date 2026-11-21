"""CPU fixtures for the Inkling 2 Mini model selection: the load-limit variable, excluded cases, served identity."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = CAMPAIGN.parent

for entry in (CAMPAIGN, CAMPAIGN / "bench"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import harness  # noqa: E402


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quality_run = load(CAMPAIGN / "quality" / "run.py", "inkling_quality_run")
pi_run = load(CAMPAIGN / "pi" / "run.py", "inkling_pi_run")


class LoadLimitSelection(unittest.TestCase):
    def test_each_model_picks_its_own_load_limit_variable(self):
        self.assertEqual(harness.load_limit_var("flint-int4"), "FLINT_LOAD_LIMIT_GB")
        self.assertEqual(harness.load_limit_var("inkling-2-mini"), "INKLING_LOAD_LIMIT_GB")

    def test_inkling_limit_fits_its_checkpoint_under_the_heavy_cap(self):
        import heavy
        self.assertEqual(heavy.INKLING_LOAD_LIMIT_GB, 26)
        self.assertLess(heavy.INKLING_LOAD_LIMIT_GB, heavy.HEAVY_CAP_GB)
        self.assertGreater(heavy.INKLING_LOAD_LIMIT_GB, 19.5, "the 19.5 GB checkpoint plus caches must fit")

    def test_flint_limit_and_value_are_unchanged(self):
        import heavy
        self.assertEqual(heavy.FLINT_LOAD_LIMIT_GB, 27)
        self.assertEqual(harness.load_limit_gb("flint-int4"), 27)

    def test_inkling_limit_is_the_inkling_constant(self):
        import heavy
        self.assertEqual(harness.load_limit_gb("inkling-2-mini"), heavy.INKLING_LOAD_LIMIT_GB)

    def test_server_env_sets_the_variable_for_its_model(self):
        flint = harness.load_limit_env(harness.model_dir("flint-int4"))
        inkling = harness.load_limit_env(harness.model_dir("inkling-2-mini"))
        self.assertEqual(flint, {"FLINT_LOAD_LIMIT_GB": "27"})
        self.assertEqual(inkling, {"INKLING_LOAD_LIMIT_GB": "26"})

    def test_model_key_for_dir_resolves_pinned_dirs_and_nothing_else(self):
        self.assertEqual(harness.model_key_for_dir(harness.model_dir("flint-int4")), "flint-int4")
        self.assertEqual(harness.model_key_for_dir(harness.model_dir("inkling-2-mini")), "inkling-2-mini")
        self.assertIsNone(harness.model_key_for_dir(Path("/tmp/not-a-pinned-model")))


class ExcludedCases(unittest.TestCase):
    def test_vision_cases_are_excluded_with_a_reason(self):
        cases = [{"id": "code-c01", "kind": "code", "prompt": "text"},
                 {"id": "vision-v01", "kind": "vision", "prompt": "describe"},
                 {"id": "chat-image", "kind": "code",
                  "prompt": [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "x"}}]}]
        kept, excluded = quality_run.applicable(cases, "inkling-2-mini")
        self.assertEqual([case["id"] for case in kept], ["code-c01"])
        self.assertEqual([case["id"] for case in excluded], ["vision-v01", "chat-image"])
        for row in excluded:
            self.assertIn("text-only", row["reason"])

    def test_flint_excludes_nothing(self):
        cases = [{"id": "vision-v01", "kind": "vision", "prompt": "describe"}]
        kept, excluded = quality_run.applicable(cases, "flint-int4")
        self.assertEqual(len(kept), 1)
        self.assertEqual(excluded, [])

    def test_the_dev_set_is_entirely_text_so_inkling_scores_it_all(self):
        document = json.loads((CAMPAIGN / "quality" / "cases.json").read_text(encoding="utf-8"))
        dev = [case for case in document["cases"] if "dev" in case["sets"]]
        kept, excluded = quality_run.applicable(dev, "inkling-2-mini")
        self.assertEqual(len(kept), 12)
        self.assertEqual(excluded, [])

    def test_pi_excludes_image_tasks_for_inkling_only(self):
        campaign_pi = CAMPAIGN / "pi"
        tasks = sorted(path for path in (campaign_pi / "tasks").iterdir() if path.is_dir())
        kept, excluded = pi_run.applicable(tasks, "inkling-2-mini")
        self.assertEqual(len(kept), len(tasks))
        self.assertEqual(excluded, [])

        class FakeTask:
            name = "fake-image-task"

            def __init__(self, directory):
                self.directory = directory

            def __truediv__(self, name):
                return self.directory / name

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "repo").mkdir()
            (root / "prompt.txt").write_text("Look at screenshot.png and describe it", encoding="utf-8")
            task = FakeTask(root)
            kept, excluded = pi_run.applicable([task], "inkling-2-mini")
            self.assertEqual(kept, [])
            self.assertEqual(excluded[0]["task"], "fake-image-task")
            self.assertIn("text-only", excluded[0]["reason"])
            self.assertEqual(pi_run.applicable([task], "flint-int4"), ([task], []))


class ServedIdentity(unittest.TestCase):
    def test_identity_name_is_the_repository_basename(self):
        self.assertEqual(harness.served_identity_name("inkling-2-mini"), "Inkling-2-Mini")
        self.assertEqual(harness.served_identity_name("flint-int4"), "flint-int4")

    def test_an_inkling_endpoint_is_recognised_by_name(self):
        self.assertTrue(harness.served_model_matches("inkling-2-mini", "Inkling-2-Mini@ed40e7da625f"))
        self.assertFalse(harness.served_model_matches("inkling-2-mini", "flint-int4@3b93f9053b53"))
        self.assertFalse(harness.served_model_matches("inkling-2-mini", None))

    def test_a_flint_endpoint_is_not_mistaken_for_inkling(self):
        self.assertTrue(harness.served_model_matches("flint-int4", "flint-int4@3b93f9053b53"))
        self.assertFalse(harness.served_model_matches("flint-int4", "Inkling-2-Mini@ed40e7da625f"))

    def test_pi_identity_check_names_the_family(self):
        problems = pi_run.identity_problems(
            {"model_ids": ["flint-int4@3b93f9053b53"], "models": [{"owned_by": "tensorfold"}]},
            binary_sha256="sha", model_key="inkling-2-mini", expected_model=None)
        self.assertTrue(any("do not name model 'inkling-2-mini'" in problem for problem in problems))


class ModelModes(unittest.TestCase):
    def test_inkling_refuses_copy_and_drafter_with_a_reason(self):
        for mode in ("copy", "drafter"):
            reason = harness.model_mode_problem("inkling-2-mini", mode)
            self.assertIsNotNone(reason, f"{mode} must be refused for Inkling")
            self.assertTrue(reason)

    def test_inkling_serves_ar(self):
        self.assertIsNone(harness.model_mode_problem("inkling-2-mini", "ar"))
        self.assertEqual(harness.default_serve_args("inkling-2-mini"), ["--no-drafts"])

    def test_flint_is_unchanged(self):
        self.assertIsNone(harness.model_mode_problem("flint-int4", "ar"))
        self.assertIsNone(harness.model_mode_problem("flint-int4", "copy"))
        self.assertEqual(harness.default_serve_args("flint-int4"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
