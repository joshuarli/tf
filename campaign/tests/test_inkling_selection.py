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

    def test_wired_buffers_already_in_the_footprint_are_charged_once(self):
        import heavy
        gb = 10**9
        self.assertEqual(heavy.charge([(20 * gb, 3 * gb)], 9 * gb), 20 * gb)
        self.assertEqual(heavy.charge([(gb, 2 * gb)], 20 * gb), 21 * gb)

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

    def test_oq4e_and_the_original_checkpoint_are_not_interchangeable(self):
        original = "Inkling-2-Mini@ed40e7da625f"
        oq4e = "Inkling-2-Mini-oQ4e-mtp@3eea042751cd"
        self.assertTrue(harness.served_model_matches("inkling-2-mini-oq4e-mtp", oq4e))
        self.assertFalse(harness.served_model_matches("inkling-2-mini", oq4e))
        self.assertFalse(harness.served_model_matches("inkling-2-mini-oq4e-mtp", original))

    def test_oq4e_is_served_under_the_inkling_contract(self):
        key = harness.model_key("inkling2-mini-oq4e")
        self.assertEqual(harness.model_dir(key).name, "Inkling-2-Mini-oQ4e-mtp@3eea042751cd")
        self.assertEqual(harness.model_key_for_dir(harness.model_dir(key)), key)
        self.assertEqual(harness.load_limit_env(harness.model_dir(key)), {"INKLING_LOAD_LIMIT_GB": "26"})
        self.assertEqual(harness.default_serve_args(key), ["--no-drafts"])
        self.assertEqual(harness.model_serve_args(key), ["--temperature", "0"])
        vision = [{"id": "vision-v01", "kind": "vision", "prompt": "describe"}]
        self.assertEqual(quality_run.applicable(vision, key)[0], [])

    def test_pi_identity_check_names_the_family(self):
        problems = pi_run.identity_problems(
            {"model_ids": ["flint-int4@3b93f9053b53"], "models": [{"owned_by": "tensorfold"}]},
            binary_sha256="sha", model_key="inkling-2-mini", expected_model=None)
        self.assertTrue(any("do not name model 'inkling-2-mini'" in problem for problem in problems))


class ModelModes(unittest.TestCase):
    def test_inkling_refuses_the_drafter_with_a_reason_and_serves_copy_drafts(self):
        reason = harness.model_mode_problem("inkling-2-mini", "drafter")
        self.assertTrue(reason, "drafter must be refused for Inkling")
        self.assertIsNone(harness.model_mode_problem("inkling-2-mini", "copy"))
        self.assertEqual(harness.copy_draft_env("inkling-2-mini"), {"INKLING_COPY_DRAFTS": "1"})
        self.assertEqual(harness.copy_draft_env("flint-int4"), {})

    def test_inkling_serves_ar(self):
        self.assertIsNone(harness.model_mode_problem("inkling-2-mini", "ar"))
        self.assertEqual(harness.default_serve_args("inkling-2-mini"), ["--no-drafts"])

    def test_every_inkling_server_gets_the_context_cap_whatever_its_caller_passes(self):
        from unittest import mock
        with mock.patch.object(harness, "model_key_for_dir", lambda path: "inkling-2-mini"):
            for extra in ([], ["--no-drafts"]):
                server = harness.Server(Path("bin"), Path("model"), extra, 2, Path("log"), {"INKLING_COPY_DRAFTS": "1"})
                server.port = 1
                argv = server.argv
                self.assertEqual(argv[argv.index("--context") + 1], "32768")
                self.assertEqual(argv[argv.index("--temperature") + 1], "0")
                self.assertEqual(server.env["INKLING_COPY_DRAFTS"], "1")
        with mock.patch.object(harness, "model_key_for_dir", lambda path: None):
            self.assertNotIn("--context", harness.Server(Path("bin"), Path("model"), [], 2, Path("log")).extra_args)

    def test_flint_is_unchanged(self):
        self.assertIsNone(harness.model_mode_problem("flint-int4", "ar"))
        self.assertIsNone(harness.model_mode_problem("flint-int4", "copy"))
        self.assertEqual(harness.default_serve_args("flint-int4"), [])
        self.assertEqual(harness.model_serve_args("flint-int4"), [])


class QualityHostConditions(unittest.TestCase):
    def test_other_gpu_users_do_not_refuse_a_quality_run_but_memory_pressure_does(self):
        gpu = "gpu-gate-ready: contended (check 1 of 1): other processes used 103.4% GPU time (limit 5); "
        self.assertEqual(quality_run.quality_blockers([gpu]), [])
        self.assertEqual(quality_run.quality_blockers([gpu, "memory pressure level 2"]), ["memory pressure level 2"])


class PiServerLoss(unittest.TestCase):
    def test_a_server_that_exits_mid_run_fails_the_run_and_stops_it(self):
        import tempfile
        from unittest import mock

        class Process:
            def __init__(self):
                self.polls = 0

            def poll(self):
                self.polls += 1
                return 4

        class Server:
            def __init__(self, *args):
                self.base, self.port, self.argv, self.process = "http://127.0.0.1:1", 1, ["serve"], Process()

            def start(self, timeout):
                pass

            def stop(self):
                pass

        ran = []

        def run_task(task_dir, name, *args):
            ran.append(name)
            return {"task": name, "completed": True, "checker": {"pass": False}, "turns": 1, "pi_event_errors": []}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a-task", "b-task"):
                (root / "tasks" / name).mkdir(parents=True)
                (root / "tasks" / name / "prompt.txt").write_text("text task")
            (root / "agent").mkdir()
            (root / "agent" / "models.json").write_text("{}")
            binary = root / "bin"
            binary.write_text("")
            identity = {"model_ids": ["Inkling-2-Mini@ed40e7da625f"], "models": []}
            with mock.patch.object(pi_run, "__file__", str(root / "run.py")), \
                    mock.patch.object(pi_run.harness, "Server", Server), \
                    mock.patch.object(pi_run.harness, "model_dir", lambda key: root), \
                    mock.patch.object(pi_run.harness, "model_hashes", lambda key: {}), \
                    mock.patch.object(pi_run.harness, "host_summary", lambda key: {}), \
                    mock.patch.object(pi_run, "served_identity", lambda base: identity), \
                    mock.patch.object(pi_run, "identity_problems", lambda *a, **k: []), \
                    mock.patch.object(pi_run, "render_models", lambda *a: {}), \
                    mock.patch.object(pi_run, "metrics", lambda base: {}), \
                    mock.patch.object(pi_run, "run_task", run_task), \
                    mock.patch.object(sys, "argv", ["run.py", "--model", "inkling2-mini", "--bin", str(binary)]):
                status = pi_run.main()
            record = json.loads((root / "results" / json.loads(
                (root / "results" / "latest-inkling2-mini.json").read_text())["result"].split("/")[-1]).read_text())
        self.assertEqual(status, 1)
        self.assertEqual(ran, ["a-task"])
        self.assertEqual(record["server"]["exit"], 4)
        self.assertFalse(record["tasks"][0]["completed"])
        self.assertEqual(record["totals"]["tasks"], 2)
        self.assertFalse(record["completion"]["complete"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
