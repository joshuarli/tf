"""The Colab harness against a scripted CLI: refusals happen before allocation and every allocation is released."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("colab_harness", HERE.parent / "colab" / "colab.py")
colab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colab)

SOURCE = {"commit": "abc", "dirty": True, "fingerprint": "f" * 64, "files": []}


class FakeCli:
    """Scripted `colab` responses keyed by subcommand; records every argv in order."""

    def __init__(self, balance=400.0, live_rate=1.8, usage_text=None, new_exit=0, exec_fn=None, results=None,
                 foreign=("someone-elses",)):
        self.balance, self.rate, self.live_rate = balance, 0.0, live_rate
        self.usage_text, self.new_exit, self.exec_fn = usage_text, new_exit, exec_fn
        self.results = results if results is not None else {}
        self.sessions = set(foreign)
        self.calls = []

    def __call__(self, argv, timeout):
        args = argv[4:]
        self.calls.append(args)
        command = args[0]
        if command == "usage":
            text = self.usage_text or (f"Current balance: {self.balance:.2f} compute units\n"
                                       f"Usage rate: {self.rate:.2f}/hr\nActive assignments: {len(self.sessions)}\n")
            return subprocess.CompletedProcess(argv, 0, text, "")
        if command == "version":
            return subprocess.CompletedProcess(argv, 0, "google-colab-cli v0.8.0", "")
        if command == "new":
            if self.new_exit:
                return subprocess.CompletedProcess(argv, self.new_exit, "", "HTTP 400: no quota")
            self.sessions.add(args[2])
            self.rate += self.live_rate
            return subprocess.CompletedProcess(argv, 0, "created", "")
        if command == "exec":
            if self.exec_fn:
                return self.exec_fn(argv, timeout)
            return subprocess.CompletedProcess(argv, 0, "ran", "")
        if command == "download":
            remote, local = args[3], Path(args[4])
            name = remote.split("/")[-2]
            if name in self.results:
                local.write_text(json.dumps(self.results[name]))
                return subprocess.CompletedProcess(argv, 0, "", "")
            return subprocess.CompletedProcess(argv, 1, "", "No such file")
        if command == "upload":
            return subprocess.CompletedProcess(argv, 0, "", "")
        if command == "stop":
            name = args[2]
            if name not in self.sessions:
                return subprocess.CompletedProcess(argv, 1, "", "Session not found")
            self.sessions.discard(name)
            self.rate -= self.live_rate
            self.balance -= 0.05
            return subprocess.CompletedProcess(argv, 0, "stopped", "")
        raise AssertionError(f"unexpected colab call {args}")

    def commands(self):
        return [c[0] for c in self.calls]


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(colab, "source_fingerprint", return_value=SOURCE)
        patch.start()
        self.addCleanup(patch.stop)

    def run_with(self, cli, gpu="T4", names=("smoke",), max_cu=0.25):
        harness = colab.Colab(runner=cli, work=Path(self.tmp.name))
        return colab.run(gpu, list(names), max_cu, colab=harness), harness

    def test_a_clean_smoke_allocates_runs_retrieves_then_releases(self):
        cli = FakeCli(results={"smoke": {"status": "ok", "runtime": {"gpu": "Tesla T4"}, "correctness": {"outside_bound": 0}}})
        receipt, harness = self.run_with(cli)
        self.assertEqual(receipt["status"], "ok")
        commands = cli.commands()
        self.assertLess(commands.index("download"), commands.index("stop"))
        self.assertTrue(receipt["released"])
        self.assertEqual(receipt["consumed_cu"], 0.05)
        self.assertEqual(receipt["live_rate"], 1.8)
        self.assertEqual(harness.owned(), [])
        self.assertEqual(cli.sessions, {"someone-elses"})
        self.assertTrue(Path(self.tmp.name, next(p for p in Path(self.tmp.name).iterdir() if p.is_dir()),
                             "receipt.json").exists())

    def test_an_accelerator_needing_approval_is_refused_before_allocation(self):
        cli = FakeCli()
        receipt, _ = self.run_with(cli, gpu="A100")
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("approval", receipt["reason"])
        self.assertNotIn("new", cli.commands())

    def test_an_experiment_the_accelerator_cannot_run_is_refused(self):
        cli = FakeCli()
        receipt, _ = self.run_with(cli, gpu="CPU", names=("precision-error",))
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("sm_75", receipt["reason"])
        self.assertNotIn("new", cli.commands())

    def test_an_unknown_experiment_is_refused_not_skipped(self):
        receipt, _ = self.run_with(FakeCli(), names=("smoke", "nope"))
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("nope", receipt["reason"])

    def test_an_estimate_over_the_budget_is_refused(self):
        cli = FakeCli()
        receipt, _ = self.run_with(cli, gpu="L4", max_cu=0.25)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("MAX_CU", receipt["reason"])
        self.assertNotIn("new", cli.commands())

    def test_an_exhausted_balance_is_refused(self):
        cli = FakeCli(balance=0.1)
        receipt, _ = self.run_with(cli)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("balance", receipt["reason"])
        self.assertNotIn("new", cli.commands())

    def test_missing_authentication_refuses_without_allocation(self):
        cli = FakeCli(usage_text="To authorize colab-cli, visit this URL in any browser:\n")
        receipt, _ = self.run_with(cli)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("not authenticated", receipt["reason"])
        self.assertNotIn("new", cli.commands())

    def test_a_quota_refusal_at_allocation_leaves_nothing_owned(self):
        cli = FakeCli(new_exit=1)
        receipt, harness = self.run_with(cli)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("no quota", receipt["reason"])
        self.assertEqual(harness.owned(), [])

    def test_a_live_rate_above_the_ceiling_stops_the_runtime(self):
        cli = FakeCli(live_rate=3.5)
        receipt, harness = self.run_with(cli)
        self.assertEqual(receipt["status"], "refused")
        self.assertIn("live rate", receipt["reason"])
        self.assertNotIn("exec", cli.commands())
        self.assertTrue(receipt["released"])
        self.assertEqual(harness.owned(), [])

    def test_a_failed_experiment_still_retrieves_and_releases(self):
        def fail(argv, timeout):
            return subprocess.CompletedProcess(argv, 1, "", "Traceback: boom")
        cli = FakeCli(exec_fn=fail, results={"smoke": {"status": "ok", "runtime": {"gpu": "Tesla T4"}}})
        receipt, _ = self.run_with(cli)
        self.assertEqual(receipt["status"], "error")
        self.assertEqual(receipt["results"]["smoke"]["status"], "error")
        self.assertIn("download", cli.commands())
        self.assertTrue(receipt["released"])

    def test_a_runtime_without_the_requested_gpu_is_an_error(self):
        receipt, _ = self.run_with(FakeCli(results={"smoke": {"status": "ok", "runtime": {}}}))
        self.assertEqual(receipt["status"], "error")
        self.assertIn("not the requested T4", receipt["results"]["smoke"]["reason"])
        self.assertTrue(receipt["released"])

    def test_a_missing_artifact_is_an_error(self):
        receipt, _ = self.run_with(FakeCli(results={}))
        self.assertEqual(receipt["status"], "error")
        self.assertIn("not retrieved", receipt["results"]["smoke"]["reason"])
        self.assertTrue(receipt["released"])

    def test_an_interrupted_run_releases_the_runtime(self):
        def interrupt(argv, timeout):
            raise KeyboardInterrupt
        cli = FakeCli(exec_fn=interrupt)
        receipt, harness = self.run_with(cli)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(cli.commands()[-2:], ["stop", "usage"])
        self.assertEqual(harness.owned(), [])

    def test_the_deadline_stops_a_hung_experiment(self):
        def hang(argv, timeout):
            raise subprocess.TimeoutExpired(argv, timeout)
        cli = FakeCli(exec_fn=hang)
        receipt, _ = self.run_with(cli)
        self.assertEqual(receipt["status"], "error")
        self.assertIn("deadline", receipt["reason"])
        self.assertTrue(receipt["released"])

    def test_a_stale_remote_source_is_rejected(self):
        cli = FakeCli(live_rate=0.1, results={"env-probe": {"status": "ok", "source_fingerprint": "0" * 64}})
        receipt, _ = self.run_with(cli, gpu="CPU", names=("env-probe",))
        self.assertEqual(receipt["status"], "error")
        self.assertIn("fingerprint", receipt["results"]["env-probe"]["reason"])
        self.assertIn("upload", cli.commands())

    def test_cleanup_stops_only_owned_sessions(self):
        cli = FakeCli()
        harness = colab.Colab(runner=cli, work=Path(self.tmp.name))
        cli.sessions.add("tf-t4-abandoned")
        harness._set_owned(["tf-t4-abandoned"])
        self.assertEqual(colab.cleanup(harness), 0)
        self.assertEqual(cli.sessions, {"someone-elses"})
        self.assertEqual([c for c in cli.calls if c[0] == "stop"], [["stop", "-s", "tf-t4-abandoned"]])

    def test_a_failed_stop_keeps_the_session_owned_and_says_so(self):
        cli = FakeCli(results={"smoke": {"status": "ok", "runtime": {"gpu": "Tesla T4"}}})
        original = cli.__call__

        def stuck(argv, timeout):
            if argv[4] == "stop":
                return subprocess.CompletedProcess(argv, 1, "", "HTTP 500")
            return original(argv, timeout)
        receipt, harness = self.run_with(stuck)
        self.assertEqual(receipt["status"], "error")
        self.assertFalse(receipt["released"])
        self.assertIn("colab-cleanup", receipt["reason"])
        self.assertEqual(len(harness.owned()), 1)

    def test_the_t4_smoke_fits_a_quarter_compute_unit(self):
        _, seconds, cu = colab.admit("T4", ["smoke"], 0.25, 400)
        self.assertLessEqual(cu, 0.25)
        self.assertEqual(seconds, colab.SETUP_SECONDS + colab.EXPERIMENTS["smoke"].seconds)


class Fingerprint(unittest.TestCase):
    def test_a_dirty_file_changes_the_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "a.txt").write_text("one")
            subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
            subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "a"], cwd=repo,
                           check=True)
            clean = colab.source_fingerprint(repo, ("a.txt",))
            (repo / "a.txt").write_text("two")
            dirty = colab.source_fingerprint(repo, ("a.txt",))
            self.assertFalse(clean["dirty"])
            self.assertTrue(dirty["dirty"])
            self.assertEqual(clean["commit"], dirty["commit"])
            self.assertNotEqual(clean["fingerprint"], dirty["fingerprint"])


if __name__ == "__main__":
    unittest.main()
