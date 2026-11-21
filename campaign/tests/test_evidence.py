"""CPU regression fixtures for the campaign's comparison machinery; each names the defect it pins."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
REPO = CAMPAIGN.parent
BASE_REV = "e3f339fea76760a955d92f0973cd166fa7ec568a"

for entry in (CAMPAIGN, CAMPAIGN / "bench"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import bench  # noqa: E402
import harness  # noqa: E402
import scorecard  # noqa: E402

HOST = {"product": "Apple M5 Max", "os": "macOS 26B123", "memory_bytes": 51539607552,
        "recommended_max_working_set_bytes": 40200896512, "performance_cores": 6,
        "efficiency_cores": 12, "gpu_cores": 40, "metal": "Metal 4"}
PROMPT = {"name": "code-1024", "sha256": "a" * 64}
HASHES = {"revision": "3b93f9053b53", "aggregate_sha256": "agg-flint", "config": []}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quality_qualify = load(CAMPAIGN / "quality" / "qualify.py", "quality_qualify")
pi_qualify = load(CAMPAIGN / "pi" / "qualify.py", "pi_qualify")
pi_run = load(CAMPAIGN / "pi" / "run.py", "pi_run")


def legacy(name: str, relative: str):
    """The base-revision source of a module, loaded beside the new one so a fixture can show it failing."""
    key = f"legacy_{name}"
    if key in sys.modules:
        return sys.modules[key]
    shown = subprocess.run(["git", "show", f"{BASE_REV}:{relative}"], cwd=REPO, capture_output=True, check=True)
    directory = Path(tempfile.mkdtemp(prefix=f"legacy-{name}-"))
    path = directory / f"{name}.py"
    path.write_bytes(shown.stdout)
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def make_contract(*, mode="ar", cell=1024, cache=2, key="flint-int4", prompts=(PROMPT,)):
    flags = bench.MODES[mode]["flags"]
    return bench.contract(key=key, hashes=HASHES, host=HOST, mode=mode, flags=list(flags), cell=cell,
                          cache_gib=cache, prompts=list(prompts), copy_drafts=bench.MODES[mode]["copy_drafts"])


def trial(run_id, role, tps, *, contract, pair=1, session=None, experiment=None, prompt=PROMPT, mode="ar",
          cell=1024, cache=2, engine=None, valid=True):
    engine = engine or f"tensorfold-native-{role}"
    return {"kind": "trial", "valid": valid, "run_id": run_id, "pair": pair, "mode": mode, "cell": cell,
            "prompt_cache_gib": cache, "engine": engine,
            "binary": {"role": role, "sha256": f"{role}-sha256"},
            "pairing": bench.pairing(experiment or run_id, session or run_id, cell, mode, pair, role, contract),
            "prompt": {"name": prompt["name"], "sha256": prompt["sha256"]},
            "requests": {"cold": {"decode_tps": tps, "prompt_tokens": cell, "ttft_s": 0.1, "total_s": 4.0,
                                  "cached_tokens": 0, "cache_state": "cold"}},
            "natural_termination": None}


def speedups(trials):
    return scorecard.paired_comparisons(scorecard.collect_pairs(trials)[0])


def rejected(trials):
    return scorecard.collect_pairs(trials)[1]


class ModelSeparation(unittest.TestCase):
    """Flint at 160 tok/s and Inkling at 30 tok/s under one engine label used to pool into one 95 tok/s row."""

    def records(self, engine_for_inkling="tensorfold-native"):
        flint = trial("run-f", "baseline", 160.0, contract=make_contract(), engine="tensorfold-native")
        flint["model"] = {"key": "flint-int4", "hashes": {"revision": "3b93f9053b53"}}
        inkling = trial("run-i", "baseline", 30.0, contract=make_contract(key="inkling-2-mini"),
                        engine=engine_for_inkling)
        inkling["model"] = {"key": "inkling-2-mini", "hashes": {"revision": "ed40e7da625f"}}
        return [flint, inkling]

    def test_matched_rows_never_pool_two_models(self):
        groups = scorecard.group(self.records())
        self.assertEqual(len(groups), 2)
        rates = sorted(scorecard.summarize(v)["decode"]["median"] for v in groups.values())
        self.assertEqual(rates, [30.0, 160.0])
        self.assertEqual({key[0] for key in groups}, {"flint-int4@3b93f9053b53", "inkling-2-mini@ed40e7da625f"})

    def test_best_useful_ranks_within_a_model_only(self):
        document = scorecard.render(self.records(engine_for_inkling="other-engine"), [], [], 0)
        best = document.split("## Best useful", 1)[1].split("## Natural termination", 1)[0]
        rows = [line for line in best.splitlines() if line.startswith("| flint") or line.startswith("| inkling")]
        self.assertEqual(len(rows), 2, best)
        self.assertTrue(all(line.rstrip().endswith("| — |") for line in rows), best)
        matched = document.split("## Matched", 1)[1].split("###", 1)[0]
        self.assertNotIn("95.0", matched)

    def test_natural_termination_keys_carry_the_model(self):
        records = self.records()
        for record in records:
            record["natural_termination"] = {"completion": "stopped", "finish_reasons": ["stop"], "tokens": 10}
        self.assertEqual(len(scorecard.natural(records)), 2)


class PairingIdentity(unittest.TestCase):
    def test_integer_pair_numbers_do_not_join_unrelated_runs(self):
        contract = make_contract()
        trials = [trial("run-a", "baseline", 100.0, contract=contract),
                  trial("run-b", "candidate", 120.0, contract=contract)]
        self.assertEqual(speedups(trials), [])
        self.assertEqual(len(rejected(trials)), 2)
        old_scorecard = legacy("scorecard", "campaign/scorecard.py")
        old_rows = old_scorecard.paired_comparisons(old_scorecard.group(trials))
        self.assertEqual(len(old_rows), 1, "the base revision must pair these unrelated runs")
        self.assertAlmostEqual(old_rows[0]["geomean"], 100.0 / 120.0, places=6)

    def test_different_models_never_pair(self):
        trials = [trial("run", "baseline", 100.0, contract=make_contract(key="flint-int4")),
                  trial("run", "candidate", 200.0, contract=make_contract(key="inkling-2-mini"))]
        self.assertEqual(speedups(trials), [])
        self.assertIn("model_key", rejected(trials)[0]["reason"])

    def test_different_prompt_hashes_never_pair(self):
        other = {"name": PROMPT["name"], "sha256": "b" * 64}
        trials = [trial("run", "baseline", 100.0, contract=make_contract()),
                  trial("run", "candidate", 200.0, contract=make_contract(), prompt=other)]
        self.assertEqual(speedups(trials), [])
        self.assertIn("prompt counterparts differ", rejected(trials)[0]["reason"])

    def test_different_prompt_names_never_pair(self):
        other = {"name": "prose-1024", "sha256": PROMPT["sha256"]}
        trials = [trial("run", "baseline", 100.0, contract=make_contract()),
                  trial("run", "candidate", 200.0, contract=make_contract(), prompt=other)]
        self.assertEqual(speedups(trials), [])
        self.assertIn("prompt counterparts differ", rejected(trials)[0]["reason"])

    def test_role_direction_is_candidate_over_baseline(self):
        trials = [trial("run", "baseline", 100.0, contract=make_contract(), engine="a-baseline"),
                  trial("run", "candidate", 200.0, contract=make_contract(), engine="z-candidate")]
        rows = speedups(trials)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["speedup"], 2.0, places=6)
        old_scorecard = legacy("scorecard", "campaign/scorecard.py")
        old_rows = old_scorecard.paired_comparisons(old_scorecard.group(trials))
        self.assertAlmostEqual(old_rows[0]["geomean"], 0.5, places=6,
                               msg="the base revision's baseline/candidate direction is the reciprocal here")

    def test_role_comes_from_the_record_not_the_engine_name(self):
        trials = [trial("run", "baseline", 200.0, contract=make_contract(), engine="z-baseline"),
                  trial("run", "candidate", 100.0, contract=make_contract(), engine="a-candidate")]
        rows = speedups(trials)
        self.assertAlmostEqual(rows[0]["speedup"], 0.5, places=6)
        old_scorecard = legacy("scorecard", "campaign/scorecard.py")
        old_rows = old_scorecard.paired_comparisons(old_scorecard.group(trials))
        self.assertAlmostEqual(old_rows[0]["geomean"], 0.5, places=6,
                               msg="the base revision sorts engine names, so 'a-candidate' becomes the baseline")

    def test_same_binary_keeps_roles_apart(self):
        trials = [trial("run", "baseline", 100.0, contract=make_contract()),
                  trial("run", "candidate", 100.0, contract=make_contract())]
        rows = speedups(trials)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["speedup"], 1.0, places=6)


class IncompatibleConfiguration(unittest.TestCase):
    def assert_rejected(self, candidate_contract, needle):
        trials = [trial("run", "baseline", 100.0, contract=make_contract()),
                  trial("run", "candidate", 200.0, contract=candidate_contract)]
        self.assertEqual(speedups(trials), [])
        reasons = " ".join(row["reason"] for row in rejected(trials))
        self.assertIn(needle, reasons)

    def test_model_key(self):
        self.assert_rejected(make_contract(key="inkling-2-mini"), "model_key")

    def test_model_revision(self):
        contract = make_contract()
        contract["model_revision"] = "other"
        self.assert_rejected(contract, "model_revision")

    def test_quantization(self):
        contract = make_contract()
        contract["quantization"] = "flint-int4:fp8"
        self.assert_rejected(contract, "quantization")

    def test_host(self):
        contract = make_contract()
        contract["host_signature"] = dict(contract["host_signature"], product="Apple M4")
        self.assert_rejected(contract, "host_signature")

    def test_context_policy(self):
        contract = make_contract()
        contract["context_policy"]["served_model"] = "flint-int4@other"
        self.assert_rejected(contract, "context_policy")

    def test_output_policy(self):
        contract = make_contract()
        contract["output_policy"]["max_tokens"] = 64
        self.assert_rejected(contract, "output_policy")

    def test_cache_policy(self):
        contract = make_contract()
        contract["cache_policy"]["prompt_cache_gib"] = 0
        self.assert_rejected(contract, "cache_policy")

    def test_acceleration_mode(self):
        contract = make_contract()
        contract["acceleration"]["mode"] = "copy"
        self.assert_rejected(contract, "acceleration")

    def test_missing_contract_field_is_not_a_match(self):
        contract = make_contract()
        contract["output_policy"] = None
        self.assert_rejected(contract, "output_policy not recorded")

    def test_observed_acceleration_must_match_the_declared_mode(self):
        contract = make_contract()
        baseline = trial("run", "baseline", 100.0, contract=contract)
        candidate = trial("run", "candidate", 200.0, contract=contract)
        candidate["acceleration_observed"] = {"declared": "ar", "drafted_tokens": 5}
        self.assertEqual(speedups([baseline, candidate]), [])
        self.assertIn("drafted 5 tokens", rejected([baseline, candidate])[0]["reason"])

    def test_observed_cache_policy_must_match_the_declared_policy(self):
        contract = make_contract()
        baseline = trial("run", "baseline", 100.0, contract=contract)
        candidate = trial("run", "candidate", 200.0, contract=contract)
        candidate["cache_observed"] = {"prompt_cache_gib": 0, "cold_cached_tokens": 0}
        self.assertEqual(speedups([baseline, candidate]), [])
        self.assertIn("cache observed", rejected([baseline, candidate])[0]["reason"])

    def test_legacy_records_without_a_contract_are_not_promoted(self):
        def legacy_trial(role, tps):
            return {"kind": "trial", "valid": True, "run_id": "old", "pair": 1, "mode": "ar", "cell": 1024,
                    "prompt_cache_gib": 2, "engine": f"tensorfold-native-{role}",
                    "binary": {"role": role, "sha256": "sha"},
                    "prompt": {"name": PROMPT["name"], "sha256": PROMPT["sha256"]},
                    "requests": {"cold": {"decode_tps": tps}}}
        trials = [legacy_trial("baseline", 100.0), legacy_trial("candidate", 200.0)]
        self.assertEqual(speedups(trials), [])
        self.assertIn("not recorded", rejected(trials)[0]["reason"])
        old_scorecard = legacy("scorecard", "campaign/scorecard.py")
        self.assertEqual(len(old_scorecard.paired_comparisons(old_scorecard.group(trials))), 1)


class Counterparts(unittest.TestCase):
    def test_missing_candidate_counterpart(self):
        trials = [trial("run", "baseline", 100.0, contract=make_contract())]
        self.assertEqual(speedups(trials), [])
        self.assertIn("missing candidate counterpart", rejected(trials)[0]["reason"])

    def test_missing_baseline_counterpart(self):
        trials = [trial("run", "candidate", 100.0, contract=make_contract())]
        self.assertIn("missing baseline counterpart", rejected(trials)[0]["reason"])

    def test_duplicate_baseline_counterpart(self):
        contract = make_contract()
        trials = [trial("run", "baseline", 100.0, contract=contract),
                  trial("run", "baseline", 110.0, contract=contract),
                  trial("run", "candidate", 200.0, contract=contract)]
        self.assertEqual(speedups(trials), [])
        self.assertIn("duplicate baseline prompt counterpart", rejected(trials)[0]["reason"])

    def test_missing_prompt_hash(self):
        contract = make_contract()
        trials = [trial("run", "baseline", 100.0, contract=contract, prompt={"name": "code-1024", "sha256": None}),
                  trial("run", "candidate", 200.0, contract=contract)]
        self.assertIn("no prompt hash", rejected(trials)[0]["reason"])


class Uncertainty(unittest.TestCase):
    def test_unit_of_uncertainty_is_the_process_pair(self):
        prompts = [{"name": f"code-{n}", "sha256": chr(ord("a") + n) * 64} for n in range(3)]
        contract = make_contract(prompts=prompts)
        trials = []
        for prompt in prompts:
            trials.append(trial("run", "baseline", 100.0, contract=contract, prompt=prompt))
            trials.append(trial("run", "candidate", 200.0, contract=contract, prompt=prompt))
        rows = speedups(trials)
        self.assertEqual(rows[0]["pairs"], 1)
        self.assertIsNone(rows[0]["low"], "one process pair cannot carry an interval")
        self.assertFalse(rows[0]["repeatability"])
        self.assertEqual(rows[0]["sessions"], 1)

    def test_two_pairs_in_two_sessions_give_a_descriptive_interval(self):
        contract = make_contract()
        trials = []
        for run, session, tps in (("run-1", "s1", 200.0), ("run-2", "s2", 220.0)):
            trials.append(trial(run, "baseline", 100.0, contract=contract, session=session, experiment="exp"))
            trials.append(trial(run, "candidate", tps, contract=contract, session=session, experiment="exp"))
        rows = speedups(trials)
        self.assertEqual(rows[0]["pairs"], 2)
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertIsNotNone(rows[0]["low"])
        self.assertLess(rows[0]["low"], rows[0]["speedup"])
        self.assertGreater(rows[0]["high"], rows[0]["speedup"])
        self.assertFalse(rows[0]["repeatability"], "the six-pair protocol is required for repeatability")

    def test_six_pairs_across_two_sessions_are_measured(self):
        contract = make_contract()
        trials = []
        for index in range(6):
            session = f"session-{index % 2}"
            run = f"run-{index}"
            trials.append(trial(run, "baseline", 100.0, contract=contract, pair=index + 1, session=session,
                                experiment="exp"))
            trials.append(trial(run, "candidate", 110.0, contract=contract, pair=index + 1, session=session,
                                experiment="exp"))
        rows = speedups(trials)
        self.assertEqual(rows[0]["pairs"], 6)
        self.assertEqual(rows[0]["sessions"], 2)
        self.assertTrue(rows[0]["repeatability"])

    def test_old_interval_puts_a_positive_rate_below_zero(self):
        old = legacy("scorecard", "campaign/scorecard.py")
        summary = old.stats([100.0, 110.0, 400.0])
        self.assertEqual(summary["median"], 110.0)
        self.assertLess(summary["lo"], 0.0,
                        "a mean standard error added to a median can put a strictly positive rate below zero")
        measured = scorecard.measure(
            [{"pairing": {"process_pair": f"p{n}", "session": f"s{n}"},
              "requests": {"cold": {"decode_tps": value}}} for n, value in enumerate((100.0, 110.0, 400.0))],
            lambda r: r["requests"]["cold"]["decode_tps"])
        self.assertGreaterEqual(measured["min"], 100.0)
        self.assertLessEqual(measured["max"], 400.0)

    def test_descriptive_range_reports_min_and_max(self):
        summary = scorecard.measure([{"pairing": {"process_pair": "p1", "session": "s"},
                                      "requests": {"cold": {"decode_tps": value}}} for value in (1.0, 5.0)],
                                    lambda r: r["requests"]["cold"]["decode_tps"])
        self.assertEqual((summary["min"], summary["max"]), (1.0, 5.0))
        self.assertEqual(summary["pairs"], 1)


class Corpus(unittest.TestCase):
    manifest = json.loads((CAMPAIGN / "prompts" / "manifest.json").read_text(encoding="utf-8"))

    def test_heldout_selector_returns_the_nine_holdout_prompts(self):
        selected = bench.select(self.manifest, [1024, 4096, 32768], "heldout")
        names = sorted(prompt["name"] for prompts in selected.values() for prompt in prompts)
        self.assertEqual(len(names), 9)
        self.assertTrue(all(name.startswith("heldout-") for name in names))
        self.assertEqual(sorted(selected), [1024, 4096, 32768])
        self.assertEqual([len(selected[cell]) for cell in (1024, 4096, 32768)], [3, 3, 3])

    def test_holdout_alias_matches(self):
        self.assertEqual(bench.select(self.manifest, [1024], "holdout"),
                         bench.select(self.manifest, [1024], "heldout"))

    def test_base_revision_selection_is_empty(self):
        old_bench = legacy("bench", "campaign/bench/bench.py")
        self.assertEqual(old_bench.select(self.manifest, [1024, 4096, 32768], "heldout"), {})

    def test_heldout_selection_hashes_and_coverage(self):
        selected = bench.select(self.manifest, [1024, 4096, 32768], "heldout")
        bench.verify_prompts(selected)

    def test_dev_selector_still_returns_the_dev_prompts(self):
        selected = bench.select(self.manifest, [1024, 4096, 32768], "all")
        self.assertEqual(sum(len(prompts) for prompts in selected.values()), 9)
        self.assertTrue(all(prompt["set"] == "dev" for prompts in selected.values() for prompt in prompts))


class Modes(unittest.TestCase):
    def test_combined_label_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            bench.mode_args("combined")
        self.assertIn("unsupported", str(caught.exception))

    def test_drafter_and_copy_have_distinct_policies(self):
        ar_flags, ar_drafter, ar_copy = bench.mode_args("ar")
        copy_flags, copy_drafter, copy_copy = bench.mode_args("copy")
        draft_flags, draft_drafter, draft_copy = bench.mode_args("drafter")
        self.assertEqual(ar_flags, ["--no-drafts"])
        self.assertEqual((ar_drafter, ar_copy), (False, False))
        self.assertEqual((copy_flags, copy_drafter, copy_copy), ([], False, True))
        self.assertEqual((draft_flags, draft_drafter, draft_copy), ([], True, True))

    def test_base_revision_gave_drafter_and_combined_identical_arguments(self):
        old_bench = legacy("bench", "campaign/bench/bench.py")
        self.assertEqual(old_bench.mode_args("drafter"), old_bench.mode_args("combined"))
        self.assertEqual(old_bench.mode_args("drafter"), ([], True))

    def test_ar_problems_when_it_drafts(self):
        self.assertTrue(bench.acceleration_problems("ar", {"drafted_tokens": 5, "verification_rounds": 100}))

    def test_drafter_problems_without_counters_or_drafts(self):
        self.assertTrue(bench.acceleration_problems("drafter", {"speculative": None, "drafted_tokens": 0}))
        self.assertTrue(bench.acceleration_problems("drafter", {"speculative": {"rounds": 3}, "drafted_tokens": 0}))
        self.assertEqual(bench.acceleration_problems("drafter", {"speculative": {"rounds": 3}, "drafted_tokens": 7}),
                         [])

    def test_cold_cache_policy(self):
        self.assertTrue(bench.cache_problems(0, {"cached_tokens": 10, "cache_state": "reused"}))
        self.assertEqual(bench.cache_problems(0, {"cached_tokens": 0, "cache_state": "cold"}), [])
        self.assertEqual(bench.cache_problems(2, {"cached_tokens": 0, "cache_state": "cold"}), [])

    def test_cell_cells_select_the_decision_eight_cache(self):
        self.assertEqual(bench.parse_cells("1024,cold:4096,~32768"), [(1024, 2), (4096, 0), (32768, 0)])


def run_script(script, *args):
    return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True)


def write_json(directory, name, document):
    path = Path(directory) / name
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def quality_result(cases, *, set_name="dev", temperature=0, reasons=None):
    reasons = reasons or {}
    return {"schema": 1, "set": set_name, "protocol": {"temperature": temperature, "seed": 1234, "max_tokens": 8192},
            "suite": {"cases": len(cases)}, "invalid_reasons": [],
            "cases": [{"id": case_id, "kind": "code", "passed": passed, "reason": reasons.get(case_id, "ok")}
                      for case_id, passed in cases],
            "totals": {"cases": len(cases), "passed": sum(1 for _, passed in cases if passed),
                       "failed": sum(1 for _, passed in cases if not passed)}}


def base_source(relative: str) -> str:
    return subprocess.run(["git", "show", f"{BASE_REV}:{relative}"], cwd=REPO, capture_output=True,
                          check=True).stdout.decode()


def legacy_quality_exit(record):
    """The base revision returned 0 when only scoring failed, so its exit code was not a qualification."""
    request_failures = sum(1 for case in record["cases"] if case["reason"].startswith("request failed"))
    return 1 if request_failures else 0


def legacy_pi_exit(record):
    """The base revision returned 0 without any checker outcome entering the exit code."""
    totals = record["totals"]
    return 0 if totals["completed"] == totals["tasks"] and totals["parse_errors"] == 0 else 1


class QualityCompletion(unittest.TestCase):
    def test_identical_run_qualifies(self):
        cases = [(f"case-{n}", True) for n in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(cases))
            candidate = write_json(directory, "candidate.json", quality_result(cases))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_one_lost_case_is_allowed(self):
        base = [(f"case-{n}", True) for n in range(12)]
        cand = [("case-0", False)] + base[1:]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(base))
            candidate = write_json(directory, "candidate.json", quality_result(cand))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_two_lost_cases_are_refused(self):
        base = [(f"case-{n}", True) for n in range(12)]
        cand = [("case-0", False), ("case-1", False)] + base[2:]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(base))
            candidate = write_json(directory, "candidate.json", quality_result(cand))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 1)
        self.assertIn("2 cases lost", result.stderr)

    def test_base_revision_exit_ignored_scoring_failures(self):
        source = base_source("campaign/quality/run.py")
        self.assertIn("return 1 if failures else 0", source)
        self.assertNotIn("complete coverage", source)
        self.assertTrue((CAMPAIGN / "quality" / "qualify.py").is_file(),
                        "qualification must live in a separate command")

    def test_zero_exit_after_a_scoring_failure_is_not_a_qualification(self):
        cases = [(f"case-{n}", True) for n in range(12)]
        scored = quality_result(cases, reasons={"case-3": "scoring failed: KeyError: 'x'"})
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(cases))
            candidate = write_json(directory, "candidate.json", scored)
            self.assertEqual(legacy_quality_exit(scored), 0)
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid run", result.stderr)

    def test_incomplete_coverage_is_refused(self):
        base = [(f"case-{n}", True) for n in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(base))
            candidate = write_json(directory, "candidate.json", quality_result(base[:10]))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not scored", result.stderr)

    def test_protocol_drift_is_refused(self):
        base = [(f"case-{n}", True) for n in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(base))
            candidate = write_json(directory, "candidate.json", quality_result(base, temperature=0.7))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 1)
        self.assertIn("temperature", result.stderr)

    def test_baseline_need_not_score_one_hundred_percent(self):
        base = [(f"case-{n}", n < 9) for n in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            baseline = write_json(directory, "baseline.json", quality_result(base))
            candidate = write_json(directory, "candidate.json", quality_result(base))
            result = run_script(CAMPAIGN / "quality" / "qualify.py",
                                "--result", str(candidate), "--baseline", str(baseline))
        self.assertEqual(result.returncode, 0, result.stderr)


def pi_result(tasks, *, completed=None, parse_errors=0, owned=True, model_id="flint",
              identity_ids=None, base="http://127.0.0.1:53210"):
    completed = completed or {}
    outcomes = list(tasks)
    tasks = [{"task": name, "completed": completed.get(name, True),
              "exit_code": 0, "timed_out": False,
              "checker": {"pass": passed, "detail": ""}, "parse_errors": {"total": 0}}
             for name, passed in outcomes]
    return {"schema": 1, "model": "flint", "run_id": "run",
            "pi": {"provider": "tensorfold", "tools": "read,bash,edit,write", "model_id": model_id,
                   "path": "/pi"},
            "server": {"base": base, "port": 53210, "owned": owned,
                       "binary": {"path": "/bin/tensorfold-native", "sha256": "sha"},
                       "identity": {"model_ids": identity_ids if identity_ids is not None else [model_id]}},
            "agent": {"rendered": {"providers": {"tensorfold": {"baseUrl": f"{base}/v1"}}}},
            "tasks": tasks,
            "totals": {"tasks": len(tasks), "completed": sum(1 for task in tasks if task["completed"]),
                       "checker_passed": sum(1 for _, passed in outcomes if passed), "parse_errors": parse_errors},
            "completion": {"complete": all(task["completed"] for task in tasks)}}


class PiCompletion(unittest.TestCase):
    names = ["add-function", "bugfix-dedupe", "bugfix-intervals", "bugfix-window"]

    def qualify(self, candidate, baseline=None):
        with tempfile.TemporaryDirectory() as directory:
            result = write_json(directory, "candidate.json", candidate)
            args = ["--result", str(result)]
            if baseline is not None:
                args += ["--baseline", str(write_json(directory, "baseline.json", baseline))]
            return run_script(CAMPAIGN / "pi" / "qualify.py", *args)

    def test_complete_run_with_all_checkers_qualifies(self):
        result = self.qualify(pi_result([(name, True) for name in self.names]))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_base_revision_exit_ignored_checker_outcomes(self):
        source = base_source("campaign/pi/run.py")
        self.assertIn('return 0 if totals["completed"] == totals["tasks"] and totals["parse_errors"] == 0 else 1',
                      source)
        exit_line = next(line for line in source.splitlines() if line.strip().startswith("return 0 if totals"))
        self.assertNotIn("checker_passed", exit_line)

    def test_base_revision_recorded_the_requested_binary_for_a_reused_server(self):
        source = base_source("campaign/pi/run.py")
        self.assertIn("SERVER_PORT = 8888", source)
        self.assertIn("if healthy(base):\n        return None, False", source)
        self.assertIn('"binary": {"path": str(binary), "sha256": harness.sha256_file(binary)}', source)

    def test_zero_exit_with_a_failed_checker_is_not_a_qualification(self):
        candidate = pi_result([(name, name != "bugfix-window") for name in self.names])
        self.assertEqual(legacy_pi_exit(candidate), 0)
        result = self.qualify(candidate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("bugfix-window", result.stderr)

    def test_one_lost_checker_is_allowed_against_the_baseline(self):
        baseline = pi_result([(name, True) for name in self.names])
        candidate = pi_result([(name, name != "bugfix-window") for name in self.names])
        result = self.qualify(candidate, baseline)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_two_lost_checkers_are_refused(self):
        baseline = pi_result([(name, True) for name in self.names])
        candidate = pi_result([(name, name not in ("bugfix-window", "bugfix-dedupe")) for name in self.names])
        result = self.qualify(candidate, baseline)
        self.assertEqual(result.returncode, 1)
        self.assertIn("2 checker cases lost", result.stderr)

    def test_incomplete_run_is_refused(self):
        candidate = pi_result([(name, True) for name in self.names], completed={"bugfix-window": False})
        result = self.qualify(candidate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("did not complete", result.stderr)

    def test_unowned_server_cannot_be_attributed(self):
        result = self.qualify(pi_result([(name, True) for name in self.names], owned=False))
        self.assertEqual(result.returncode, 1)
        self.assertIn("not owned", result.stderr)

    def test_served_identity_must_hold_pis_model(self):
        candidate = pi_result([(name, True) for name in self.names], model_id="inkling",
                              identity_ids=["flint-int4@3b93f9053b53"])
        result = self.qualify(candidate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("does not hold", result.stderr)

    def test_identity_must_be_captured(self):
        candidate = pi_result([(name, True) for name in self.names])
        candidate["server"]["identity"] = {"model_ids": []}
        result = self.qualify(candidate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("not captured", result.stderr)

    def test_rendered_models_must_point_at_the_campaign_server(self):
        candidate = pi_result([(name, True) for name in self.names])
        candidate["agent"]["rendered"]["providers"]["tensorfold"]["baseUrl"] = "http://127.0.0.1:8888/v1"
        result = self.qualify(candidate)
        self.assertEqual(result.returncode, 1)
        self.assertIn("models.json points", result.stderr)

    def test_parse_errors_refuse(self):
        result = self.qualify(pi_result([(name, True) for name in self.names], parse_errors=2))
        self.assertEqual(result.returncode, 1)
        self.assertIn("parse errors", result.stderr)


class PiServerIdentity(unittest.TestCase):
    def test_wrong_owner_is_refused(self):
        problems = pi_run.identity_problems(
            {"model_ids": ["other"], "models": [{"id": "other", "owned_by": "llama.cpp"}]},
            binary_sha256="sha", model_key="flint-int4", expected_model=None)
        self.assertTrue(any("not owned" in problem for problem in problems))

    def test_wrong_model_is_refused(self):
        problems = pi_run.identity_problems(
            {"model_ids": ["inkling-2-mini@x"], "models": [{"owned_by": "tensorfold"}]},
            binary_sha256="sha", model_key="flint-int4", expected_model=None)
        self.assertTrue(any("do not name model" in problem for problem in problems))

    def test_matching_identity_has_no_problems(self):
        problems = pi_run.identity_problems(
            {"model_ids": ["flint-int4@3b93f9053b53"], "models": [{"owned_by": "tensorfold"}]},
            binary_sha256="sha", model_key="flint-int4", expected_model="flint-int4@3b93f9053b53")
        self.assertEqual(problems, [])

    def test_render_models_rewrites_endpoint_and_model(self):
        template = CAMPAIGN / "pi" / "agent" / "models.json"
        rendered = pi_run.render_models(template, "http://127.0.0.1:53210", "flint-int4@3b93f9053b53")
        provider = rendered["providers"]["tensorfold"]
        self.assertEqual(provider["baseUrl"], "http://127.0.0.1:53210/v1")
        self.assertEqual([model["id"] for model in provider["models"]], ["flint-int4@3b93f9053b53"])
        self.assertNotIn("127.0.0.1:8888", json.dumps(rendered))


class Memory(unittest.TestCase):
    def record(self, *, peak, lifetime, samples):
        entry = trial("run", "baseline", 100.0, contract=make_contract())
        entry["footprint"] = {"peak_bytes": peak, "process_lifetime_peak_bytes": lifetime,
                              "ready_bytes": 1, "steady_bytes": 1, "device_peak_bytes": None,
                              "samples": samples, "source": "test"}
        return entry

    def test_lifetime_peak_is_exposed_when_samples_are_empty(self):
        entry = self.record(peak=None, lifetime=24 * 2**30, samples=0)
        document = scorecard.render([entry], [], ["run.jsonl"], 0)
        self.assertIn("| 24.00 |", document, "lifetime peak must appear without sampled values")
        self.assertIn("Sampled request peak GiB", document)
        self.assertIn("Lifetime peak GiB", document)

    def test_sampled_peak_and_lifetime_peak_are_separate_columns(self):
        entry = self.record(peak=22 * 2**30, lifetime=25 * 2**30, samples=10)
        row = next(line for line in scorecard.render([entry], [], ["run.jsonl"], 0).splitlines()
                   if "| tensorfold-native-baseline |" in line)
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        self.assertEqual(cells[12], "22.00")
        self.assertEqual(cells[13], "25.00")

    def test_render_is_deterministic(self):
        entry = self.record(peak=22 * 2**30, lifetime=25 * 2**30, samples=10)
        self.assertEqual(scorecard.render([entry], [], ["run.jsonl"], 0),
                         scorecard.render([entry], [], ["run.jsonl"], 0))


class ExistingRecords(unittest.TestCase):
    def test_committed_records_keep_their_recorded_settings(self):
        trials, unavailable, files, malformed = scorecard.load()
        self.assertTrue(files)
        self.assertEqual(malformed, 0)
        for record in trials:
            pairing = scorecard.pairing_of(record)
            self.assertEqual(pairing["role"], (record.get("binary") or {}).get("role"))
        for record in trials:
            contract = scorecard.pairing_of(record)["contract"]
            self.assertEqual(contract["cache_policy"]["prompt_cache_gib"], record["prompt_cache_gib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
