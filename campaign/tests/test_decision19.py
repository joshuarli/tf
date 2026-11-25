"""Decision 19's executable gate reproduces PH2's recorded verdict and fails each rule it states."""

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parent.parent
PH2 = CAMPAIGN / "kernels" / "inkling-ph2-default-runs"
spec = importlib.util.spec_from_file_location("decision19", CAMPAIGN / "quality" / "decision19.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["decision19"] = gate
spec.loader.exec_module(gate)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Decision19(unittest.TestCase):
    exact_q, variant_q = load(PH2 / "quality/off/result.json"), load(PH2 / "quality/half/result.json")
    exact_pi, variant_pi = load(PH2 / "pi/off.json"), load(PH2 / "pi/half.json")
    exact_o = gate.oracle("2863/3015:0.0194994")
    variant_o = gate.oracle(str(PH2 / "oracle-half.txt"))

    def run_gate(self, **changes) -> dict:
        args = {"exact_q": self.exact_q, "variant_q": self.variant_q, "exact_o": self.exact_o,
                "variant_o": self.variant_o, "exact_pi": self.exact_pi, "variant_pi": self.variant_pi} | changes
        return gate.verdict(**args)

    def test_ph2_passes_and_names_its_lost_case(self):
        result = self.run_gate()
        self.assertTrue(result["pass"], result["problems"])
        q = result["quality"]
        self.assertEqual((q["exact_passed"], q["variant_passed"], q["scored"]), (28, 29, 32))
        self.assertEqual(q["lost"], ["retrieval-t01"])
        self.assertEqual(sorted(q["gained"]), ["code-c15", "retrieval-t04"])
        self.assertEqual((len(q["runaways"]["exact"]), len(q["runaways"]["variant"])), (3, 2))
        self.assertEqual((len(q["unsupported"]), len(q["excluded"])), (12, 4))
        self.assertEqual((result["pi"]["exact"], result["pi"]["variant"]), (7, 7))
        self.assertTrue(any("retrieval-t01" in n for n in result["notes"]))

    def test_a_second_lost_case_fails(self):
        variant = copy.deepcopy(self.variant_q)
        case = next(c for c in variant["cases"] if c["id"] == "code-c01")
        case.update(passed=False, reason="tests failed")
        self.assertFalse(self.run_gate(variant_q=variant)["pass"])

    def test_one_more_runaway_than_exact_fails(self):
        variant = copy.deepcopy(self.variant_q)
        for cid in ("reasoning-gsm8k-3", "reasoning-gsm8k-23"):
            next(c for c in variant["cases"] if c["id"] == cid).update(
                passed=False, reason="truncated at 8192 tokens: no answer", finish_reason="length")
        problems = self.run_gate(variant_q=variant)["problems"]
        self.assertTrue(any("runaways" in p for p in problems))

    def test_oracle_bounds_fail(self):
        self.assertFalse(self.run_gate(variant_o={"top1": 2863, "rows": 3015, "kl": 0.0194994 * 1.11})["pass"])
        self.assertFalse(self.run_gate(variant_o={"top1": 2846, "rows": 3015, "kl": 0.0194994})["pass"])

    def test_a_case_missing_or_unsupported_on_one_side_fails(self):
        variant = copy.deepcopy(self.variant_q)
        variant["cases"] = [c for c in variant["cases"] if c["id"] != "code-c02"]
        self.assertTrue(any("one side only" in p for p in self.run_gate(variant_q=variant)["problems"]))
        variant = copy.deepcopy(self.variant_q)
        next(c for c in variant["cases"] if c["id"] == "code-c02").update(
            passed=False, reason="request failed: HTTPError: HTTP Error 400: Bad Request")
        self.assertFalse(self.run_gate(variant_q=variant)["pass"])

    def test_a_worse_pi_result_fails(self):
        pi = copy.deepcopy(self.variant_pi)
        next(t for t in pi["tasks"] if (t.get("checker") or {}).get("pass"))["checker"]["pass"] = False
        self.assertFalse(self.run_gate(variant_pi=pi)["pass"])


if __name__ == "__main__":
    unittest.main()
