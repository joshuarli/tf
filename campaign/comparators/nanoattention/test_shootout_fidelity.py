import math
import unittest

try:
    from shootout import clean_diagnostics, fidelity_row
except ModuleNotFoundError:
    fidelity_row = None


@unittest.skipIf(fidelity_row is None, 'numerical comparison dependencies unavailable')
class FidelityTests(unittest.TestCase):
    def test_redaction_keeps_the_rest_of_the_gate_receipt(self):
        self.assertEqual(clean_diagnostics('busy\nprocess command=private\nquiet'),
                         'busy\n[unrelated process details removed]\nquiet')

    def test_confident_disagreement_and_kl_use_the_oracle_distribution(self):
        confident, agreement, kl = fidelity_row([math.log(3), 0], [0, 0])
        self.assertTrue(confident)
        self.assertTrue(agreement)
        self.assertAlmostEqual(kl, .75 * math.log(1.5) + .25 * math.log(.5))
        confident, agreement, kl = fidelity_row([0, math.log(3)], [1, 0])
        self.assertTrue(confident)
        self.assertFalse(agreement)
        self.assertGreater(kl, 0)

    def test_unconfident_and_nonfinite_rows(self):
        self.assertFalse(fidelity_row([0, .01], [0, 0])[0])
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            fidelity_row([0, float('nan')], [0, 0])
