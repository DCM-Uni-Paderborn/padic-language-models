import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_heldout_quality import comparison_plan, family_decisions
from audit_heldout_statistics import frequency_bootstrap, linear_order_statistic


class StatisticsTests(unittest.TestCase):
    def primary_fixture(self):
        return [dict(zip(("role", "family", "candidate", "reference", "endpoint"), identity), upper_975=.004)
                for identity in comparison_plan() if identity[0] == "primary"]

    def test_intersection_requires_every_seed_comparator_and_endpoint(self):
        rows = self.primary_fixture()
        self.assertTrue(all(r["passes_frozen_screen"] for r in family_decisions(rows, 40, .01).values()))
        # A single affected-target comparison of the last seed equals the margin.
        rows[17]["upper_975"] = .01
        decision = family_decisions(rows, 40, .01)
        self.assertFalse(decision["p2"]["passes_frozen_screen"])
        self.assertEqual(decision["p2"]["comparisons_with_upper_below_margin"], 17)
        self.assertTrue(decision["p3"]["passes_frozen_screen"])

    def test_too_few_documents_cannot_classify_and_missing_duplicate_rows_fail(self):
        rows = self.primary_fixture()
        self.assertFalse(any(r["passes_frozen_screen"] for r in family_decisions(rows, 39, .01).values()))
        for invalid in (rows[:-1], rows[:-1] + [rows[-2]], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                family_decisions(invalid, 40, .01)

    def test_frequency_statistic_preserves_unequal_document_weights(self):
        frequencies = np.array([[2, 0], [1, 1], [0, 2]])
        estimate, samples = frequency_bootstrap([12, 24], [10, 20], [2, 8], frequencies)
        np.testing.assert_allclose(samples, [1, .6, .5], rtol=0, atol=1e-15)
        self.assertAlmostEqual(estimate["paired_delta_nll"], .6)
        self.assertAlmostEqual(estimate["lower_025"], .505)
        self.assertAlmostEqual(estimate["upper_975"], .98)

    def test_explicit_interpolation_has_expected_order_statistic(self):
        self.assertEqual(linear_order_statistic([7, 1, 3, 2], .5), 2.5)
        self.assertEqual(linear_order_statistic([7, 1, 3, 2], 0), 1)
        self.assertEqual(linear_order_statistic([7, 1, 3, 2], 1), 7)


if __name__ == "__main__":
    unittest.main()
