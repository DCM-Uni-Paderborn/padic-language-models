import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.corpus import Article
from padic_lm.heldout import inventory, paired_document_bootstrap


class HeldoutTests(unittest.TestCase):
    def test_all_eligible_articles_and_exclusion_inventory(self):
        docs = [Article(i, i + 1, str(i), text, key) for i, (text, key) in
                enumerate([("long", "a"), ("short", "b"), ("long", "a"), ("long", "c"), ("long", "d")])]
        records, selected, ids = inventory(docs, lambda text: list(range(2048 if text == "long" else 7)), {"c"})
        self.assertEqual([r["start_row"] for r in selected], [0, 4])
        self.assertEqual([r["exclusion_reasons"] for r in records],
                         [[], ["short_article"], ["repeated_test_exact_text"], ["train_or_validation_exact_text"], []])
        self.assertEqual(ids.shape, (2, 2048))
        self.assertTrue(np.array_equal(ids[0], np.arange(2048)))

    def test_no_eligible_prefixes_has_explicit_empty_shape(self):
        records, selected, ids = inventory([Article(0, 1, "short", "x", "a")], lambda _: [1], set())
        self.assertEqual(len(records), 1)
        self.assertEqual(selected, [])
        self.assertEqual(ids.shape, (0, 2048))

    def test_bootstrap_recomputes_token_weighted_sums(self):
        # Unequal documents: averaging their NLLs would produce the wrong result.
        bounds, samples = paired_document_bootstrap([12, 24], [10, 20], [2, 8],
                                                     np.array([[0, 0], [0, 1], [1, 1]]))
        np.testing.assert_allclose(samples, [1, .6, .5], rtol=0, atol=1e-15)
        self.assertAlmostEqual(bounds["paired_delta_nll"], .6)
        self.assertAlmostEqual(bounds["lower_025"], .505)
        self.assertAlmostEqual(bounds["upper_975"], .98)

    def test_zero_difference_and_invalid_units(self):
        bounds, samples = paired_document_bootstrap([3, 4], [3, 4], [1, 2], np.array([[1, 0]]))
        self.assertEqual(bounds, {"paired_delta_nll": 0., "lower_025": 0., "upper_975": 0.})
        self.assertEqual(samples.tolist(), [0.])
        for counts, draws in [([0, 1], [[0, 1]]), ([1, 1], [[0, 2]]), ([1, 1], [[0., 1.]]), ([1, 1], [[0]])]:
            with self.assertRaises(ValueError):
                paired_document_bootstrap([3, 4], [3, 4], counts, np.array(draws))


if __name__ == "__main__":
    unittest.main()
