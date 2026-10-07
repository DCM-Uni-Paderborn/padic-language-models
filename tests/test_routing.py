"""Mathematics, causal selection, calibration and independent tree controls."""

import sys
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.routing import (
    PrefixTree, QuantileCodebook, angular_codes, angular_hyperplanes,
    attention_probabilities, common_padic_depth, digits_at_bit_budget, finite_padic_distances,
    pack_codes, reverse_bits, reverse_digits, select_causal_angular, select_causal_padic,
    select_causal_random, select_from_scores, unpack_codes,
)


class FiniteMetricTests(unittest.TestCase):
    def test_odd_prime_metrics_match_valuation_and_tree_at_mixed_precision(self):
        rng = np.random.default_rng(552)
        for prime in (2, 3, 5):
            layout = digits_at_bit_budget(prime, 8, 2)
            keys = np.stack([rng.integers(0, prime ** count, 43) for count in layout], axis=1).astype(np.uint16)
            tree = PrefixTree(2, layout, prime=prime)
            for position, key in enumerate(keys):
                tree.append(key)
                query = np.array([rng.integers(0, prime ** count) for count in layout], dtype=np.uint16)
                distances = finite_padic_distances(query, keys[:position + 1], layout, prime=prime)
                for index, other in enumerate(keys[:position + 1]):
                    scalar_distances = []
                    for a, b in zip(query, other):
                        difference = abs(int(a) - int(b))
                        valuation = 0
                        if difference:
                            while difference % prime == 0:
                                difference //= prime
                                valuation += 1
                        scalar_distances.append(0.0 if int(a) == int(b) else prime ** -valuation)
                    self.assertEqual(float(distances[index]), max(scalar_distances))
                for budget in (1, 9, 100):
                    for recent in (0, 3, 12):
                        selected = select_causal_padic(query, keys, position, budget,
                                                      digits=layout, prime=prime, recent_window=recent)
                        np.testing.assert_array_equal(selected, tree.select(query, budget, recent))

    def test_odd_prime_packing_and_digit_reversal_use_same_stored_bytes(self):
        rng = np.random.default_rng(885)
        expected = {2: ((4, 4), 256), 3: ((3, 2), 243), 5: ((2, 1), 125)}
        for prime, (layout, alphabet) in expected.items():
            self.assertEqual(digits_at_bit_budget(prime, 8, 2), layout)
            codes = np.stack([rng.integers(0, prime ** count, 101) for count in layout], axis=1).astype(np.uint16)
            np.testing.assert_array_equal(reverse_digits(reverse_digits(codes, layout, prime), layout, prime), codes)
            packed = pack_codes(codes, layout, prime)
            self.assertEqual(packed.shape, (101, 1))
            np.testing.assert_array_equal(unpack_codes(packed, 2, layout, prime), codes)
            if prime != 2:
                with self.assertRaises(ValueError):
                    unpack_codes(np.array([[alphabet]], dtype=np.uint8), 2, layout, prime)

    def test_distances_agree_with_integer_valuation_and_ultrametric(self):
        codes = np.arange(16, dtype=np.uint16)[:, None]
        matrix = np.stack([finite_padic_distances(code, codes, 4) for code in codes])
        for a in range(16):
            for b in range(16):
                difference = abs(a - b)
                expected = 0.0 if not difference else 2.0 ** -((difference & -difference).bit_length() - 1)
                self.assertEqual(matrix[a, b], expected)
                for c in range(16):
                    self.assertLessEqual(matrix[a, c], max(matrix[a, b], matrix[b, c]))
        np.testing.assert_array_equal(matrix, matrix.T)
        np.testing.assert_array_equal(np.diag(matrix), 0)

    def test_product_metric_and_zero_duplicate_codes(self):
        keys = np.array([[0, 0], [0, 0], [4, 8], [1, 0], [2, 4]], dtype=np.uint16)
        np.testing.assert_array_equal(common_padic_depth(keys[0], keys, 4), [4, 4, 2, 0, 1])
        np.testing.assert_array_equal(finite_padic_distances(keys[0], keys, 4), [0, 0, .25, 1, .5])
        self.assertEqual(float(finite_padic_distances(np.array([1]), np.array([[17]]), 5)[0]), 1 / 16)

    def test_bit_reversal_embeds_coarse_real_paths(self):
        bins = np.arange(16, dtype=np.uint16)[:, None]
        codes = reverse_bits(bins, 4)
        np.testing.assert_array_equal(reverse_bits(codes, 4), bins)
        # Conventional ordered bins 0..7 share the coarse high bit. Their
        # p-adic residues share the low bit after reversal.
        self.assertTrue(np.all((codes[:8] & 1) == 0))
        self.assertTrue(np.all((codes[8:] & 1) == 1))
        self.assertEqual(int(common_padic_depth(codes[0], codes[[1]], 4)[0]), 3)

    def test_packing_roundtrip_and_padding(self):
        rng = np.random.default_rng(71)
        for digits in (1, 3, 4, 8, 16):
            for coordinates in (1, 2, 3):
                codes = rng.integers(0, 1 << digits, size=(2, 3, 7, coordinates), dtype=np.uint16)
                packed = pack_codes(codes, digits)
                self.assertEqual(packed.shape[-1], (coordinates * digits + 7) // 8)
                np.testing.assert_array_equal(unpack_codes(packed, coordinates, digits), codes)
        with self.assertRaises(ValueError):
            unpack_codes(np.array([[128]], dtype=np.uint8), 1, 3)

    def test_invalid_codes_and_precision_rejected(self):
        for values in (np.array([[-1]]), np.array([[16]])):
            with self.assertRaises(ValueError):
                reverse_bits(values, 4)
        with self.assertRaises(TypeError):
            reverse_bits(np.array([[1.5]]), 4)
        with self.assertRaises(ValueError):
            reverse_bits(np.array([[1]]), 17)
        with self.assertRaises(TypeError):
            reverse_bits(np.array([[1]]), True)


class CausalSelectionTests(unittest.TestCase):
    def test_padic_and_independent_tree_match_many_codes_and_budgets(self):
        rng = np.random.default_rng(119)
        for digits, coordinates in ((1, 1), (4, 2), (7, 3)):
            keys = rng.integers(0, 1 << digits, (43, coordinates), dtype=np.uint16)
            queries = rng.integers(0, 1 << digits, (43, coordinates), dtype=np.uint16)
            tree = PrefixTree(coordinates, digits)
            for position in range(len(keys)):
                self.assertEqual(tree.append(keys[position]), position)
                for budget in (1, 3, 9, 64):
                    for recent in (0, 2, 20):
                        actual = select_causal_padic(queries[position], keys, position, budget,
                                                    digits=digits, recent_window=recent)
                        expected = tree.select(queries[position], budget, recent)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(len(actual), min(budget, position + 1))
                        self.assertEqual(len(np.unique(actual)), len(actual))
                        self.assertTrue(np.all(actual <= position))
                        mandatory = min(recent, budget, position + 1)
                        self.assertTrue(set(range(position + 1 - mandatory, position + 1)).issubset(actual))

    def test_future_keys_cannot_change_selection(self):
        keys = np.array([[0], [1], [2], [3], [0], [0]], dtype=np.uint16)
        original = select_causal_padic(np.array([0]), keys, 3, 2, digits=3)
        keys[4:] = 7
        np.testing.assert_array_equal(select_causal_padic(np.array([0]), keys, 3, 2, digits=3), original)
        # Future values need not even be canonical: no forward computation
        # validates or examines that future suffix.
        keys[4:] = 1000
        np.testing.assert_array_equal(select_causal_padic(np.array([0]), keys, 3, 2, digits=3), original)

    def test_zero_duplicate_code_ties_choose_latest_and_early_budget_is_full(self):
        keys = np.zeros((8, 2), dtype=np.uint16)
        tree = PrefixTree(2, 4)
        for key in keys:
            tree.append(key)
        np.testing.assert_array_equal(tree.select(keys[0], 3), [5, 6, 7])
        np.testing.assert_array_equal(select_causal_padic(keys[0], keys, 7, 3, digits=4), [5, 6, 7])
        np.testing.assert_array_equal(select_causal_padic(keys[0], keys, 1, 3, digits=4), [0, 1])

    def test_recent_window_has_priority_and_score_ties_are_deterministic(self):
        np.testing.assert_array_equal(select_from_scores([10, 9, 8, 0, 0], 3, 2), [0, 3, 4])
        np.testing.assert_array_equal(select_from_scores(np.zeros(5), 2), [3, 4])
        np.testing.assert_array_equal(select_from_scores([10, 9, 8, 0, 0], 3, 20), [2, 3, 4])

    def test_seeded_random_and_angular_are_causal_deterministic(self):
        first = select_causal_random(40, 11, seed=17, chunk=3, head=1, recent_window=4)
        np.testing.assert_array_equal(first, select_causal_random(40, 11, seed=17, chunk=3, head=1, recent_window=4))
        self.assertEqual(len(first), 11)
        self.assertTrue(np.all(first <= 40))
        self.assertTrue(set(range(37, 41)).issubset(first))
        planes = angular_hyperplanes(2, 5, 8, 17)
        np.testing.assert_array_equal(planes, angular_hyperplanes(2, 5, 8, 17))
        codes = angular_codes(np.zeros((1, 2, 6, 5)), planes)
        np.testing.assert_array_equal(select_causal_angular(codes[0, 0, 2], codes[0, 0], 2, 2), [1, 2])


class CalibrationAndAttentionTests(unittest.TestCase):
    def test_odd_prime_encoder_fits_mixed_bins_and_canonical_ranges(self):
        rng = np.random.default_rng(239)
        q, k = rng.normal(size=(2, 3, 17, 5)), rng.normal(size=(2, 3, 17, 5))
        for prime in (2, 3, 5):
            book = QuantileCodebook.fit(q, k, prime=prime, code_bit_budget=8, coordinates=2)
            self.assertEqual(book.digits, digits_at_bit_budget(prime, 8, 2))
            codes = book.encode(q)
            for coordinate, count in enumerate(book.digits):
                self.assertTrue(np.all(codes[..., coordinate] < prime ** count))
            np.testing.assert_array_equal(unpack_codes(pack_codes(codes, book.digits, prime), 2, book.digits, prime), codes)
            self.assertTrue(np.all(np.isfinite(book.thresholds)))

    def test_calibration_encoder_reproducible_shared_and_eval_does_not_refit(self):
        rng = np.random.default_rng(812)
        q, k = rng.normal(size=(2, 3, 17, 5)), rng.normal(size=(2, 3, 17, 5))
        for kind in ("pca", "random"):
            book = QuantileCodebook.fit(q, k, coordinates=2, digits=4, projection_kind=kind)
            again = QuantileCodebook.fit(q, k, coordinates=2, digits=4, projection_kind=kind)
            np.testing.assert_array_equal(book.projection, again.projection)
            np.testing.assert_array_equal(book.thresholds, again.thresholds)
            before = book.thresholds.copy()
            same_input = np.concatenate((q, k), axis=0)
            encoded = book.encode(same_input)
            np.testing.assert_array_equal(encoded[:2], book.encode(q))
            np.testing.assert_array_equal(encoded[2:], book.encode(k))
            book.encode(np.full((1, 3, 9, 5), 1e6))
            np.testing.assert_array_equal(before, book.thresholds)
            np.testing.assert_allclose(np.einsum("hdm,hdn->hmn", book.projection, book.projection),
                                       np.tile(np.eye(2), (3, 1, 1)), atol=1e-12)

    def test_padding_not_used_to_fit_bins_or_center(self):
        rng = np.random.default_rng(9)
        q, k = rng.normal(size=(2, 1, 7, 3)), rng.normal(size=(2, 1, 7, 3))
        mask = np.tile(np.arange(7) < 4, (2, 1))
        first = QuantileCodebook.fit(q, k, mask=mask)
        q[:, :, 4:] = 1e8
        k[:, :, 4:] = -1e9
        second = QuantileCodebook.fit(q, k, mask=mask)
        np.testing.assert_array_equal(first.center, second.center)
        np.testing.assert_array_equal(first.thresholds, second.thresholds)
        self.assertEqual(first.calibration_tokens, 8)

    def test_zero_calibration_has_safe_duplicate_thresholds(self):
        data = np.zeros((1, 2, 5, 3))
        book = QuantileCodebook.fit(data, data, coordinates=2, digits=4)
        codes = book.encode(data)
        self.assertTrue(np.all(codes == codes[0, 0, 0]))
        self.assertTrue(np.all(np.isfinite(book.thresholds)))

    def test_attention_and_oracle_for_zero_logits_and_nonzero(self):
        probabilities = attention_probabilities(np.zeros(3), np.zeros((7, 3)))
        np.testing.assert_allclose(probabilities, np.ones(7) / 7)
        np.testing.assert_array_equal(select_from_scores(probabilities, 3), [4, 5, 6])
        q = np.array([10., -20., 30.])
        keys = np.array([[1., 2., 3.], [-1., -2., -3.], [10., -10., 20.]])
        probabilities = attention_probabilities(q, keys)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertEqual(int(select_from_scores(probabilities, 1)[0]), 2)


class DiagnosticPipelineTests(unittest.TestCase):
    def test_complete_diagnostic_preserves_sources_codes_and_calibration_split(self):
        path = Path(__file__).resolve().parents[1] / "experiments" / "routing_diagnostic.py"
        spec = importlib.util.spec_from_file_location("routing_diagnostic", path)
        diagnostic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diagnostic)
        rng = np.random.default_rng(404)
        q, k, v = (rng.normal(size=(3, 2, 13, 4)).astype(np.float32) for _ in range(3))
        with tempfile.TemporaryDirectory(prefix="routing-check-") as temporary:
            root = Path(temporary)
            trace = root / "trace.npz"
            np.savez(trace, queries=q, keys=k, values=v, valid_lengths=np.array([10, 11, 12]))
            result = diagnostic.run(trace, root / "first", calibration_chunks=1, budgets=[3, 7],
                                    recent_window=2, prime=3, code_bit_budget=8)
            self.assertEqual(result["config"]["digits"], [3, 2])
            self.assertEqual(result["config"]["evaluation_chunks"], [1, 2])
            self.assertEqual(result["equivalence_checks"]["padic_prefix_exact_selection_checks"], 92)
            self.assertEqual(result["memory_accounting"]["serialized_code_bytes_per_key_per_head"], 1)
            for name, expected in result["config"]["code_sha256"].items():
                captured = (root / "first" / result["config"]["code_snapshots"][name]).read_bytes()
                self.assertEqual(hashlib.sha256(captured).hexdigest(), expected)
            with np.load(root / "first" / "rawstats.npz", allow_pickle=False) as stats:
                first_thresholds = stats["encoder_thresholds"].copy()
                np.testing.assert_array_equal(stats["selected_indices"][:, 0], stats["selected_indices"][:, 1])
                np.testing.assert_array_equal(unpack_codes(stats["packed_key_codes"], 2, (3, 2), 3), stats["key_codes"])
            q[1:] *= 100
            k[1:] *= -10
            np.savez(trace, queries=q, keys=k, values=v, valid_lengths=np.array([10, 11, 12]))
            second = diagnostic.run(trace, root / "second", calibration_chunks=1, budgets=[3],
                                    recent_window=2, prime=3, code_bit_budget=8)
            self.assertNotEqual(result["config"]["trace_sha256"], second["config"]["trace_sha256"])
            self.assertEqual(result["config"]["calibration_qk_sha256"], second["config"]["calibration_qk_sha256"])
            with np.load(root / "second" / "rawstats.npz", allow_pickle=False) as stats:
                np.testing.assert_array_equal(first_thresholds, stats["encoder_thresholds"])
            saved = json.loads((root / "first" / "results.json").read_text())
            self.assertEqual(saved["status"], "component_diagnostic_completed")


if __name__ == "__main__":
    unittest.main()
