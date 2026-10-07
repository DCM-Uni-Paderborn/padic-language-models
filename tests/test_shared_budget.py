import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.shared_budget import (twice_midranks, shared_rank_mask, fixed_group_mask,
                                  validate_group_mask, depth_row, grid_coordinates, grid_row, mass_group_mask)
from padic_lm.routing import common_padic_depth, reverse_digits


class SharedBudgetTests(unittest.TestCase):
    def test_midranks_match_pairwise_counts_and_are_invariant_to_monotone_transform(self):
        a = np.random.default_rng(2).integers(-3, 5, size=(6, 41))
        expected = np.array([[2 * np.sum(row < x) + np.sum(row == x) - 1 for x in row] for row in a])
        np.testing.assert_array_equal(twice_midranks(a), expected)
        np.testing.assert_array_equal(twice_midranks(np.exp(a)), expected)

    def test_shared_ranking_matches_independent_scalar_borda_and_future_is_never_requested(self):
        scores = np.random.default_rng(61).integers(0, 4, size=(6, 43, 43))
        calls = []
        def row(pos, stop):
            self.assertEqual(stop, pos + 1 - 3)
            calls.append(pos)
            return scores[:, pos, :stop]
        mask = shared_rank_mask(43, 6, row, budget=11, recent=3)
        for pos in range(43):
            for group in range(2):
                if pos < 11:
                    wanted = list(range(pos + 1))
                else:
                    stop = pos - 2
                    values = scores[group * 3:group * 3 + 3, pos, :stop]
                    ranks = [sum(2 * sum(r < r[j]) + sum(r == r[j]) - 1 for r in values) for j in range(stop)]
                    best = sorted(range(stop), key=lambda j: (ranks[j], j), reverse=True)[:8]
                    wanted = sorted(best + list(range(stop, pos + 1)))
                self.assertEqual(np.flatnonzero(mask[group, pos]).tolist(), wanted)
        self.assertEqual(calls, list(range(11, 43)))
        equal = shared_rank_mask(43, 6, lambda p, n: np.ones((6, n)), budget=11, recent=3)
        np.testing.assert_array_equal(equal, fixed_group_mask(43, 2, "recency", budget=11, recent=3))

    def test_fixed_controls_and_half_locality_do_not_expand_group_budget(self):
        for kind in ("full", "recency", "sink_recency", "uniform"):
            m = fixed_group_mask(149, 5, kind)
            validate_group_mask(m, budget=149 if kind == "full" else 128)
        sink = fixed_group_mask(149, 1, "sink_recency")
        self.assertTrue(sink[0, -1, :4].all())
        half = shared_rank_mask(149, 3, lambda p, n: -np.broadcast_to(np.arange(n), (3, n)), recent=64)
        self.assertEqual(np.flatnonzero(half[0, -1]).tolist(), list(range(64)) + list(range(85, 149)))

    def test_prime_depths_and_grid_reversal_match_scalar_paths_including_short_equal_coordinate(self):
        for prime, digits in ((2, (4, 4)), (3, (3, 2))):
            rng = np.random.default_rng(prime)
            codes = np.stack([rng.integers(prime ** d, size=(6, 79)) for d in digits], axis=-1).astype(np.uint16)
            q = codes[:, 7].copy()
            q[0] = codes[0, 1]
            actual = depth_row(q, codes, digits, prime)
            for h in range(6):
                np.testing.assert_array_equal(actual[h], common_padic_depth(q[h], codes[h], digits, prime))
            grid, scales = grid_coordinates(codes, digits, prime)
            np.testing.assert_array_equal(grid // scales, reverse_digits(codes, digits, prime))
            np.testing.assert_array_equal(scales, [1, 1] if prime == 2 else [4, 13])
            self.assertTrue(np.all(grid_row(grid[:, 7], grid) <= 0))

    def test_group_mass_control_dominates_any_admissible_selection(self):
        rng = np.random.default_rng(28)
        p = rng.uniform(size=(6, 43, 43)) * np.tri(43)
        p /= p.sum(-1, keepdims=True)
        m = mass_group_mask(p, budget=11, recent=3)
        rec = fixed_group_mask(43, 2, "recency", budget=11, recent=3)
        flat = shared_rank_mask(43, 6, lambda pos, n: rng.normal(size=(6, n)), budget=11, recent=3)
        s = p.reshape(2, 3, 43, 43).sum(1)
        self.assertTrue(np.all((s * m).sum(-1) + 1e-14 >= (s * rec).sum(-1)))
        self.assertTrue(np.all((s * m).sum(-1) + 1e-14 >= (s * flat).sum(-1)))


if __name__ == "__main__":
    unittest.main()
