import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.prefix_index import CausalPrefixIndex


def exhaustive(query, keys, prime, digits, quota, coarsen=1):
    depth = np.zeros(keys.shape, dtype=np.int64)
    left_count = prime ** digits[0]
    for level in range(1, max(digits) + 1):
        left = prime ** min(level, digits[0])
        right = prime ** min(level, digits[1])
        equal = ((keys % left_count) % left == (query[:, None] % left_count) % left)
        equal &= ((keys // left_count) % right == (query[:, None] // left_count) % right)
        depth += equal
    depth //= coarsen
    ranks = np.empty_like(depth)
    for h, scores in enumerate(depth):
        for i, value in enumerate(scores):
            ranks[h, i] = 2 * np.count_nonzero(scores < value) + np.count_nonzero(scores == value) - 1
    aggregate = ranks.sum(0)
    order = sorted(range(keys.shape[1]), key=lambda i: (int(aggregate[i]), i), reverse=True)
    return np.array(sorted(order[:quota]), dtype=np.int64)


class PrefixIndexTests(unittest.TestCase):
    def test_random_binary_ternary_and_coarsened_exact_rank_selection(self):
        rng = np.random.default_rng(281204)
        for prime, digits, coarsen in ((2, (4, 4), 1), (2, (4, 4), 2), (3, (3, 2), 1), (5, (1, 2), 1)):
            for heads in (1, 3):
                for count in (13, 121, 177):
                    keys = rng.integers(0, prime ** sum(digits), (heads, count))
                    index = CausalPrefixIndex(prime, digits, heads, coarsen)
                    for position in range(count):
                        index.append(keys[:, position])
                    for _ in range(4):
                        query = rng.integers(0, prime ** sum(digits), heads)
                        selected, diagnostics = index.query(query, quota=min(120, count))
                        np.testing.assert_array_equal(selected, exhaustive(query, keys, prime, digits, min(120, count), coarsen))
                        self.assertLessEqual(diagnostics["distinct_candidates_scored"], count)
                    self.assertEqual(index.stored_position_entries(), count * heads * len(index.cut_depths))

    def test_all_ties_select_newest_without_scoring_entire_index(self):
        index = CausalPrefixIndex(2, (4, 4), 3)
        for _ in range(2040):
            index.append([0, 0, 0])
        selected, info = index.query([255, 255, 255])
        np.testing.assert_array_equal(selected, np.arange(1920, 2040))
        self.assertEqual(info["distinct_candidates_scored"], 120)
        self.assertLessEqual(info["head_list_entries_yielded"], 360)
        self.assertEqual(info["termination"], "aggregate_score_and_timestamp_threshold")

    def test_causal_appends_preserve_every_intermediate_answer(self):
        rng = np.random.default_rng(9917)
        keys = rng.integers(0, 243, (3, 38))
        index = CausalPrefixIndex(3, (3, 2), 3)
        for position in range(keys.shape[1]):
            index.append(keys[:, position])
            query = rng.integers(0, 243, 3)
            selected, _ = index.query(query, quota=7)
            np.testing.assert_array_equal(selected, exhaustive(query, keys[:, :position + 1], 3, (3, 2), 7))

    def test_rejects_invalid_codes_and_handles_empty_or_zero_quota(self):
        index = CausalPrefixIndex(3, (3, 2), 3)
        self.assertEqual(index.query([0, 0, 0])[0].size, 0)
        for invalid in ([0, 0], [0, 0, 243], [-1, 0, 0], [0., 1., 2.]):
            with self.assertRaises(ValueError):
                index.append(invalid)
        index.append([0, 0, 0])
        self.assertEqual(index.query([0, 0, 0], quota=0)[0].size, 0)


if __name__ == "__main__":
    unittest.main()
