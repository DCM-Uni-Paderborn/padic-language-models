"""Check independent verification primitives against hand-derived fixtures."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from audit_shared_budget import count_ranks, prefix_depth, truth_mask


@unittest.skipUnless(TORCH_AVAILABLE, "Torch import required by the audit module")
class SharedAuditFixtures(unittest.TestCase):
    def test_histogram_midranks_have_exact_average_ties(self):
        scores = np.array([[5, -2, 5, 1, 1], [0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(count_ranks(scores), [[7, 0, 7, 3, 3], [4, 4, 4, 4, 4]])

    def test_prefix_traversal_respects_exhausted_equal_coordinate(self):
        q = np.array([[0, 0]], dtype=np.uint16)
        k = np.array([[[9, 0], [0, 3], [0, 0]]], dtype=np.uint16)
        np.testing.assert_array_equal(prefix_depth(q, k, (3, 2), 3), [[2, 1, 3]])
        binary = np.array([[[4, 0], [1, 0], [0, 0]]], dtype=np.uint16)
        np.testing.assert_array_equal(prefix_depth(q, binary, (4, 4), 2), [[2, 0, 4]])

    def test_shared_full_ties_reduce_to_recency_without_future_selection(self):
        length = 131
        zero = np.zeros((15, length, 2), dtype=np.uint16)
        state = {17: {"p3": {"q": zero, "k": zero, "digits": (3, 2), "prime": 3}}}
        mask = truth_mask(length, "s17_p3", state, None)
        self.assertFalse(np.triu(mask, 1).any())
        self.assertEqual(np.flatnonzero(mask[4, -1]).tolist(), list(range(3, 131)))
        self.assertTrue(np.array_equal(mask.sum(-1), np.broadcast_to(np.minimum(np.arange(1, 132), 128), (5, length))))


if __name__ == "__main__":
    unittest.main()
