"""Independent finite algebra, feature equivalence and optimization checks."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from padic_lm.ball_ffn import (finite_affine, ball_features, reversed_digits,
                              row_features, features, ridge, discrete_fit)


class BallFFNTests(unittest.TestCase):
    def test_affine_matches_python_integer_oracle_and_rejects_overflow(self):
        rng = np.random.default_rng(761)
        for p, k in ((2, 4), (3, 3), (5, 2)):
            x = rng.integers(-(1 << 60), 1 << 60, (31, 16))
            w = rng.integers(-(1 << 60), 1 << 60, (7, 16))
            b = rng.integers(-(1 << 60), 1 << 60, 7)
            expected = [[(sum(int(a)*int(c) for a, c in zip(xx, ww))+int(bb)) % p**k
                         for ww, bb in zip(w, b)] for xx in x]
            np.testing.assert_array_equal(finite_affine(x, w, b, p, k), expected)
        with self.assertRaises(OverflowError):
            finite_affine(np.zeros((1, 16), dtype=np.int64), np.zeros((1, 16), dtype=np.int64),
                          np.zeros(1, dtype=np.int64), 2, 31)
        with self.assertRaises(TypeError):
            finite_affine([[.5]], [[1]], [0])

    def test_all_ball_features_equal_low_digit_tree(self):
        for p in (2, 3, 5):
            for k in (1, 2, 3, 4):
                values = np.arange(p**k)[:, None]
                expected = []
                for (value,) in values:
                    digits = [(int(value)//p**j) % p for j in range(k)]
                    expected.append([all(d == 0 for d in digits[:ell]) for ell in range(1, k+1)])
                np.testing.assert_array_equal(ball_features(values, p, k), expected)
                reverse = reversed_digits(p, k).astype(np.int64)
                np.testing.assert_array_equal(reverse[reverse], np.arange(p**k))

    def test_signed_features_use_negative_endpoint(self):
        a = row_features(np.asarray([0, 7, 8, 15]), "signed")
        np.testing.assert_array_equal(a[:, 0], [0, 7/8, -1, -1/8])
        self.assertEqual(a[2, 3], 1.)

    def test_discrete_search_reduces_full_real_objective(self):
        rng = np.random.default_rng(157)
        x = rng.integers(0, 16, (256, 3), dtype=np.uint8)
        w = rng.integers(0, 16, (2, 3), dtype=np.uint8)
        b = rng.integers(0, 16, 2, dtype=np.uint8)
        truth = features(x, np.asarray([[1, 3, 5], [7, 1, 9]], dtype=np.uint8),
                         np.asarray([0, 3], dtype=np.uint8)) @ rng.normal(size=(8, 5))
        for kind in ("ball", "signed", "real"):
            xx = (x.astype(float)-7.5)/4 if kind == "real" else x
            scale = np.ones(2) if kind == "real" else None
            initial, final, history = discrete_fit(xx, truth, w, b, kind, scale, seed=17)
            for stage in (initial, final):
                phi = features(xx, stage["weights"], stage["bias"], kind, scale)
                residual = truth-phi @ stage["coefficient"]-stage["intercept"]
                self.assertTrue(np.isfinite(residual).all())
            last = None
            for event in history:
                if event["row"] >= 0 and last is not None:
                    self.assertLessEqual(event["sse"], last+1e-8)
                last = event["sse"]
            self.assertEqual(sum(e["proposals"] for e in history), 2*2*4*4)
            phi = features(xx, final["weights"], final["bias"], kind, scale)
            xc = phi-phi.mean(axis=0)
            yc = truth-truth.mean(axis=0)
            gradient = xc.T @ (xc @ final["coefficient"]-yc)/len(x)+.001*final["coefficient"]
            self.assertLess(np.max(np.abs(gradient)), 1e-12)


if __name__ == "__main__":
    unittest.main()
