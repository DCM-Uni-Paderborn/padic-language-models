"""Mathematical fixtures; no scientific data or fitted experiment outputs."""
import unittest
import numpy as np
from padic_lm.continuous_tree import AffineDisk, norm_int, direct_loss, quotient_logits, metrics


class ContinuousTreeTests(unittest.TestCase):
    def test_extreme_logit_nll_stays_finite(self):
        result = metrics(np.array([[0., 1000.]]), np.array([0]))
        self.assertEqual(result["nll"], 1000.)
        self.assertEqual(result["accuracy"], 0.)

    def test_signed_fractional_centers(self):
        np.testing.assert_array_equal(norm_int(np.array([0, -9, 10, 81]), 3), [0, 1/9, 1, 1/81])
        m = AffineDisk(3, 1, radius=1/3)
        m.c[:] = -1  # -1/9, not an ordinary integer center.
        np.testing.assert_allclose(m.forward(np.array([[1]]))[0], [9])

    def test_disk_representative_invariance(self):
        x = np.array([[0, 1], [1, 1], [-7, 1], [81, 1]])
        m = AffineDisk(3, 2, radius=1/3)
        a = m.forward(x)[0]
        m.c += 27  # coefficient change3, norm1/3 lies in same disks.
        np.testing.assert_allclose(a, m.forward(x)[0])

    def test_joint_tie_and_directional_finite_difference(self):
        m = AffineDisk(3, 2, radius=1)
        x = np.array([[1, 1], [2, 1], [3, 1]])
        y = np.array([0, 1, 2])
        slopes, dirs, centers, caps = m.slopes(x, "direct", y=y)
        old = direct_loss(m, x, y)
        for slope, direction, c, cap in zip(slopes, dirs, centers, caps):
            t = AffineDisk(3, 2, radius=1)
            t.c = c.copy(); t.r += min(1e-7, cap/10)*direction
            fd = (direct_loss(t, x, y)-old)/min(1e-7, cap/10)
            self.assertAlmostEqual(float(slope), fd, places=7)

    def test_ordinary_quotient_equivalence(self):
        m = AffineDisk(2, 3)
        m.c = np.array([-17, 5, 21]); m.r = np.array([.0625, .1, .25])
        x = np.array([[0, 0, 1], [1, 7, 1], [-11, 12, 1]])
        np.testing.assert_allclose(-np.log(m.forward(x)[0]), quotient_logits(x, [m.state()])[:, 0])

    def test_classification_slopes(self):
        m = AffineDisk(2, 2)
        x = np.array([[0, 1], [1, 1], [2, 1], [-3, 1]])
        g = np.array([.2, -.3, .1, -.4])
        slopes, dirs, centers, caps = m.slopes(x, "classification", derivative=g)
        old = np.mean(g*(-np.log(m.forward(x)[0])))
        for slope, direction, c, cap in zip(slopes, dirs, centers, caps):
            t = AffineDisk(2, 2); t.c = c.copy(); t.r += 1e-7*direction
            fd = (np.mean(g*(-np.log(t.forward(x)[0])))-old)/1e-7
            self.assertAlmostEqual(float(slope), fd, places=6)


if __name__ == "__main__":
    unittest.main()
