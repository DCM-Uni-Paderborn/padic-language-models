import unittest
import numpy as np
from padic_lm.modular_adapter import features


class ModularAdapterTests(unittest.TestCase):
    def test_fourier_operand_product_identity_and_periodicity(self):
        x = np.array([[0, 0, 1], [1, 2, 1], [999998, 999997, 1]])
        f, _, _ = features(x, 9, "fourier")
        np.testing.assert_allclose(f[:, 0]-f[:, 3], np.cos(2*np.pi*((x[:, 0]+x[:, 1]) % 9)/9), atol=1e-14)
        np.testing.assert_allclose(f, features(x+np.array([9, 18, 0]), 9, "fourier")[0], atol=0.)

    def test_raw_scaling_fits_training_only(self):
        x = np.array([[0, 1, 1], [2, 3, 1]])
        f, loc, scale = features(x, 4, "raw")
        np.testing.assert_allclose(f[:, :2].mean(0), 0)
        f2, loc2, scale2 = features(np.array([[100, 101, 1]]), 4, "raw", loc, scale)
        np.testing.assert_array_equal(loc, loc2)
        self.assertGreater(f2[0, 0], 90)


if __name__ == "__main__":
    unittest.main()
