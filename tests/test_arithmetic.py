"""Mathematical invariants; run with python3 -m unittest discover -s tests."""

import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.arithmetic import (
    balanced_decode,
    dot_abs_bound,
    exact_dot,
    exponent_at_bit_budget,
    modular_dot,
    modulus,
    signed_recovery_exponent,
    symmetric_quantize,
    twos_complement_dot,
)


class ArithmeticInvariantTests(unittest.TestCase):
    def test_balanced_decode_exhaustive_odd_even_and_noncanonical(self):
        for m in range(2, 80):
            decoded = [balanced_decode(r, m) for r in range(m)]
            self.assertEqual(sorted(decoded), list(range(-(m // 2), (m + 1) // 2)))
            for r in range(-2 * m, 3 * m):
                d = balanced_decode(r, m)
                self.assertEqual(d % m, r % m)
                self.assertLessEqual(-(m // 2), d)
                self.assertLess(d, (m + 1) // 2)
        self.assertEqual(balanced_decode(8, 16), -8)
        self.assertEqual(balanced_decode(2, 5), 2)
        self.assertEqual(balanced_decode(3, 5), -2)

    def test_modular_dot_homomorphism_for_arbitrary_precision_and_seeds(self):
        rng = random.Random(8317)
        for p in (2, 3, 5):
            for k in (1, 2, 5, 12):
                m = p**k
                for n in (0, 1, 13, 64):
                    for _ in range(12):
                        w = [rng.randrange(-(1 << 80), 1 << 80) for _ in range(n)]
                        x = [rng.randrange(-(1 << 80), 1 << 80) for _ in range(n)]
                        residue = modular_dot(w, x, p, k)
                        self.assertEqual(residue, exact_dot(w, x) % m)
                        self.assertEqual(residue, modular_dot([v + m for v in w], x, p, k))

    def test_numpy_dtype_cannot_overflow_multiplication(self):
        w = np.array([2**62, -(2**62)], dtype=np.int64)
        x = np.array([3, -5], dtype=np.int64)
        self.assertEqual(exact_dot(w, x), 2**65)
        self.assertEqual(dot_abs_bound(w, x), 2**65)
        for p in (2, 3, 5):
            self.assertEqual(modular_dot(w, x, p, 51), (2**65) % (p**51))

    def test_sufficient_recovery_bound_and_minimal_positive_exponent(self):
        for p in (2, 3, 5):
            for bound in range(100):
                k = signed_recovery_exponent(p, bound)
                m = modulus(p, k)
                self.assertGreater(m, 2 * bound)
                if k > 1:
                    self.assertLessEqual(p ** (k - 1), 2 * bound)
                for value in range(-bound, bound + 1):
                    self.assertEqual(balanced_decode(value % m, m), value)
        self.assertEqual(signed_recovery_exponent(2, 8), 5)
        self.assertNotEqual(balanced_decode(8, 16), 8)

    def test_dot_triangle_bound_implies_recovery(self):
        rng = random.Random(732)
        for _ in range(80):
            w = [rng.randrange(-7, 8) for _ in range(25)]
            x = [rng.randrange(-127, 128) for _ in range(25)]
            exact = exact_dot(w, x)
            bound = dot_abs_bound(w, x)
            self.assertLessEqual(abs(exact), bound)
            for p in (2, 3, 5):
                k = signed_recovery_exponent(p, bound)
                self.assertEqual(balanced_decode(modular_dot(w, x, p, k), p**k), exact)

    def test_twos_complement_control_matches_even_with_wraps(self):
        rng = random.Random(182)
        for bits in (1, 4, 8, 16, 64):
            for _ in range(80):
                w = [rng.randrange(-(1 << 70), 1 << 70) for _ in range(15)]
                x = [rng.randrange(-(1 << 70), 1 << 70) for _ in range(15)]
                decoded = balanced_decode(modular_dot(w, x, 2, bits), 1 << bits)
                self.assertEqual(twos_complement_dot(w, x, bits), decoded)

    def test_bit_budget_uses_exact_integer_cap(self):
        for p in (2, 3, 5):
            for bits in range(3, 32):
                k = exponent_at_bit_budget(p, bits)
                self.assertLessEqual(p**k, 1 << bits)
                self.assertGreater(p ** (k + 1), 1 << bits)
                if p == 2:
                    self.assertEqual(k, bits)

    def test_invalid_inputs_are_rejected(self):
        for p in (0, 1, 4, 6, -2):
            with self.assertRaises(ValueError):
                modulus(p, 1)
        with self.assertRaises(ValueError):
            modulus(2, 0)
        with self.assertRaises(TypeError):
            exact_dot([1.5], [1])
        with self.assertRaises(TypeError):
            exact_dot([True], [1])
        with self.assertRaises(ValueError):
            exact_dot([1], [])
        with self.assertRaises(ValueError):
            signed_recovery_exponent(2, -1)
        with self.assertRaises(ValueError):
            exponent_at_bit_budget(5, 1)


class QuantizationInvariantTests(unittest.TestCase):
    def test_zeros_empty_and_subnormal_are_safe(self):
        for values in (np.zeros(10), np.array([]), np.array([np.nextafter(0.0, 1.0)])):
            result = symmetric_quantize(values, 8)
            self.assertGreater(result.scale, 0)
            self.assertTrue(np.all(np.isfinite(result.dequantize())))
        self.assertEqual(symmetric_quantize(np.zeros(10), 4).scale, 1.0)

    def test_quantization_range_and_half_step_error(self):
        rng = np.random.default_rng(41)
        for bits in (2, 4, 8, 16):
            data = rng.normal(size=400)
            result = symmetric_quantize(data, bits)
            qmax = (1 << (bits - 1)) - 1
            self.assertTrue(np.all(np.abs(result.values) <= qmax))
            self.assertLessEqual(np.max(np.abs(result.dequantize() - data)), result.scale / 2 + 1e-14)
            self.assertEqual(result.values.dtype, np.dtype("int64"))

    def test_scales_remain_outside_modular_arithmetic(self):
        w = symmetric_quantize(np.array([-2.0, 0.0, 0.7, 1.2]), 4)
        x = symmetric_quantize(np.array([0.3, 1.0, -4.2, 0.1]), 8)
        exact = exact_dot(w.values, x.values)
        k = signed_recovery_exponent(3, dot_abs_bound(w.values, x.values))
        recovered = balanced_decode(modular_dot(w.values, x.values, 3, k), 3**k)
        self.assertEqual(recovered, exact)
        self.assertAlmostEqual(recovered * w.scale * x.scale, np.dot(w.dequantize(), x.dequantize()))

    def test_nonfinite_and_unsupported_bits_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                symmetric_quantize([value], 4)
        for bits in (1, 17):
            with self.assertRaises(ValueError):
                symmetric_quantize([1.0], bits)

    def test_iterable_inputs_are_supported_without_silent_truncation(self):
        expected = symmetric_quantize([-0.5, 0.0, 1.0], 4)
        actual = symmetric_quantize((v for v in [-0.5, 0.0, 1.0]), 4)
        np.testing.assert_array_equal(actual.values, expected.values)
        self.assertEqual(actual.scale, expected.scale)
        self.assertEqual(exact_dot((v for v in [3, 4]), (v for v in [5, 6])), 39)


if __name__ == "__main__":
    unittest.main()
