"""Nontrivial codec, probability, and real-objective tests for the structure pilot."""
import unittest
import importlib.util
from pathlib import Path
import numpy as np
from padic_lm.structural_intent import reverse4, paths, probabilities, objective_gradient


class StructuralIntentTests(unittest.TestCase):
    def test_separate_verifier_gradient(self):
        path = Path(__file__).resolve().parents[1]/"experiments/audit_structural_intent.py"
        spec = importlib.util.spec_from_file_location("structural_verifier", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rng = np.random.default_rng(31)
        x = rng.normal(size=(9, 3)); y = rng.integers(0, 20, size=9)
        groups = np.repeat(np.arange(10), 2); codes = rng.permutation(20)
        _, left, right = paths(codes)
        w = rng.normal(scale=.1, size=(19, 3)); b = rng.normal(scale=.1, size=19)
        a = objective_gradient(x, y, groups, w, b, "tree", left, right)
        other = module.joint_gradient(x, y, groups, w, b, codes)
        for v, u in zip(a, other): np.testing.assert_allclose(v, u, rtol=1e-12, atol=1e-12)

    def test_quotients_and_invariance(self):
        for p in (2, 3, 5):
            for k in range(1, 5):
                m = p**k
                for a in range(-12, 13):
                    for b in range(-12, 13):
                        self.assertEqual(((a+b) % (m*p)) % m, (a % m+b % m) % m)
                        self.assertEqual(((a*b) % (m*p)) % m, (a % m*b % m) % m)
                        self.assertEqual((a+b*m) % m, a % m)
        for domain in range(10):
            for fine in range(16):
                code = reverse4(domain)+16*reverse4(fine)
                self.assertEqual(reverse4(code % 16), domain)
                for tail in (-31, -1, 0, 1, 31):
                    self.assertEqual(reverse4((code+16*tail) % 16), domain)

    def test_tree_probabilities_match_independent_traversal(self):
        codes = np.array([0, 1, 3, 4, 8, 9, 15])
        nodes, left, right = paths(codes)
        rng = np.random.default_rng(3)
        x, w, b = rng.normal(size=(11, 3)), rng.normal(size=(6, 3)), rng.normal(size=6)
        p = probabilities(x, w, b, "tree", left, right)
        z = x@w.T+b
        expected = np.ones_like(p)
        for j, (k, prefix) in enumerate(nodes):
            for c, code in enumerate(codes):
                if code % (2**int(k)) == prefix:
                    v = 1/(1+np.exp(-z[:, j]))
                    expected[:, c] *= v if (code >> k) & 1 else 1-v
        np.testing.assert_allclose(p, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-14)

    def test_joint_objective_gradients(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(13, 3)); y = rng.integers(0, 20, size=13)
        group = np.repeat(np.arange(10), 2)
        nodes, left, right = paths(np.arange(20))
        for kind, heads in (("tree", 19), ("flat", 19), ("domain", 9)):
            w, b = rng.normal(scale=.1, size=(heads, 3)), rng.normal(scale=.1, size=heads)
            loss, gw, gb = objective_gradient(x, y, group, w, b, kind, left, right)
            self.assertTrue(np.isfinite(loss))
            for row in (0, heads-1):
                for col in range(3):
                    up, down = w.copy(), w.copy(); up[row, col] += 1e-6; down[row, col] -= 1e-6
                    fd = (objective_gradient(x, y, group, up, b, kind, left, right)[0]-
                          objective_gradient(x, y, group, down, b, kind, left, right)[0])/2e-6
                    self.assertAlmostEqual(fd, gw[row, col], places=7)
                up, down = b.copy(), b.copy(); up[row] += 1e-6; down[row] -= 1e-6
                fd = (objective_gradient(x, y, group, w, up, kind, left, right)[0]-
                      objective_gradient(x, y, group, w, down, kind, left, right)[0])/2e-6
                self.assertAlmostEqual(fd, gb[row], places=7)


if __name__ == "__main__":
    unittest.main()
