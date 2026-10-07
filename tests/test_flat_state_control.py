"""Endpoint and resource fixtures for the single state-capped flat control."""
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from flat_state_control import fit_capped, readouts, summaries, run, METHODS


class StateCappedControlTests(unittest.TestCase):
    def test_realistic_buffer_cap_and_actual_lloyd_fit(self):
        rng = np.random.default_rng(272)
        keys = rng.normal(size=(2, 9, 128, 64))
        center = np.zeros((9, 64), dtype=np.float64)
        projection = np.zeros((9, 64, 2), dtype=np.float64)
        projection[:, :2] = np.eye(2)
        mask = np.ones((2, 128), dtype=bool)
        fitted = fit_capped(keys, center, projection, mask)
        self.assertEqual(fitted.state_bytes, 15891)
        self.assertLessEqual(fitted.state_bytes, 16004)
        self.assertTrue(np.all(fitted.centroid_counts <= 13))
        self.assertTrue(np.all(fitted.iterations >= 1))
        self.assertTrue(np.all(fitted.iterations <= 20))
        self.assertTrue(np.all(fitted.unique_key_counts == 256))
        self.assertEqual(center.tobytes(), fitted.center.tobytes())
        self.assertEqual(projection.tobytes(), fitted.projection.tobytes())
        with self.assertRaises(AssertionError):
            fit_capped(keys, center, projection, mask, byte_cap=15890)

    def test_calibration_fit_excludes_invalid_padding_and_keeps_state(self):
        rng = np.random.default_rng(273)
        keys = rng.normal(size=(2, 1, 32, 2))
        mask = np.ones((2, 32), dtype=bool)
        mask[:, 29:] = False
        arguments = (np.zeros((1, 2)), np.eye(2)[None], mask)
        first = fit_capped(keys, *arguments)
        keys[:, :, 29:] = np.nan
        second = fit_capped(keys, *arguments)
        self.assertEqual(int(first.calibration_counts[0]), 58)
        for name, value in first.state_arrays.items():
            self.assertEqual(value.tobytes(), second.state_arrays[name].tobytes())
        before = {n: a.tobytes() for n, a in first.state_arrays.items()}
        first.encode_keys(rng.normal(size=(3, 1, 22, 2)))
        first.project_queries(rng.normal(size=(3, 1, 22, 2)))
        self.assertEqual(before, {n: a.tobytes() for n, a in first.state_arrays.items()})

    def test_readout_matches_independent_dense_and_restricted_weights(self):
        q = np.array([.7, -.2])
        keys = np.array([[1., 0.], [0., 1.], [-1., 2.]])
        values = np.array([[2., -1.], [-1., 3.], [4., 2.]])
        selections = [np.array([0, 1, 2]), np.array([0, 2]), np.array([1])]
        masses, errors, energy = readouts(q, keys, values, selections)
        logits = np.einsum('d,nd->n', q, keys) / np.sqrt(2)
        weights = np.exp(logits - max(logits)); weights /= weights.sum()
        reference = sum(weights[i] * values[i] for i in range(3))
        self.assertAlmostEqual(energy, sum(reference * reference), places=14)
        for index, indices in enumerate(selections):
            normalized = weights[indices] / sum(weights[indices])
            selected = sum(w * values[int(i)] for w, i in zip(normalized, indices))
            self.assertAlmostEqual(masses[index], sum(weights[indices]), places=14)
            self.assertAlmostEqual(errors[index], sum((selected - reference)**2), places=14)
        self.assertEqual(errors[0], 0)
        self.assertGreater(errors[1], 0)

    def test_aggregation_uses_energy_sums_and_excludes_query_31(self):
        ids = np.array([[2, 0, 31], [2, 0, 32], [2, 0, 33]])
        errors = np.tile([0., 1., 9.], (len(METHODS), 1))
        mass = np.tile([1., .5, .8], (len(METHODS), 1))
        reference = np.array([100., 1., 99.])
        agreement = np.zeros_like(errors, dtype=bool)
        records = summaries(ids, mass, errors, reference, agreement, 1)
        late = next(r for r in records if r['method']=='recency' and r['stratum']=='affected_queries')
        whole = next(r for r in records if r['method']=='recency' and r['stratum']=='all_queries')
        self.assertEqual(late['all_heads']['queries'], 2)
        self.assertAlmostEqual(late['all_heads']['value_output_nrmse'], np.sqrt(10/100))
        self.assertAlmostEqual(whole['all_heads']['value_output_nrmse'], np.sqrt(10/200))
        self.assertNotAlmostEqual(late['all_heads']['value_output_nrmse'], (1+np.sqrt(9/99))/2)

    def test_existing_output_and_invalid_limits_never_modify_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root/'output'; existing.mkdir()
            sentinel = existing/'keep'; sentinel.write_text('unchanged')
            for seconds in (True, np.bool_(True), 0, 601, np.nan, np.inf):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    run(root, root, root, root, existing, max_seconds=seconds)
            with self.assertRaises(FileExistsError):
                run(root, root, root, root, existing)
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual(list(existing.iterdir()), [sentinel])


if __name__ == '__main__':
    unittest.main()
