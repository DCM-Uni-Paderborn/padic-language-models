"""Independent fixtures for the calibration-matched flat one-byte router."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from padic_lm.flat_routing import FlatKeyCodebook
from padic_lm.routing import QuantileCodebook


def identity_fit(points, **kwargs):
    points = np.asarray(points, dtype=np.float64)
    return FlatKeyCodebook.fit(points[None, None], center=np.zeros((1, 2)),
                               projection=np.eye(2)[None], **kwargs)


def independent_selection(scores, position, budget, recent):
    """Python ordering, independent of the imported NumPy score selector."""
    count = min(position + 1, budget)
    mandatory = list(range(position + 1 - min(recent, count), position + 1))
    pool = [index for index in range(position + 1) if index not in mandatory]
    ranked = sorted(pool, key=lambda index: (scores[index], index), reverse=True)
    return sorted(mandatory + ranked[:count - len(mandatory)])


class FlatCalibrationTests(unittest.TestCase):
    def test_shared_projection_is_exact_and_held_state_is_not_refitted(self):
        rng = np.random.default_rng(103)
        queries, keys = rng.normal(size=(2, 3, 17, 5)), rng.normal(size=(2, 3, 17, 5))
        for projection_kind in ("pca", "random"):
            shared = QuantileCodebook.fit(queries, keys, projection_kind=projection_kind)
            flat = FlatKeyCodebook.fit(keys, center=shared.center, projection=shared.projection)
            np.testing.assert_array_equal(flat.center, shared.center)
            np.testing.assert_array_equal(flat.projection, shared.projection)
            expected = np.stack([(queries[:, head] - shared.center[head]) @ shared.projection[head]
                                 for head in range(3)], axis=1)
            np.testing.assert_array_equal(flat.project_queries(queries), expected)
            before = {name: value.tobytes() for name, value in flat.state_arrays.items()}
            flat.encode_keys(np.full((1, 3, 7, 5), 1e4))
            flat.project_queries(np.full((1, 3, 7, 5), -1e4))
            self.assertEqual(before, {name: value.tobytes() for name, value in flat.state_arrays.items()})
            self.assertFalse(np.shares_memory(flat.center, shared.center))
            self.assertFalse(np.shares_memory(flat.projection, shared.projection))

    def test_256_unique_keys_have_byte_codes_and_exact_projected_roundtrip(self):
        points = np.column_stack((np.arange(256, dtype=np.float64), np.arange(256) % 7))
        flat = identity_fit(points)
        encoded = flat.encode_keys(points[None, None])
        self.assertEqual(encoded.dtype, np.dtype(np.uint8))
        self.assertEqual(encoded.nbytes, 256)
        self.assertEqual(int(flat.centroid_counts[0]), 256)
        self.assertEqual(int(flat.unique_key_counts[0]), 256)
        self.assertEqual(int(flat.iterations[0]), 0)
        np.testing.assert_array_equal(encoded[0, 0], np.arange(256, dtype=np.uint8))
        np.testing.assert_array_equal(flat.decode_keys(encoded)[0, 0], points)
        np.testing.assert_array_equal(flat.encode_keys(flat.decode_keys(encoded)), encoded)

    def test_duplicate_points_and_signed_zero_have_canonical_centroid_ids(self):
        points = np.array([[1, 2], [-0.0, 0], [1, 2], [0, -0.0], [-2, 1]])
        first, second = identity_fit(points), identity_fit(points[::-1])
        np.testing.assert_array_equal(first.centroids[0], [[-2, 1], [0, 0], [1, 2]])
        self.assertEqual(first.centroids.tobytes(), second.centroids.tobytes())
        self.assertFalse(np.any(np.signbit(first.centroids[first.centroids == 0])))
        np.testing.assert_array_equal(first.encode_keys(points[None, None])[0, 0], [2, 1, 2, 1, 0])
        self.assertEqual(int(first.unique_key_counts[0]), 3)
        self.assertEqual(int(first.calibration_counts[0]), 5)

    def test_lloyd_uses_duplicate_weights_and_prespecified_cap(self):
        flat = identity_fit([[0, 0], [0, 0], [6, 0]], max_centroids=1)
        np.testing.assert_array_equal(flat.centroids[0], [[2, 0]])
        self.assertEqual(flat.max_centroids, 1)
        self.assertEqual(flat.max_iterations, 20)
        self.assertEqual(flat.seed, 17)
        self.assertGreaterEqual(int(flat.iterations[0]), 1)
        self.assertLessEqual(int(flat.iterations[0]), 20)
        one_step = identity_fit([[0, 0], [0, 0], [6, 0]], max_centroids=1, max_iterations=1)
        np.testing.assert_array_equal(one_step.centroids, flat.centroids)
        self.assertEqual(int(one_step.iterations[0]), 1)

    def test_compressed_fit_is_seeded_canonical_and_permutation_invariant(self):
        rng = np.random.default_rng(118)
        points = rng.normal(size=(100, 2))
        points = np.concatenate((points, points[:13]), axis=0)
        first = identity_fit(points, max_centroids=7, seed=29)
        second = identity_fit(points[rng.permutation(len(points))], max_centroids=7, seed=29)
        for name, value in first.state_arrays.items():
            np.testing.assert_array_equal(value, second.state_arrays[name])
        self.assertLessEqual(int(first.centroid_counts[0]), 7)
        self.assertLessEqual(int(first.iterations[0]), 20)
        # Every active centroid is distinct and ordered, including after updates.
        active = first.centroids[0, :int(first.centroid_counts[0])]
        np.testing.assert_array_equal(active, np.unique(active, axis=0))

    def test_masked_padding_never_changes_calibration_fit(self):
        keys = np.array([[[[1, 0], [3, 0], [9, 9]]], [[[2, 0], [4, 0], [8, 8]]]], dtype=float)
        mask = np.array([[True, True, False], [True, True, False]])
        arguments = {"center": np.zeros((1, 2)), "projection": np.eye(2)[None], "mask": mask}
        first = FlatKeyCodebook.fit(keys, **arguments)
        keys[:, :, 2] = [np.nan, np.inf]
        second = FlatKeyCodebook.fit(keys, **arguments)
        for name, value in first.state_arrays.items():
            np.testing.assert_array_equal(value, second.state_arrays[name])
        self.assertEqual(int(first.calibration_counts[0]), 4)
        np.testing.assert_array_equal(first.centroids[0, :, 0], [1, 2, 3, 4])

    def test_all_zero_keys_and_per_head_padding_are_counted_exactly(self):
        keys = np.zeros((2, 2, 3, 2))
        keys[:, 1, :, 0] = np.array([[0, 1, 2], [0, 1, 2]])
        flat = FlatKeyCodebook.fit(keys, center=np.zeros((2, 2)),
                                   projection=np.repeat(np.eye(2)[None], 2, axis=0))
        np.testing.assert_array_equal(flat.centroid_counts, [1, 3])
        np.testing.assert_array_equal(flat.unique_key_counts, [1, 3])
        np.testing.assert_array_equal(flat.centroids[0], np.zeros((3, 2)))
        encoded = flat.encode_keys(keys)
        np.testing.assert_array_equal(encoded[:, 0], 0)
        np.testing.assert_array_equal(flat.decode_keys(encoded), flat.project_queries(keys))
        expected_bytes = sum(value.nbytes for value in flat.state_arrays.values())
        # H=2,D=2,Kpad=3: center32 + map64 + centers96 + counts4 +
        # unique16 + calibration16 + iterations2 + parameters24 =254 bytes.
        self.assertEqual(expected_bytes, 254)
        self.assertEqual(flat.state_bytes, expected_bytes)
        self.assertTrue(all(not value.flags.writeable for value in flat.state_arrays.values()))

    def test_nonfinite_retained_values_encoder_and_numeric_overflow_are_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    identity_fit([[0, 0], [bad, 1]])
                with self.assertRaises(ValueError):
                    FlatKeyCodebook.fit(np.zeros((1, 1, 2, 2)), center=np.array([[bad, 0]]),
                                        projection=np.eye(2)[None])
                projection = np.eye(2)[None].copy()
                projection[0, 0, 0] = bad
                with self.assertRaises(ValueError):
                    FlatKeyCodebook.fit(np.zeros((1, 1, 2, 2)), center=np.zeros((1, 2)),
                                        projection=projection)
        flat = identity_fit([[0, 0]])
        with self.assertRaises(ValueError):
            flat.project_queries(np.full((1, 1, 2, 2), np.nan))
        with self.assertRaises(ValueError):
            flat.encode_keys(np.full((1, 1, 2, 2), 1e300))
        with self.assertRaises(ValueError):
            identity_fit([[-1e300, 0], [1e300, 0]], max_centroids=1)

    def test_invalid_fit_shapes_masks_and_parameters_are_rejected(self):
        base = np.zeros((1, 1, 2, 2))
        arguments = {"center": np.zeros((1, 2)), "projection": np.eye(2)[None]}
        for name, value in (("max_centroids", 0), ("max_centroids", 257),
                            ("max_iterations", 0), ("max_iterations", 21), ("seed", -1)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                FlatKeyCodebook.fit(base, **arguments, **{name: value})
        for name in ("max_centroids", "max_iterations", "seed"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                FlatKeyCodebook.fit(base, **arguments, **{name: True})
        for keys in (np.zeros((1, 1, 2)), np.zeros((0, 1, 2, 2)), np.zeros((1, 2, 2, 2))):
            with self.assertRaises(ValueError):
                FlatKeyCodebook.fit(keys, **arguments)
        for mask in (np.zeros((1, 2), dtype=bool), np.ones((2, 1), dtype=bool)):
            with self.assertRaises(ValueError):
                FlatKeyCodebook.fit(base, **arguments, mask=mask)
        with self.assertRaises(TypeError):
            FlatKeyCodebook.fit(base, **arguments, mask=np.ones((1, 2)))
        with self.assertRaises(ValueError):
            FlatKeyCodebook.fit(base, center=arguments["center"], projection=np.ones((1, 2, 3)))


class FlatScoringTests(unittest.TestCase):
    def test_nearest_centroid_ties_choose_lexicographic_id(self):
        flat = identity_fit([[1, 0], [-1, 0]])
        encoded = flat.encode_keys(np.array([[[[0., 0.], [0., 1.]]]]))
        np.testing.assert_array_equal(encoded, [[[0, 0]]])

    def test_compressed_codes_and_masks_match_independent_centroid_distances(self):
        rng = np.random.default_rng(208)
        calibration = rng.normal(size=(45, 2))
        evaluation = rng.normal(size=(23, 2))
        flat = identity_fit(calibration, max_centroids=5)
        centroids = flat.centroids[0, :int(flat.centroid_counts[0])]
        codes = flat.encode_keys(evaluation[None, None])
        expected_codes = []
        for point in evaluation:
            distances = [sum((float(a) - float(b)) ** 2 for a, b in zip(point, center))
                         for center in centroids]
            expected_codes.append(min(range(len(centroids)), key=lambda index: (distances[index], index)))
        np.testing.assert_array_equal(codes[0, 0], expected_codes)
        np.testing.assert_array_equal(flat.encode_keys(flat.decode_keys(codes)), codes)
        query = np.array([.4, -.3])
        scores = [-sum((float(a) - float(b)) ** 2 for a, b in zip(query, centroids[code]))
                  for code in expected_codes]
        actual = flat.select_causal(query, codes[0, 0], 22, 7, 3)
        np.testing.assert_array_equal(actual, independent_selection(scores, 22, 7, 3))

    def test_independent_euclidean_masks_and_recent_budget_at_every_position(self):
        points = np.array([[0, 0], [3, 1], [1, 2], [-2, 1], [0, 0], [5, 0]])
        flat = identity_fit(points)
        codes = flat.encode_keys(points[None, None])[0, 0]
        query = np.array([.5, 1.])
        scores = [-sum((float(q) - float(k)) ** 2 for q, k in zip(query, key)) for key in points]
        for position in range(len(points)):
            for budget in (1, 3, 8):
                for recent in (0, 2, 20):
                    actual = flat.select_causal(query, codes, position, budget, recent)
                    np.testing.assert_array_equal(actual, independent_selection(scores, position, budget, recent))
                    self.assertEqual(len(actual), min(position + 1, budget))
                    self.assertTrue(np.all(actual <= position))
                    self.assertEqual(len(np.unique(actual)), len(actual))

    def test_causal_suffix_cannot_change_selection_even_if_future_ids_invalid(self):
        flat = identity_fit([[0, 0], [1, 0], [2, 0], [3, 0]])
        codes = np.array([0, 1, 2, 3, 0, 0], dtype=np.int64)
        for score_kind in ("euclidean", "dot_product"):
            original = flat.select_causal(np.array([0., 0.]), codes, 3, 2, 1, score_kind=score_kind)
            codes[4:] = [-999, 999]
            np.testing.assert_array_equal(
                flat.select_causal(np.array([0., 0.]), codes, 3, 2, 1, score_kind=score_kind), original)
        np.testing.assert_array_equal(flat.select_causal(np.array([0., 0.]), codes, 3, 2, 1), [0, 3])

    def test_zero_score_ties_choose_latest_and_full_prefix_is_unchanged(self):
        flat = identity_fit(np.zeros((6, 2)))
        codes = np.zeros(6, dtype=np.uint8)
        for score_kind in ("euclidean", "dot_product"):
            np.testing.assert_array_equal(flat.select_causal(np.zeros(2), codes, 5, 3, 0,
                                                             score_kind=score_kind), [3, 4, 5])
            np.testing.assert_array_equal(flat.select_causal(np.zeros(2), codes, 2, 32, 8,
                                                             score_kind=score_kind), [0, 1, 2])

    def test_nonzero_center_dot_control_matches_independent_uncentered_projection(self):
        # Centered dots would prefer key 1; restoring the original origin
        # prefers key 0: (10,1).(11,0)=110 > (10,1).(10,2)=102.
        center = np.array([[10., 0., -3.]])
        projection = np.array([[[1., 0.], [0., 1.], [0., 0.]]])
        keys = np.array([[[[11., 0., 2.], [10., 2., -1.]]]])
        query = np.array([[[[10., 1., 8.]]]])
        flat = FlatKeyCodebook.fit(keys, center=center, projection=projection)
        projected_query = flat.project_queries(query)[0, 0, 0]
        key_codes = flat.encode_keys(keys)[0, 0]
        restored_query = projected_query + center[0] @ projection[0]
        restored_keys = flat.decode_keys(key_codes[None, None])[0, 0] + center[0] @ projection[0]
        np.testing.assert_array_equal(restored_query, query[0, 0, 0] @ projection[0])
        np.testing.assert_array_equal(restored_keys, keys[0, 0] @ projection[0])
        direct_scores = [sum(float(q) * float(k) for q, k in zip(query[0, 0, 0] @ projection[0],
                                                               key @ projection[0])) for key in keys[0, 0]]
        np.testing.assert_array_equal(flat.select_causal(projected_query, key_codes, 1, 1, 0,
                                                         score_kind="dot_product"),
                                      independent_selection(direct_scores, 1, 1, 0))
        np.testing.assert_array_equal(flat.select_causal(projected_query, key_codes, 1, 1, 0,
                                                         score_kind="dot_product"), [0])
        before = {name: values.tobytes() for name, values in flat.state_arrays.items()}
        flat.select_causal(projected_query, key_codes, 1, 1, 0, score_kind="euclidean")
        self.assertEqual(before, {name: values.tobytes() for name, values in flat.state_arrays.items()})

    def test_key_norm_can_make_euclidean_and_dot_rankings_differ(self):
        flat = identity_fit([[1, 0], [10, 0]])
        codes = np.array([0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(flat.select_causal(np.array([1., 0.]), codes, 1, 1, 0), [0])
        np.testing.assert_array_equal(flat.select_causal(np.array([1., 0.]), codes, 1, 1, 0,
                                                         score_kind="dot_product"), [1])

    def test_random_independent_dot_masks_match_original_projected_features(self):
        rng = np.random.default_rng(820)
        keys = rng.normal(size=(2, 2, 9, 4))
        queries = rng.normal(size=keys.shape)
        shared = QuantileCodebook.fit(queries, keys, projection_kind="random")
        flat = FlatKeyCodebook.fit(keys, center=shared.center, projection=shared.projection)
        codes = flat.encode_keys(keys)
        projected_queries = flat.project_queries(queries)
        for chunk in range(2):
            for head in range(2):
                for position in range(9):
                    uncentered_q = queries[chunk, head, position] @ shared.projection[head]
                    uncentered_keys = keys[chunk, head] @ shared.projection[head]
                    scores = [sum(float(a) * float(b) for a, b in zip(uncentered_q, key))
                              for key in uncentered_keys]
                    np.testing.assert_array_equal(
                        flat.select_causal(projected_queries[chunk, head, position], codes[chunk, head],
                                           position, 4, 2, head=head, score_kind="dot_product"),
                        independent_selection(scores, position, 4, 2))

    def test_invalid_codes_query_shape_options_and_overflow_rejected(self):
        flat = identity_fit([[0, 0], [1, 0]])
        for codes in (np.array([-1, 0]), np.array([0, 2])):
            with self.assertRaises(ValueError):
                flat.select_causal(np.zeros(2), codes, 1, 1, 0)
            with self.assertRaises(ValueError):
                flat.decode_keys(codes[None, None])
        for codes in (np.array([0., 1.]), np.array([False, True])):
            with self.assertRaises(TypeError):
                flat.select_causal(np.zeros(2), codes, 1, 1, 0)
        for query in (np.zeros(3), np.array([np.nan, 0]), np.array([np.inf, 0])):
            with self.assertRaises(ValueError):
                flat.select_causal(query, np.array([0, 1]), 1)
        for arguments in ({"head": 1}, {"score_kind": "unknown"}, {"budget": 0}, {"recent_window": -1}):
            with self.assertRaises(ValueError):
                flat.select_causal(np.zeros(2), np.array([0, 1]), 1, **arguments)
        with self.assertRaises(ValueError):
            flat.select_causal(np.array([1e300, 1e300]), np.array([0, 1]), 1)
        enormous = identity_fit([[1e300, 0]])
        with self.assertRaises(ValueError):
            enormous.select_causal(np.array([1e300, 0]), np.array([0]), 0, score_kind="dot_product")


if __name__ == "__main__":
    unittest.main()
