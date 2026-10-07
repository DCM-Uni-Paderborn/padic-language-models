"""Fixed-gate pipeline, data separation, causal controls and provenance checks."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))
import routing_alignment as alignment
from padic_lm.routing import attention_probabilities, common_padic_depth, unpack_codes


def fixture(seed=561):
    """Small head count/valid prefixes within the exact archived trace layout."""
    rng = np.random.default_rng(seed)
    queries = rng.normal(size=(32, 2, 128, 4)).astype(np.float32)
    keys = rng.normal(size=queries.shape).astype(np.float32)
    values = rng.normal(size=(32, 2, 128, 3)).astype(np.float32)
    lengths = np.full(32, 35, dtype=np.int64)
    lengths[:2] = [6, 7]
    return queries, keys, values, lengths


class AlignmentPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="routing-alignment-check-")
        cls.directory = Path(cls.temporary.name)
        cls.q, cls.k, cls.v, cls.lengths = fixture()
        cls.trace = cls.directory / "trace.npz"
        np.savez(cls.trace, queries=cls.q, keys=cls.k, values=cls.v, valid_lengths=cls.lengths)
        cls.output = cls.directory / "first"
        cls.result = alignment.run(cls.trace, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_pipeline_pins_fixed_cases_affected_positions_and_sources(self):
        config = self.result["config"]
        self.assertEqual(config["calibration_chunks"], [0, 1])
        self.assertEqual(config["evaluation_chunks"], list(range(2, 32)))
        self.assertEqual(config["calibration_tokens_per_head_per_role"], 13)
        self.assertEqual([(v["projection_kind"], v["seed"]) for v in self.result["variants"]],
                         list(alignment.FAMILIES))
        self.assertEqual(config["methods"], list(alignment.METHODS))
        self.assertEqual(self.result["checks"]["padic_trie_exact_selection_checks"], 4 * 2 * 30 * 35 * 2)
        self.assertEqual(self.result["checks"]["packed_code_roundtrips"], 16)
        self.assertEqual(config["trace_sha256"], hashlib.sha256(self.trace.read_bytes()).hexdigest())
        self.assertEqual(len(config["code_sha256"]), 4)
        completed = json.loads((self.output / "completed.json").read_text())
        self.assertEqual(self.result["completion_sha256"], hashlib.sha256((self.output / "completed.json").read_bytes()).hexdigest())
        self.assertEqual((self.output / "completed.sha256").read_text().strip(), self.result["completion_sha256"])
        for name, digest in config["code_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / config["code_snapshots"][name]).read_bytes()).hexdigest(), digest)
        for name, digest in completed["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), digest)
            self.assertEqual((self.output / name).stat().st_size, completed["artifact_bytes"][name])
        self.assertEqual(json.loads((self.output / "results.json").read_text())["status"], "alignment_diagnostic_completed")
        for variant in self.result["variants"]:
            all_rows = [r for r in variant["summaries"] if r["stratum"] == "all_queries"]
            late_rows = [r for r in variant["summaries"] if r["stratum"] == "affected_queries"]
            self.assertEqual(len(all_rows), len(alignment.METHODS))
            self.assertEqual(len(late_rows), len(alignment.METHODS))
            self.assertTrue(all(r["all_heads"]["queries"] == 30 * 35 * 2 for r in all_rows))
            self.assertTrue(all(r["all_heads"]["queries"] == 30 * 3 * 2 for r in late_rows))
            self.assertTrue(all(len(r["per_head"]) == 2 for r in late_rows))

    def test_saved_codes_embeddings_state_bytes_and_independent_trees(self):
        for variant in self.result["variants"]:
            directory = self.output / variant["name"]
            with np.load(directory / "encoder_state.npz", allow_pickle=False) as state:
                np.testing.assert_array_equal(state["p2_center"], state["p3_center"])
                np.testing.assert_array_equal(state["p2_center"], state["flat_center"])
                np.testing.assert_array_equal(state["p2_projection"], state["p3_projection"])
                np.testing.assert_array_equal(state["p2_projection"], state["flat_projection"])
                for prefix in ("p2", "p3", "flat"):
                    state_bytes = sum(state[name].nbytes for name in state.files if name.startswith(prefix + "_"))
                    self.assertEqual(state_bytes, variant["memory_accounting"][prefix]["numeric_state_array_bytes"])
                self.assertEqual(state["flat_centroid_counts"].tolist(), [13, 13])
                self.assertEqual(state["flat_iterations"].tolist(), [0, 0])
                with np.load(directory / "codes.npz", allow_pickle=False) as codes:
                    for prime in (2, 3):
                        digits = tuple(state[f"p{prime}_digits"].tolist())
                        for role in ("query", "key"):
                            packed = codes[f"p{prime}_packed_{role}_codes"]
                            self.assertEqual(packed.dtype, np.uint8)
                            self.assertEqual(packed.shape[-1], 1)
                            np.testing.assert_array_equal(unpack_codes(packed, 2, digits, prime), codes[f"p{prime}_{role}_codes"])
                    self.assertEqual(codes["flat_key_codes"].dtype, np.uint8)
            with np.load(directory / "rawstats.npz", allow_pickle=False) as raw:
                selected = raw["selected_indices"]
                for prime in (2, 3):
                    p_index, t_index = alignment.METHODS.index(f"padic_{prime}"), alignment.METHODS.index(f"trie_{prime}")
                    np.testing.assert_array_equal(selected[p_index], selected[t_index])
                    np.testing.assert_array_equal(raw["kept_attention_mass"][p_index], raw["kept_attention_mass"][t_index])
                    np.testing.assert_array_equal(raw["output_error_squared"][p_index], raw["output_error_squared"][t_index])

    def test_all_saved_masks_are_causal_equal_budget_and_distributions_count_them(self):
        for variant in self.result["variants"]:
            with np.load(self.output / variant["name"] / "rawstats.npz", allow_pickle=False) as raw:
                identities = raw["identifiers"]
                self.assertTrue(np.all(identities[:, 0] >= 2))
                self.assertTrue(np.all(identities[:, 0] <= 31))
                for method, rows in enumerate(raw["selected_indices"]):
                    all_histogram = np.zeros((2, 128), dtype=np.int64)
                    late_histogram = np.zeros_like(all_histogram)
                    lag_histogram = np.zeros_like(all_histogram)
                    for index, (_, head, position) in enumerate(identities):
                        count = min(int(position) + 1, 32)
                        selected = rows[index, :count]
                        alignment.validate_selection(selected, int(position))
                        self.assertTrue(np.all(rows[index, count:] == -1))
                        all_histogram[head] += np.bincount(selected, minlength=128)
                        lag_histogram[head] += np.bincount(position - selected, minlength=128)
                        if position >= 32:
                            late_histogram[head] += np.bincount(selected, minlength=128)
                    np.testing.assert_array_equal(raw["selected_position_counts"][method, :, 0], all_histogram)
                    np.testing.assert_array_equal(raw["selected_position_counts"][method, :, 1], late_histogram)
                    np.testing.assert_array_equal(raw["selected_lag_counts"][method, :, 0], lag_histogram)
                early = identities[:, 2] <= 31
                np.testing.assert_allclose(raw["kept_attention_mass"][:, early], 1.0, atol=1e-15)
                np.testing.assert_array_equal(raw["output_error_squared"][:, early], 0.0)

    def test_raw_attention_metrics_match_direct_value_reconstruction_and_match_definitions(self):
        directory = self.output / "pca-17"
        with np.load(directory / "rawstats.npz", allow_pickle=False) as raw, \
                np.load(directory / "codes.npz", allow_pickle=False) as codes, \
                np.load(directory / "encoder_state.npz", allow_pickle=False) as state:
            index = int(np.flatnonzero((raw["identifiers"][:, 0] == 2) & (raw["identifiers"][:, 2] == 34))[0])
            q = self.q[2, 0, 34]
            k = self.k[2, 0, :35]
            v = self.v[2, 0, :35].astype(np.float64)
            probabilities = attention_probabilities(q, k)
            reference = probabilities @ v
            for method in range(len(alignment.METHODS)):
                selected = raw["selected_indices"][method, index]
                restricted = attention_probabilities(q, k[selected])
                difference = restricted @ v[selected] - reference
                self.assertAlmostEqual(raw["kept_attention_mass"][method, index], probabilities[selected].sum())
                self.assertAlmostEqual(raw["output_error_squared"][method, index], difference @ difference)
            for pi, prime in enumerate((2, 3)):
                depths = common_padic_depth(codes[f"p{prime}_query_codes"][2, 0, 34],
                    codes[f"p{prime}_key_codes"][2, 0, :35], tuple(state[f"p{prime}_digits"]), prime)
                self.assertEqual(raw["causal_depth_one_match_counts"][pi, index], np.count_nonzero(depths >= 1))
                self.assertEqual(raw["optional_older_depth_one_match_counts"][pi, index], np.count_nonzero(depths[:27] >= 1))

    def test_evaluation_and_padding_cannot_change_fitted_state_or_calibration_geometry(self):
        q, k, v, lengths = fixture()
        q[2:] = 100 * q[2:] + 30
        k[2:] = -100 * k[2:] - 70
        # Padding in the calibration chunks is not a fitting observation.
        for chunk in (0, 1):
            q[chunk, :, lengths[chunk]:] = 1e7
            k[chunk, :, lengths[chunk]:] = -1e7
        changed = self.directory / "changed-trace.npz"
        np.savez(changed, queries=q, keys=k, values=v, valid_lengths=lengths)
        second = alignment.run(changed, self.directory / "second")
        self.assertNotEqual(second["config"]["trace_sha256"], self.result["config"]["trace_sha256"])
        for before, after in zip(self.result["variants"], second["variants"]):
            self.assertEqual(before["calibration_geometry"], after["calibration_geometry"])
            self.assertEqual(before["embedding_sha256"], after["embedding_sha256"])
            with np.load(self.output / before["name"] / "encoder_state.npz", allow_pickle=False) as a, \
                    np.load(self.directory / "second" / after["name"] / "encoder_state.npz", allow_pickle=False) as b:
                self.assertEqual(a.files, b.files)
                for name in a.files:
                    np.testing.assert_array_equal(a[name], b[name])

    def test_existing_output_is_never_overwritten(self):
        before = (self.output / "completed.json").read_bytes()
        with self.assertRaises(FileExistsError):
            alignment.run(self.trace, self.output)
        self.assertEqual((self.output / "completed.json").read_bytes(), before)


class AlignmentValidationTests(unittest.TestCase):
    def test_invalid_or_nonfixed_trace_and_missing_values_fail_before_output_creation(self):
        q, k, v, lengths = fixture(721)
        cases = [
            {"queries": q, "keys": k},
            {"queries": q[:-1], "keys": k[:-1], "values": v[:-1]},
            {"queries": q[:, :, :127], "keys": k[:, :, :127], "values": v[:, :, :127]},
            {"queries": q, "keys": k, "values": v, "valid_lengths": np.full(32, 32)},
            {"queries": q, "keys": k, "values": v, "valid_lengths": np.full(32, 129)},
        ]
        holes = np.ones((32, 128), dtype=bool)
        holes[:, 2] = False
        cases.append({"queries": q, "keys": k, "values": v, "attention_mask": holes})
        mask = np.arange(128)[None, :] < lengths[:, None]
        cases.append({"queries": q, "keys": k, "values": v, "valid_lengths": lengths + 1, "attention_mask": mask})
        nonfinite = q.copy()
        nonfinite[3, 0, 5, 0] = np.nan
        cases.append({"queries": nonfinite, "keys": k, "values": v})
        with tempfile.TemporaryDirectory(prefix="routing-invalid-") as temporary:
            root = Path(temporary)
            for index, arrays in enumerate(cases):
                trace, output = root / f"trace-{index}.npz", root / f"output-{index}"
                np.savez(trace, **arrays)
                with self.assertRaises((ValueError, TypeError)):
                    alignment.run(trace, output)
                self.assertFalse(output.exists())

    def test_timeout_leaves_failed_marker_and_no_completion_and_cap_cannot_expand(self):
        q, k, v, lengths = fixture(173)
        with tempfile.TemporaryDirectory(prefix="routing-time-limit-") as temporary:
            root = Path(temporary)
            trace = root / "trace.npz"
            np.savez(trace, queries=q, keys=k, values=v, valid_lengths=lengths)
            for value in (-1, 0, 601, float("inf"), float("nan"), True):
                with self.assertRaises(ValueError):
                    alignment.run(trace, root / "invalid", max_seconds=value)
            with patch.object(alignment.time, "perf_counter", side_effect=[0.0, 2.0, 3.0]):
                with self.assertRaises(TimeoutError):
                    alignment.run(trace, root / "limited", max_seconds=1.0)
            failed = json.loads((root / "limited" / "failed.json").read_text())
            self.assertEqual(failed["error_type"], "TimeoutError")
            self.assertFalse((root / "limited" / "completed.json").exists())

    def test_occupancy_entropy_and_calibration_mean_projection_have_explicit_definitions(self):
        codes = np.array([[[0, 1, 1, 1, 3]]], dtype=np.uint8)
        rows, histogram = alignment.occupancy(codes, np.array([[True, True, True, True, False]]), 4)
        self.assertEqual(rows[0]["occupied_codes"], 2)
        self.assertEqual(rows[0]["tokens"], 4)
        self.assertAlmostEqual(rows[0]["entropy_bits"], -(.25 * np.log2(.25) + .75 * np.log2(.75)))
        np.testing.assert_array_equal(histogram, [[1, 3, 0, 0]])
        q = np.array([[[[5., 0.], [5., 0.], [1e9, 1e9]]]])
        k = -q
        mask = np.array([[True, True, False]])
        projection = np.eye(2)[None]
        row = alignment.calibration_geometry(q, k, mask, projection)[0]
        self.assertEqual(row["mean_separation_l2"], 10)
        self.assertEqual(row["first_direction_squared_fraction"], 1)
        self.assertEqual(row["calibration_tokens_per_role"], 2)
        row = alignment.calibration_geometry(q, q, mask, projection)[0]
        self.assertEqual(row["mean_separation_l2"], 0)
        self.assertEqual(row["all_directions_squared_fraction"], 0)

    def test_stable_restricted_softmax_when_full_mass_underflows(self):
        q = np.array([1000., 0.])
        k = np.array([[1000., 0.], [-1000., 0.], [-999., 0.]])
        probabilities = attention_probabilities(q, k)
        np.testing.assert_array_equal(probabilities, [1., 0., 0.])
        restricted = attention_probabilities(q, k[[1, 2]])
        self.assertTrue(np.all(np.isfinite(restricted)))
        self.assertAlmostEqual(restricted.sum(), 1)
        self.assertGreater(restricted[1], .999)


if __name__ == "__main__":
    unittest.main()
