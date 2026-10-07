"""Independent logical GQA-union and immutable-input provenance fixtures."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
import alignment_access as access


def small_capture():
    physical_keys = np.array([[[[1., 2.], [3., 4.], [5., 6.]],
                               [[10., 20.], [30., 40.], [50., 60.]]]])
    physical_values = -physical_keys
    keys = np.repeat(physical_keys, 2, axis=1)
    values = np.repeat(physical_values, 2, axis=1)
    manifest = {"query_heads": 4, "physical_kv_heads": 2, "gqa_groups": 2}
    return keys, values, manifest


def selection_fixture():
    # Query heads 0/1 share physical KV0. Each keeps 2 keys including the
    # shared current/recent key 2, while {0,2} union {1,2} contains 3 keys.
    identifiers = np.array([(0, head, position) for head in range(2) for position in range(3)], dtype=np.int32)
    selected = np.full((1, len(identifiers), 2), -1, dtype=np.int16)
    for index, (_, head, position) in enumerate(identifiers):
        if position == 0:
            selected[0, index, 0] = 0
        elif position == 1:
            selected[0, index] = [0, 1]
        else:
            selected[0, index] = [head, 2]
    arguments = {"lengths": np.array([3]), "evaluation_chunks": [0],
                 "head_to_kv": np.array([0, 0]), "budget": 2, "recent_window": 1}
    return identifiers, selected, np.minimum(identifiers[:, 2] + 1, 2), arguments


def completed_fixture(base: Path) -> tuple[Path, Path, Path]:
    """Build synthetic frozen-schema files without loading an LLM or dataset."""
    directory = base / "alignment"
    directory.mkdir()
    (directory / "sources").mkdir()
    rng = np.random.default_rng(882)
    physical_keys = rng.normal(size=(32, 3, 128, 2)).astype(np.float32)
    physical_values = rng.normal(size=(32, 3, 128, 2)).astype(np.float32)
    keys, values = np.repeat(physical_keys, 3, axis=1), np.repeat(physical_values, 3, axis=1)
    queries = rng.normal(size=keys.shape).astype(np.float32)
    lengths = np.full(32, 33, dtype=np.int32)
    trace = base / "trace.npz"
    np.savez_compressed(trace, queries=queries, keys=keys, values=values, valid_lengths=lengths)
    trace_digest = access.file_hash(trace)
    source = directory / "sources" / "routing_alignment.py"
    source.write_text("fixture only: no real model execution\n")
    source_hashes = {source.name: access.file_hash(source)}
    config = {"trace_sha256": trace_digest, "budget": 32, "recent_window": 8,
              "calibration_chunks": [0, 1], "evaluation_chunks": list(range(2, 32)),
              "methods": list(access.METHODS), "qk_shape": list(queries.shape),
              "value_shape": list(values.shape), "valid_lengths": lengths.tolist(),
              "code_sha256": source_hashes, "code_snapshots": {source.name: "sources/" + source.name}}
    access.write_json(directory / "config.json", config)
    access.write_json(directory / "results.json", {"status": "alignment_diagnostic_completed", "config": config})
    identifiers = np.array([(chunk, head, position) for chunk in range(2, 32)
                            for head in range(9) for position in range(33)], dtype=np.int32)
    selected = np.full((len(access.METHODS), len(identifiers), 32), -1, dtype=np.int16)
    counts = np.minimum(identifiers[:, 2] + 1, 32)
    for index, (_, head, position) in enumerate(identifiers):
        count = int(counts[index])
        for method in range(len(access.METHODS)):
            choice = list(range(int(position) + 1))
            if position == 32:
                choice.remove(0 if method == 0 else int(head) % 3)
            selected[method, index, :count] = choice
    for family in access.FAMILIES:
        variant = directory / family
        variant.mkdir()
        order = rng.permutation(len(identifiers))
        np.savez_compressed(variant / "rawstats.npz", methods=np.asarray(access.METHODS),
                            identifier_columns=np.asarray(["chunk", "head", "position"]),
                            identifiers=identifiers[order], selected_indices=selected[:, order], selected_counts=counts[order])
    artifacts = sorted(path for path in directory.rglob("*") if path.is_file())
    completion = {"status": "alignment_diagnostic_completed", "trace_sha256": trace_digest,
                  "code_sha256": source_hashes,
                  "artifact_sha256": {str(path.relative_to(directory)): access.file_hash(path) for path in artifacts},
                  "artifact_bytes": {str(path.relative_to(directory)): path.stat().st_size for path in artifacts}}
    access.write_json(directory / "completed.json", completion)
    (directory / "completed.sha256").write_text(access.file_hash(directory / "completed.json") + "\n")
    model_manifest = base / "model-manifest.json"
    access.write_json(model_manifest, {"model": "fixture/model", "model_revision": "a" * 40,
                                      "model_loaded_state_sha256": "b" * 64, "layer_index": 0,
                                      "sequence_length": 128, "calibration_chunk_indices": [0, 1],
                                      "evaluation_chunk_indices": list(range(2, 32)),
                                      "trace_sha256": trace_digest, "query_heads": 9,
                                      "physical_kv_heads": 3, "gqa_groups": 3})
    return directory, trace, model_manifest


class GroupCaptureTests(unittest.TestCase):
    def test_pinned_repeat_layout_and_both_captures_are_verified(self):
        keys, values, manifest = small_capture()
        np.testing.assert_array_equal(access.validate_gqa_capture(keys, values, manifest), [0, 0, 1, 1])
        for role in ("keys", "values"):
            changed_keys, changed_values = keys.copy(), values.copy()
            (changed_keys if role == "keys" else changed_values)[0, 1, 0, 0] += 1
            with self.assertRaises(ValueError):
                access.validate_gqa_capture(changed_keys, changed_values, manifest)

    def test_mixed_head_grouping_and_wrong_repeat_factor_are_rejected(self):
        keys, values, manifest = small_capture()
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(keys, values, manifest, head_to_kv=np.array([0, 1, 0, 1]))
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(keys, values, {**manifest, "query_head_to_physical_kv_head": [0, 1, 0, 1]})
        mixed = keys[:, [0, 2, 1, 3]]
        mixed_values = values[:, [0, 2, 1, 3]]
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(mixed, mixed_values, manifest)
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(keys, values, {**manifest, "gqa_groups": 3})

    def test_signed_zero_bit_mismatch_and_nonfinite_capture_are_rejected(self):
        keys, values, manifest = small_capture()
        keys[:, :, 0, 0] = 0
        keys[:, 1, 0, 0] = -0.0
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(keys, values, manifest)
        keys, values, manifest = small_capture()
        values[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            access.validate_gqa_capture(keys, values, manifest)


class UnionSelectionTests(unittest.TestCase):
    def test_independent_union_three_despite_two_per_head_and_shared_recent(self):
        identities, selections, counts, arguments = selection_fixture()
        raw = access.collect_unions(identities, selections, counts, **arguments)
        np.testing.assert_array_equal(raw["union_identifiers"], [[0, 0, 0], [0, 0, 1], [0, 0, 2]])
        np.testing.assert_array_equal(raw["union_counts"], [[1, 2, 3]])
        np.testing.assert_array_equal(raw["shared_recent_union_bounds"], [1, 2, 3])
        summary = access.summarize(("fixture",), raw, 2)[1]
        self.assertEqual(summary["selected_position_union_mean"], 3)
        self.assertEqual(summary["per_query_head_selected_positions_mean"], 2)

    def test_randomized_raw_row_order_has_identical_joined_unions(self):
        identities, selections, counts, arguments = selection_fixture()
        expected = access.collect_unions(identities, selections, counts, **arguments)
        order = np.array([5, 0, 3, 1, 4, 2])
        actual = access.collect_unions(identities[order], selections[:, order], counts[order], **arguments)
        for name in ("union_identifiers", "union_counts", "shared_recent_union_bounds"):
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_padding_is_filtered_without_treating_minus_one_as_a_key(self):
        identities, selections, counts, arguments = selection_fixture()
        # Padding is deliberately placed before the valid one-key prefix.
        selections[0, identities[:, 2] == 0] = [-1, 0]
        raw = access.collect_unions(identities, selections, counts, **arguments)
        self.assertEqual(raw["union_counts"][0, 0], 1)

    def test_future_duplicate_out_of_order_bad_padding_and_missing_recent_rejected(self):
        identities, selections, counts, arguments = selection_fixture()
        for invalid in ([0, 3], [2, 2], [2, 0], [-2, 2], [0, 1]):
            changed = selections.copy()
            changed[0, 2] = invalid
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                access.collect_unions(identities, changed, counts, **arguments)

    def test_missing_duplicate_mixed_ids_and_bad_counts_mapping_rejected(self):
        identities, selections, counts, arguments = selection_fixture()
        for changed in (identities[:-1], np.concatenate((identities[:-1], identities[:1])),
                        identities + np.array([0, 1, 0])):
            with self.assertRaises(ValueError):
                access.collect_unions(changed, selections[:, :len(changed)], counts[:len(changed)], **arguments)
        changed_counts = counts.copy()
        changed_counts[2] = 1
        with self.assertRaises(ValueError):
            access.collect_unions(identities, selections, changed_counts, **arguments)
        for mapping in (np.array([0, 1, 0, 1]), np.array([-1, -1])):
            with self.assertRaises(ValueError):
                access.collect_unions(identities, selections, counts, **{**arguments, "head_to_kv": mapping})

    def test_shared_recent_bound_when_recent_exceeds_budget(self):
        identities, selections, counts, arguments = selection_fixture()
        selections[0, 2] = [1, 2]
        raw = access.collect_unions(identities, selections, counts, **{**arguments, "recent_window": 9})
        np.testing.assert_array_equal(raw["union_counts"], [[1, 2, 2]])
        np.testing.assert_array_equal(raw["shared_recent_union_bounds"], [1, 2, 2])


class AccountingPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="alignment-access-check-")
        cls.base = Path(cls.temporary.name)
        cls.directory, cls.trace, cls.model_manifest = completed_fixture(cls.base)
        cls.before = access.file_hash(cls.directory / "completed.json")
        cls.output = cls.base / "accounting"
        cls.result = access.run(cls.directory, cls.trace, cls.model_manifest, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_fixture_pins_inputs_sources_and_logical_only_scope(self):
        manifest = self.result["manifest"]
        self.assertEqual(manifest["query_head_to_physical_kv_head"], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(manifest["alignment_completion_sha256"], self.before)
        self.assertEqual(manifest["trace_sha256"], access.file_hash(self.trace))
        self.assertEqual(manifest["model_manifest_sha256"], access.file_hash(self.model_manifest))
        self.assertIn("all real K/V remain retained", manifest["scope"])
        self.assertEqual(access.file_hash(self.directory / "completed.json"), self.before)
        for name, digest in manifest["source_sha256"].items():
            self.assertEqual(access.file_hash(self.output / "sources" / name), digest)
        completion = json.loads((self.output / "completed.json").read_text())
        self.assertEqual((self.output / "completed.sha256").read_text().strip(), access.file_hash(self.output / "completed.json"))
        for name, digest in completion["artifact_sha256"].items():
            self.assertEqual(access.file_hash(self.output / name), digest)

    def test_affected_slice_excludes_position31_and_saves_actual_row_ids(self):
        for variant in self.result["variants"]:
            late = [row for row in variant["summaries"] if row["stratum"] == "affected_queries"]
            self.assertEqual(len(late), 7)
            self.assertTrue(all(row["logical_group_query_count"] == 30 * 3 for row in late))
            self.assertTrue(all(row["per_query_head_count"] == 30 * 9 for row in late))
            self.assertTrue(all(row["per_query_head_selected_positions_mean"] == 32 for row in late))
            self.assertEqual(late[0]["selected_position_union_mean"], 32)
            self.assertTrue(all(row["selected_position_union_mean"] == 33 for row in late[1:]))
        with np.load(self.output / "union_counts.npz", allow_pickle=False) as raw:
            ids = raw["pca_17_union_identifiers"]
            self.assertEqual(ids.shape, (30 * 3 * 33, 3))
            query_ids = raw["pca_17_query_identifiers"]
            self.assertEqual(query_ids.shape, (30 * 9 * 33, 3))
            late = ids[:, 2] == 32
            np.testing.assert_array_equal(raw["pca_17_union_counts"][0, late], 32)
            np.testing.assert_array_equal(raw["pca_17_union_counts"][1:, late], 33)

    def test_existing_output_and_wrong_trace_model_hash_are_rejected(self):
        with self.assertRaises(FileExistsError):
            access.run(self.directory, self.trace, self.model_manifest, self.output)
        changed = self.base / "wrong-model.json"
        model = json.loads(self.model_manifest.read_text())
        model["trace_sha256"] = "0" * 64
        access.write_json(changed, model)
        with self.assertRaises(ValueError):
            access.run(self.directory, self.trace, changed, self.base / "wrong-output")
        self.assertFalse((self.base / "wrong-output").exists())

    def test_tampered_completion_artifact_and_unlisted_files_rejected(self):
        original = (self.directory / "config.json").read_bytes()
        try:
            (self.directory / "config.json").write_bytes(original + b" ")
            with self.assertRaises(ValueError):
                access.verify_completion(self.directory)
        finally:
            (self.directory / "config.json").write_bytes(original)
        unexpected = self.directory / "unexpected.txt"
        try:
            unexpected.write_text("not pinned")
            with self.assertRaises(ValueError):
                access.verify_completion(self.directory)
        finally:
            unexpected.unlink()
        self.assertEqual(access.verify_completion(self.directory)[1], self.before)

    def test_unresolved_revision_and_wrong_model_split_are_rejected(self):
        model = json.loads(self.model_manifest.read_text())
        for number, fields in enumerate(({"model_revision": "main"},
                                          {"model_loaded_state_sha256": "unresolved"},
                                          {"evaluation_chunk_indices": [2]},
                                          {"sequence_length": 512})):
            changed = self.base / f"invalid-model-{number}.json"
            access.write_json(changed, {**model, **fields})
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                access.run(self.directory, self.trace, changed, self.base / f"invalid-output-{number}")


if __name__ == "__main__":
    unittest.main()
