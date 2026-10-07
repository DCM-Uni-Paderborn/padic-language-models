"""One fixed, state-capped PCA flat-code control on the archived 135M trace.

Thirteen centroids replace the previous exemplar dictionary. PCA and existing
binary/recency selections remain frozen. No model forward or training occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from padic_lm import flat_routing, routing
from padic_lm.flat_routing import FlatKeyCodebook
from padic_lm.routing import attention_probabilities
import alignment_access
import routing_alignment
import routing_diagnostic
from alignment_access import collect_unions, validate_gqa_capture, verify_completion
from routing_diagnostic import array_hash, file_hash, load_trace

METHODS = ("recency", "padic_2", "trie_2", "flat_euclidean", "flat_dot_product")
MAX_CENTROIDS = 13
STATE_CAP = 16004
BUDGET, RECENT = 32, 8
TRACE_HASH = "1410127348792eec13aa013d38f4710bed49855d918a509c6481452e1b639e9e"
PARENT_HASH = "41782ff6103e9936415faa88f4eb4838da7744828229854339879f82cf18a012"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_STATE_HASH = "d9bcdca3b4d83d1d8505eb3b0e85f0c21dc1c2641e72109b3b4897a965500fb1"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")


def fit_capped(keys, center, projection, mask, *, byte_cap=STATE_CAP):
    book = FlatKeyCodebook.fit(keys, center=center, projection=projection,
                               max_centroids=MAX_CENTROIDS, seed=17,
                               max_iterations=20, mask=mask)
    if book.center.tobytes() != center.tobytes() or book.projection.tobytes() != projection.tobytes():
        raise AssertionError("frozen encoder buffers changed")
    if book.state_bytes > byte_cap or np.any(book.centroid_counts > MAX_CENTROIDS):
        raise AssertionError("flat codebook exceeds the fixed numeric-state cap")
    return book


def readouts(query, keys, values, selections):
    """Original full-dimensional FP64 teacher and restricted real readout."""
    probabilities = attention_probabilities(query, keys)
    real_values = values.astype(np.float64)
    reference = probabilities @ real_values
    masses, errors = [], []
    for selected in selections:
        selected = np.asarray(selected)
        masses.append(float(probabilities[selected].sum()))
        restricted = attention_probabilities(query, keys[selected])
        difference = restricted @ real_values[selected] - reference
        errors.append(float(difference @ difference))
    return np.asarray(masses), np.asarray(errors), float(reference @ reference)


def summaries(ids, masses, errors, reference, agreements, heads):
    result = []
    for stratum in ("all_queries", "affected_queries"):
        active = np.ones(len(ids), dtype=bool) if stratum == "all_queries" else ids[:, 2] >= BUDGET
        for index, method in enumerate(METHODS):
            groups = []
            for head in (None, *range(heads)):
                chosen = active if head is None else active & (ids[:, 1] == head)
                numerator, denominator = float(errors[index, chosen].sum()), float(reference[chosen].sum())
                groups.append({"head": head, "queries": int(chosen.sum()),
                    "kept_mass_mean": float(masses[index, chosen].mean()),
                    "value_output_nrmse": float(np.sqrt(numerator / denominator)) if denominator else
                        (0.0 if numerator == 0 else None),
                    "error_squared_sum": numerator, "reference_squared_sum": denominator,
                    "recency_exact_agreement_fraction": float(agreements[index, chosen].mean())})
            result.append({"method": method, "stratum": stratum, "all_heads": groups[0], "per_head": groups[1:]})
    return result


def run(parent: Path, trace: Path, model_manifest: Path, plan: Path, output: Path, *, max_seconds=600.0):
    if isinstance(max_seconds, (bool, np.bool_)) or not np.isfinite(max_seconds) or not 0 < max_seconds <= 600:
        raise ValueError("CPU and wall limits must be positive and at most 600 seconds")
    wall, cpu = time.perf_counter(), time.process_time()
    parent, trace, model_manifest, plan, output = map(lambda p: Path(p).resolve(),
                                                    (parent, trace, model_manifest, plan, output))
    if output.exists():
        raise FileExistsError("refusing to overwrite existing output")
    paths = (Path(__file__).resolve(), Path(flat_routing.__file__), Path(routing.__file__),
             Path(alignment_access.__file__), Path(routing_alignment.__file__), Path(routing_diagnostic.__file__))
    source_bytes = {p.name: p.read_bytes() for p in paths}
    plan_bytes = plan.read_bytes()
    completed, parent_digest = verify_completion(parent)
    if parent_digest != PARENT_HASH or file_hash(trace) != TRACE_HASH:
        raise ValueError("this gate requires the exact frozen parent and original trace")
    config = json.loads((parent / "config.json").read_text())
    parent_results = json.loads((parent / "results.json").read_text())
    manifest = json.loads(model_manifest.read_text())
    model_digest = file_hash(model_manifest)
    if (config != parent_results["config"] or config["trace_sha256"] != TRACE_HASH
            or manifest.get("trace_sha256") != TRACE_HASH or manifest.get("model_revision") != MODEL_REVISION
            or manifest.get("model_loaded_state_sha256") != MODEL_STATE_HASH
            or manifest.get("query_heads") != 9 or manifest.get("physical_kv_heads") != 3
            or manifest.get("gqa_groups") != 3
            or manifest.get("layer_index") != 0 or manifest.get("sequence_length") != 128
            or manifest.get("calibration_chunk_indices") != [0, 1]
            or manifest.get("evaluation_chunk_indices") != list(range(2, 32))
            or config["budget"] != BUDGET or config["recent_window"] != RECENT):
        raise ValueError("parent/model provenance differs from the fixed gate")
    for name in ("flat_routing.py", "routing.py", "routing_alignment.py", "routing_diagnostic.py"):
        if hashlib.sha256(source_bytes[name]).hexdigest() != config["code_sha256"][name]:
            raise ValueError(f"executed dependency differs from frozen parent: {name}")
    queries, keys, values, lengths, _ = load_trace(trace)
    if queries.shape != (32, 9, 128, 64) or values is None or values.shape != queries.shape:
        raise ValueError("fixed gate requires original 135M capture shape")
    head_to_kv = validate_gqa_capture(keys, values, manifest)
    if lengths.tolist() != config["valid_lengths"]:
        raise ValueError("trace valid lengths differ from parent")
    with np.load(parent / "pca-17/encoder_state.npz", allow_pickle=False) as state:
        center, projection = state["p2_center"].copy(), state["p2_projection"].copy()
        binary_arrays = {name: state[name].copy() for name in state.files if name.startswith("p2_")}
    if sum(a.nbytes for a in binary_arrays.values()) != STATE_CAP:
        raise ValueError("archived binary buffers differ from numeric-state cap")
    with np.load(parent / "pca-17/rawstats.npz", allow_pickle=False) as original:
        prior = {name: original[name].copy() for name in original.files}
    parent_methods = prior["methods"].tolist()
    control_indices = [parent_methods.index(method) for method in METHODS[:3]]
    ids = prior["identifiers"]
    valid = np.arange(128)[None] < lengths[:, None]
    proposed = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "parent": str(parent),
        "parent_completion_sha256": parent_digest, "trace": str(trace), "trace_sha256": TRACE_HASH,
        "model_manifest": str(model_manifest), "model_manifest_sha256": model_digest,
        "model_revision": MODEL_REVISION, "model_loaded_state_sha256": manifest["model_loaded_state_sha256"],
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(), "methods": list(METHODS),
        "calibration_chunks": [0, 1], "evaluation_chunks": list(range(2, 32)),
        "max_centroids": MAX_CENTROIDS, "seed": 17, "max_lloyd_iterations": 20,
        "numeric_state_cap_bytes": STATE_CAP, "numeric_state_upper_bound_bytes": 15891,
        "budget": BUDGET, "recent_window": RECENT, "affected_start_position": 32,
        "fit_objective": "calibration-key squared reconstruction error in the frozen two-dimensional PCA space",
        "primary_score": "flat_euclidean", "secondary_score": "flat_dot_product",
        "teacher": "float64 causal softmax(Q K^T / sqrt(64)) from original post-RoPE capture",
        "readout": "original real V, restricted teacher softmax; before output projection; no value reallocation",
        "query_head_to_physical_kv_head": head_to_kv.tolist(),
        "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in source_bytes.items()},
        "max_cpu_seconds": max_seconds, "max_wall_seconds": max_seconds,
        "python": platform.python_version(), "numpy": np.__version__, "command": list(sys.argv),
    }
    output.mkdir(parents=True)
    (output / "sources").mkdir()
    for name, content in source_bytes.items():
        (output / "sources" / name).write_bytes(content)
    (output / "prospective_plan.txt").write_bytes(plan_bytes)
    write_json(output / "config.json", proposed)

    def check_time():
        if time.perf_counter() - wall > max_seconds or time.process_time() - cpu > max_seconds:
            raise TimeoutError("state-capped control exceeded CPU or wall limit")

    try:
        check_time()
        book = fit_capped(keys[:2], center, projection, valid[:2])
        codes, projected = book.encode_keys(keys), book.project_queries(queries)
        saved_book = {n: a.tobytes() for n, a in book.state_arrays.items()}
        selected = np.full((len(METHODS), len(ids), BUDGET), -1, dtype=np.int16)
        selected[:3] = prior["selected_indices"][control_indices]
        masses, errors = np.empty((len(METHODS), len(ids))), np.empty((len(METHODS), len(ids)))
        reference = np.empty(len(ids))
        agreement = np.empty_like(masses, dtype=bool)
        for row, (chunk, head, position) in enumerate(ids):
            if row % 512 == 0:
                check_time()
            chunk, head, position = int(chunk), int(head), int(position)
            count = min(position + 1, BUDGET)
            for index, score in enumerate(("euclidean", "dot_product"), start=3):
                selected[index, row, :count] = book.select_causal(projected[chunk, head, position],
                    codes[chunk, head], position, BUDGET, RECENT, head=head, score_kind=score)
            selections = [indices[row, :count] for indices in selected]
            masses[:, row], errors[:, row], reference[row] = readouts(queries[chunk, head, position],
                keys[chunk, head, :position + 1], values[chunk, head, :position + 1], selections)
            agreement[:, row] = [np.array_equal(s, selections[0]) for s in selections]
        # Exact same-runtime controls are mandatory, not a tolerance-based claim.
        np.testing.assert_array_equal(masses[:3], prior["kept_attention_mass"][control_indices])
        np.testing.assert_array_equal(errors[:3], prior["output_error_squared"][control_indices])
        np.testing.assert_array_equal(reference, prior["output_reference_squared"])
        np.testing.assert_array_equal(selected[1], selected[2])
        if saved_book != {n: a.tobytes() for n, a in book.state_arrays.items()}:
            raise AssertionError("evaluation mutated the fitted codebook")
        unions = collect_unions(ids, selected, prior["selected_counts"], lengths=lengths,
            evaluation_chunks=list(range(2, 32)), head_to_kv=head_to_kv, budget=BUDGET, recent_window=RECENT)
        eval_mask = valid.copy()
        eval_mask[:2] = False
        cal_mask = valid.copy()
        cal_mask[2:] = False
        occupancy = {split: routing_alignment.occupancy(codes, mask, 256)[0]
                     for split, mask in (("calibration", cal_mask), ("evaluation", eval_mask))}
        np.savez_compressed(output / "encoder_state.npz", **book.state_arrays)
        np.savez_compressed(output / "key_codes.npz", key_codes=codes, valid_lengths=lengths)
        np.savez_compressed(output / "rawstats.npz", methods=np.asarray(METHODS), identifiers=ids,
            identifier_columns=np.asarray(["chunk", "head", "position"]), selected_indices=selected,
            selected_counts=prior["selected_counts"], kept_attention_mass=masses,
            output_error_squared=errors, output_reference_squared=reference,
            recency_exact_agreement=agreement, **unions)
        memory = {"state_array_bytes": {n: int(a.nbytes) for n, a in book.state_arrays.items()},
            "numeric_state_bytes": book.state_bytes, "binary_numeric_state_bytes": STATE_CAP,
            "key_code_bytes": int(codes.nbytes), "evaluation_key_code_bytes": int(eval_mask.sum()) * 9,
            "numeric_state_plus_key_code_bytes": book.state_bytes + int(codes.nbytes),
            "binary_numeric_state_plus_key_code_bytes": STATE_CAP + int(codes.nbytes),
            "active_centroids": book.centroid_counts.tolist(), "unique_calibration_keys": book.unique_key_counts.tolist(),
            "calibration_counts": book.calibration_counts.tolist(), "lloyd_iterations": book.iterations.tolist(),
            "score_variants_share_state_and_key_payload": True, "dot_offset_cached_bytes": 0}
        fit_error = []
        projected_keys = book.project_queries(keys[:2])
        decoded = book.decode_keys(codes[:2])
        for head in range(9):
            delta = (projected_keys[:, head] - decoded[:, head])[valid[:2]]
            fit_error.append(float(np.mean(np.sum(delta * delta, axis=-1))))
        checks = {"unchanged_parent_control_masks_and_metrics_exact": True, "same_pca_buffers_exact": True,
            "numeric_state_within_cap": True, "all_selections_causal_exact_budget_recent": True,
            "padic_trie_masks_exact": True, "group_mapping_capture_bitwise_verified": True,
            "logical_group_union_bounds_verified": True, "codebook_unchanged_during_evaluation": True}
        result = {"status": "state_capped_flat_completed", "config": proposed, "checks": checks,
            "memory": memory, "calibration_projected_key_mse_per_head": fit_error, "occupancy": occupancy,
            "summaries": summaries(ids, masses, errors, reference, agreement, 9),
            "logical_access_summaries": alignment_access.summarize(METHODS, unions, BUDGET),
            "scope": "Reused-validation component diagnostic; no model forwarding, token NLL, confirmation, native IO, cache compression or speed claim",
            "elapsed_wall_seconds": time.perf_counter() - wall, "elapsed_cpu_seconds": time.process_time() - cpu}
        write_json(output / "results.json", result)
        check_time()
        if (file_hash(trace) != TRACE_HASH or file_hash(model_manifest) != model_digest
                or verify_completion(parent)[1] != parent_digest or plan.read_bytes() != plan_bytes
                or any(p.read_bytes() != source_bytes[p.name] for p in paths)):
            raise RuntimeError("input/source/prospective plan changed during execution")
        artifacts = sorted(p for p in output.rglob("*") if p.is_file())
        final = {"status": result["status"], "trace_sha256": TRACE_HASH,
            "parent_completion_sha256": parent_digest, "source_sha256": proposed["source_sha256"],
            "artifact_sha256": {str(p.relative_to(output)): file_hash(p) for p in artifacts},
            "artifact_bytes": {str(p.relative_to(output)): p.stat().st_size for p in artifacts},
            "elapsed_wall_seconds": time.perf_counter() - wall, "elapsed_cpu_seconds": time.process_time() - cpu}
        check_time()
        write_json(output / "completed.json", final)
        (output / "completed.sha256").write_text(file_hash(output / "completed.json") + "\n")
        return result
    except Exception as error:
        (output / "completed.json").unlink(missing_ok=True)
        (output / "completed.sha256").unlink(missing_ok=True)
        write_json(output / "failed.json", {"status": "state_capped_flat_failed", "error_type": type(error).__name__,
            "message": str(error), "elapsed_wall_seconds": time.perf_counter() - wall,
            "elapsed_cpu_seconds": time.process_time() - cpu})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "trace", "model-manifest", "plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args()
    result = run(args.parent, args.trace, args.model_manifest, args.plan, args.output, max_seconds=args.max_seconds)
    print(json.dumps({"status": result["status"], "memory": result["memory"],
        "affected": [{"method": r["method"], **r["all_heads"]} for r in result["summaries"]
            if r["stratum"] == "affected_queries"]}, indent=2))


if __name__ == "__main__":
    main()
