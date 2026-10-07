"""Bounded encoder diagnostic on the archived first-layer Q/K/V trace.

The fixed development gate compares PCA and three shared random embeddings,
finite p=2/p=3 product balls, independent tries, a calibration-matched flat
one-byte key codebook and recency. It neither trains nor forwards an LLM.
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "experiments"))
from padic_lm import flat_routing, routing
from padic_lm.flat_routing import FlatKeyCodebook
from padic_lm.routing import (
    PrefixTree, QuantileCodebook, attention_probabilities, common_padic_depth,
    pack_codes, select_causal_padic, unpack_codes,
)
import routing_diagnostic
from routing_diagnostic import array_hash, file_hash, load_trace


CALIBRATION = (0, 1)
EVALUATION = tuple(range(2, 32))
BUDGET = 32
RECENT = 8
COORDINATES = 2
BIT_BUDGET = 8
FLAT_SEED = 17
FLAT_ITERATIONS = 20
FAMILIES = (("pca", 17), ("random", 17), ("random", 29), ("random", 43))
METHODS = ("recency", "flat_euclidean", "flat_dot_product", "padic_2", "trie_2", "padic_3", "trie_3")
PRIMES = (2, 3)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def occupancy(codes: np.ndarray, mask: np.ndarray, alphabet: int) -> tuple[list[dict], np.ndarray]:
    """Per-head histograms on explicitly selected valid token positions."""
    codes, mask = np.asarray(codes), np.asarray(mask, dtype=bool)
    if codes.ndim != 3 or mask.shape != (codes.shape[0], codes.shape[2]):
        raise ValueError("occupancy expects codes [chunks,heads,seq] and matching mask")
    if not np.any(mask) or np.any(codes < 0) or np.any(codes >= alphabet):
        raise ValueError("occupancy needs valid canonical codes and at least one token")
    rows, histograms = [], []
    for head in range(codes.shape[1]):
        histogram = np.bincount(codes[:, head][mask].astype(np.int64), minlength=alphabet)
        probabilities = histogram[histogram > 0] / histogram.sum()
        rows.append({"head": head, "tokens": int(histogram.sum()),
                     "occupied_codes": int(np.count_nonzero(histogram)), "alphabet": alphabet,
                     "occupied_fraction": float(np.count_nonzero(histogram) / alphabet),
                     "entropy_bits": float(-np.sum(probabilities * np.log2(probabilities)))})
        histograms.append(histogram)
    return rows, np.stack(histograms)


def calibration_geometry(queries: np.ndarray, keys: np.ndarray, mask: np.ndarray,
                         projection: np.ndarray) -> list[dict]:
    """Mean separation and squared projection fractions from calibration only."""
    rows = []
    for head in range(queries.shape[1]):
        query_mean = queries[:, head][mask].astype(np.float64).mean(axis=0)
        key_mean = keys[:, head][mask].astype(np.float64).mean(axis=0)
        difference = query_mean - key_mean
        separation_squared = float(difference @ difference)
        projected = difference @ projection[head]
        rows.append({"head": head, "calibration_tokens_per_role": int(mask.sum()),
                     "query_mean": query_mean.tolist(), "key_mean": key_mean.tolist(),
                     "mean_separation_l2": float(np.sqrt(separation_squared)),
                     "signed_first_direction_separation": float(projected[0]),
                     "first_direction_squared_fraction": float(projected[0] ** 2 / separation_squared)
                         if separation_squared else 0.0,
                     "all_directions_squared_fraction": float(projected @ projected / separation_squared)
                         if separation_squared else 0.0})
    return rows


def validate_selection(selected: np.ndarray, position: int) -> None:
    count = min(position + 1, BUDGET)
    if (selected.ndim != 1 or len(selected) != count or
            not np.array_equal(selected, np.unique(selected)) or
            np.any(selected < 0) or np.any(selected > position)):
        raise AssertionError("selection must be unique, chronological, causal and exact-budget")
    recent = min(RECENT, count)
    if not set(range(position + 1 - recent, position + 1)).issubset(selected.tolist()):
        raise AssertionError("selection omits a mandatory recent key")


def summarize_variant(identifiers: np.ndarray, masses: np.ndarray, errors: np.ndarray,
                      reference_squared: np.ndarray, agreement: np.ndarray,
                      overlap: np.ndarray, match_counts: np.ndarray,
                      optional_match_counts: np.ndarray, heads: int) -> list[dict]:
    summaries = []
    for stratum in ("all_queries", "affected_queries"):
        positions = np.ones(len(identifiers), dtype=bool)
        if stratum == "affected_queries":
            # Position 31 still has a full 32-key causal prefix.
            positions = identifiers[:, 2] >= BUDGET
        for method_index, method in enumerate(METHODS):
            groups = []
            for head in (None, *range(heads)):
                chosen = positions if head is None else positions & (identifiers[:, 1] == head)
                if not np.any(chosen):
                    continue
                error_sq = float(errors[method_index, chosen].sum())
                ref_sq = float(reference_squared[chosen].sum())
                row = {"head": head, "queries": int(chosen.sum()),
                       "kept_mass_mean": float(masses[method_index, chosen].mean()),
                       "value_output_nrmse": float(np.sqrt(error_sq / ref_sq)) if ref_sq
                           else (0.0 if error_sq == 0 else None),
                       "output_error_squared_sum": error_sq,
                       "output_reference_squared_sum": ref_sq,
                       "recency_exact_agreement_fraction": float(agreement[method_index, chosen].mean()),
                       "recency_selected_overlap_fraction": float(overlap[method_index, chosen].mean())}
                if method in {"padic_2", "trie_2", "padic_3", "trie_3"}:
                    prime_index = 0 if method.endswith("2") else 1
                    row["no_causal_depth_one_match_fraction"] = float((match_counts[prime_index, chosen] == 0).mean())
                    row["causal_depth_one_matches_mean"] = float(match_counts[prime_index, chosen].mean())
                    row["no_optional_older_depth_one_match_fraction"] = float((optional_match_counts[prime_index, chosen] == 0).mean())
                    row["optional_older_depth_one_matches_mean"] = float(optional_match_counts[prime_index, chosen].mean())
                groups.append(row)
            if groups:
                summaries.append({"method": method, "stratum": stratum,
                                  "all_heads": groups[0], "per_head": groups[1:]})
    return summaries


def run(input_path: Path, output: Path, *, max_seconds: float = 600.0) -> dict:
    """Execute only the fixed 32-chunk, 128-position development workload."""
    if (isinstance(max_seconds, (bool, np.bool_)) or not np.isfinite(max_seconds)
            or not 0 < max_seconds <= 600):
        raise ValueError("max_seconds must be positive and at most 600")
    start = time.perf_counter()
    cpu_start = time.process_time()
    input_path, output = Path(input_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    sources = {path.name: path.read_bytes() for path in (
        Path(__file__).resolve(), Path(routing.__file__).resolve(),
        Path(flat_routing.__file__).resolve(), Path(routing_diagnostic.__file__).resolve())}
    trace_hash = file_hash(input_path)
    queries, keys, values, lengths, source_metadata = load_trace(input_path)
    if values is None:
        raise ValueError("alignment gate requires captured values")
    chunks, heads, length, width = queries.shape
    if chunks != 32 or length != 128 or width < COORDINATES:
        raise ValueError("fixed gate requires exactly 32 chunks, 128 positions and at least two head dimensions")
    valid = np.arange(length)[None, :] < lengths[:, None]
    calibration_mask = valid[:2]
    evaluation_mask = valid.copy()
    evaluation_mask[:2] = False
    identifiers = np.asarray([(chunk, head, position) for chunk in EVALUATION
                              for head in range(heads) for position in range(int(lengths[chunk]))], dtype=np.int32)
    if not np.any(identifiers[:, 2] >= BUDGET):
        raise ValueError("at least one evaluation query must exceed the 32-key budget")
    config = {
        "input": str(input_path), "trace_sha256": trace_hash,
        "code_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
        "code_snapshots": {name: f"sources/{name}" for name in sources},
        "source_metadata": source_metadata, "qk_shape": list(queries.shape), "value_shape": list(values.shape),
        "qk_dtypes": [str(queries.dtype), str(keys.dtype)], "value_dtype": str(values.dtype),
        "valid_lengths": lengths.tolist(), "calibration_chunks": list(CALIBRATION),
        "evaluation_chunks": list(EVALUATION), "calibration_tokens_per_head_per_role": int(calibration_mask.sum()),
        "calibration_qk_sha256": array_hash(queries[:2], keys[:2], calibration_mask),
        "budget": BUDGET, "recent_window": RECENT, "coordinates": COORDINATES,
        "code_bit_budget": BIT_BUDGET, "primes": list(PRIMES),
        "embedding_families": [{"kind": kind, "seed": seed} for kind, seed in FAMILIES],
        "flat_max_centroids": 256, "flat_seed": FLAT_SEED, "flat_max_iterations": FLAT_ITERATIONS,
        "primary_flat_score": "negative squared Euclidean distance in the shared centered 2D embedding",
        "secondary_flat_score": "uncentered projected dot product, restoring center @ projection to query and assigned centroid",
        "secondary_flat_preresults_amendment_utc": "2026-10-03T20:53:12Z",
        "flat_score_parameter_sharing": "exact same centroids, key codes, center and projection; no extra fitting",
        "flat_dot_offset_policy": "center @ projection derived per selection call, not cached; zero additional stored state",
        "depth_one_match_definition": "all causal keys including current and mandatory recent; optional_older excludes recent min(8,budget,prefix)",
        "selection_ties": "recency", "includes_current_key": True,
        "affected_query_definition": "zero_based_position >= 32, causal_prefix_length > budget",
        "historical_slice_note": "position 31 is still full attention and is excluded from affected_queries",
        "attention_definition": "float64 softmax(Q K^T / sqrt(head_dim)), causal prefixes, no extra bias",
        "attention_scope": "NumPy reconstruction from captured post-RoPE vectors; not native BF16 score replay",
        "output_endpoint": "unprojected per-query-head weighted V; aggregate sqrt(sum squared error / sum squared reference norm)",
        "zero_reference_nrmse": "zero if both reference and error energy are zero; null if only reference energy is zero",
        "product_metric": "max_coordinate_finite_padic_distance; zero for equal finite coordinate",
        "methods": list(METHODS), "max_wall_seconds": float(max_seconds), "max_cpu_seconds": float(max_seconds),
        "python_version": platform.python_version(), "numpy_version": np.__version__, "platform": platform.platform(),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    for name, content in sources.items():
        (output / "sources" / name).write_bytes(content)
    write_json(output / "config.json", config)

    def check_time() -> None:
        if time.perf_counter() - start > max_seconds or time.process_time() - cpu_start > max_seconds:
            raise TimeoutError("fixed alignment diagnostic exceeded its CPU or wall-time cap")

    variant_results = []
    total_tree_checks = 0
    try:
        for kind, seed in FAMILIES:
            check_time()
            name = f"{kind}-{seed}"
            directory = output / name
            directory.mkdir()
            books = {prime: QuantileCodebook.fit(
                queries[:2], keys[:2], coordinates=COORDINATES, prime=prime,
                code_bit_budget=BIT_BUDGET, projection_kind=kind, seed=seed, mask=calibration_mask)
                for prime in PRIMES}
            reference_book = books[2]
            np.testing.assert_array_equal(reference_book.center, books[3].center)
            np.testing.assert_array_equal(reference_book.projection, books[3].projection)
            flat = FlatKeyCodebook.fit(keys[:2], center=reference_book.center,
                                       projection=reference_book.projection, mask=calibration_mask,
                                       max_centroids=256, seed=FLAT_SEED, max_iterations=FLAT_ITERATIONS)
            np.testing.assert_array_equal(flat.center, reference_book.center)
            np.testing.assert_array_equal(flat.projection, reference_book.projection)
            flat_codes = flat.encode_keys(keys)
            projected_queries = flat.project_queries(queries)
            states = {f"flat_{key}": value for key, value in flat.state_arrays.items()}
            code_arrays = {"flat_key_codes": flat_codes}
            code_occupancy, occupancy_arrays, memory = {}, {}, {}
            query_codes, key_codes = {}, {}
            for prime, book in books.items():
                query_codes[prime], key_codes[prime] = book.encode(queries), book.encode(keys)
                packed_q = pack_codes(query_codes[prime], book.digits, prime)
                packed_k = pack_codes(key_codes[prime], book.digits, prime)
                if packed_k.dtype != np.uint8 or packed_k.shape[-1] != 1:
                    raise AssertionError("finite code payload must be exactly one byte")
                np.testing.assert_array_equal(unpack_codes(packed_q, COORDINATES, book.digits, prime), query_codes[prime])
                np.testing.assert_array_equal(unpack_codes(packed_k, COORDINATES, book.digits, prime), key_codes[prime])
                prefix = f"p{prime}"
                code_arrays.update({f"{prefix}_query_codes": query_codes[prime], f"{prefix}_key_codes": key_codes[prime],
                                    f"{prefix}_packed_query_codes": packed_q, f"{prefix}_packed_key_codes": packed_k})
                state_arrays = {"center": book.center, "projection": book.projection,
                                "thresholds": book.thresholds, "digits": np.asarray(book.digits, dtype=np.uint16),
                                "prime_seed": np.asarray([prime, seed], dtype=np.uint64)}
                states.update({f"{prefix}_{key}": value for key, value in state_arrays.items()})
                alphabet = prime ** sum(book.digits)
                memory[prefix] = {"joint_alphabet": alphabet, "digits": list(book.digits),
                                  "code_bytes_per_key_per_head": 1,
                                  "code_entropy_cap_bits": float(np.log2(alphabet)),
                                  "numeric_state_array_bytes": sum(int(value.nbytes) for value in state_arrays.values()),
                                  "state_array_bytes": {key: int(value.nbytes) for key, value in state_arrays.items()},
                                  "full_trace_key_payload_bytes": int(packed_k.nbytes),
                                  "evaluation_valid_key_payload_bytes": int(evaluation_mask.sum()) * heads,
                                  "full_trace_query_payload_bytes": int(packed_q.nbytes)}
                memory[prefix]["numeric_state_plus_full_trace_key_payload_bytes"] = (
                    memory[prefix]["numeric_state_array_bytes"] + int(packed_k.nbytes))
                for role, packed in (("query", packed_q), ("key", packed_k)):
                    coarse = (query_codes[prime] if role == "query" else key_codes[prime])
                    coarse_scalar = ((coarse[..., 0] % prime).astype(np.int64) +
                                     prime * (coarse[..., 1] % prime).astype(np.int64))
                    for split, mask in (("calibration", np.pad(calibration_mask, ((0, 30), (0, 0)))),
                                        ("evaluation", evaluation_mask)):
                        label = f"{prefix}_{role}_{split}"
                        rows, counts = occupancy(packed[..., 0], mask, alphabet)
                        coarse_rows, coarse_counts = occupancy(coarse_scalar, mask, prime ** COORDINATES)
                        code_occupancy[label] = {"joint": rows, "depth_one": coarse_rows}
                        occupancy_arrays[label] = counts
                        occupancy_arrays[f"{label}_depth_one"] = coarse_counts
            if flat_codes.dtype != np.uint8 or flat_codes.shape != keys.shape[:3]:
                raise AssertionError("flat code payload must be one uint8 per key/head")
            for split, mask in (("calibration", np.pad(calibration_mask, ((0, 30), (0, 0)))),
                                ("evaluation", evaluation_mask)):
                rows, counts = occupancy(flat_codes, mask, 256)
                code_occupancy[f"flat_key_{split}"] = rows
                occupancy_arrays[f"flat_key_{split}"] = counts
            if flat.state_bytes != sum(int(value.nbytes) for value in flat.state_arrays.values()):
                raise AssertionError("flat state-byte accounting disagrees with serialized arrays")
            memory["flat"] = {"code_bytes_per_key_per_head": 1, "numeric_state_array_bytes": int(flat.state_bytes),
                              "state_array_bytes": {key: int(value.nbytes) for key, value in flat.state_arrays.items()},
                              "full_trace_key_payload_bytes": int(flat_codes.nbytes),
                              "evaluation_valid_key_payload_bytes": int(evaluation_mask.sum()) * heads,
                              "numeric_state_plus_full_trace_key_payload_bytes": int(flat.state_bytes + flat_codes.nbytes),
                              "centroid_counts": flat.centroid_counts.tolist(),
                              "unique_calibration_keys": flat.unique_key_counts.tolist(),
                              "calibration_counts": flat.calibration_counts.tolist(), "iterations": flat.iterations.tolist()}
            memory["flat"]["dot_offset_cached_bytes"] = 0
            memory["flat"]["score_variants_share_all_state_and_key_payload"] = True
            np.savez_compressed(directory / "encoder_state.npz", **states)
            np.savez_compressed(directory / "codes.npz", **code_arrays, valid_lengths=lengths)
            np.savez_compressed(directory / "occupancy.npz", **occupancy_arrays)
            query_count = len(identifiers)
            selected_indices = np.full((len(METHODS), query_count, BUDGET), -1, dtype=np.int16)
            masses = np.zeros((len(METHODS), query_count), dtype=np.float64)
            errors = np.zeros_like(masses)
            reference_squared = np.zeros(query_count, dtype=np.float64)
            agreement = np.zeros_like(masses, dtype=bool)
            overlap = np.zeros_like(masses)
            match_counts = np.zeros((len(PRIMES), query_count), dtype=np.uint16)
            optional_match_counts = np.zeros_like(match_counts)
            selected_positions = np.zeros((len(METHODS), heads, 2, length), dtype=np.int64)
            selected_lags = np.zeros_like(selected_positions)
            row_index = 0
            for chunk in EVALUATION:
                check_time()
                for head in range(heads):
                    trees = {prime: PrefixTree(COORDINATES, books[prime].digits, prime=prime) for prime in PRIMES}
                    for position in range(int(lengths[chunk])):
                        count = min(position + 1, BUDGET)
                        recency = np.arange(position + 1 - count, position + 1, dtype=np.int64)
                        selections = [recency, *(flat.select_causal(projected_queries[chunk, head, position],
                            flat_codes[chunk, head], position, budget=BUDGET, recent_window=RECENT,
                            head=head, score_kind=score_kind) for score_kind in ("euclidean", "dot_product"))]
                        for prime_index, prime in enumerate(PRIMES):
                            trees[prime].append(key_codes[prime][chunk, head, position])
                            selected = select_causal_padic(query_codes[prime][chunk, head, position],
                                key_codes[prime][chunk, head], position, BUDGET, digits=books[prime].digits,
                                recent_window=RECENT, prime=prime)
                            trie_selected = trees[prime].select(query_codes[prime][chunk, head, position], BUDGET, RECENT)
                            np.testing.assert_array_equal(selected, trie_selected)
                            total_tree_checks += 1
                            selections.extend((selected, trie_selected))
                            depth = common_padic_depth(query_codes[prime][chunk, head, position],
                                key_codes[prime][chunk, head, :position + 1], books[prime].digits, prime)
                            match_counts[prime_index, row_index] = np.count_nonzero(depth >= 1)
                            optional_match_counts[prime_index, row_index] = np.count_nonzero(
                                depth[:position + 1 - min(RECENT, count)] >= 1)
                        q = queries[chunk, head, position]
                        causal_k = keys[chunk, head, :position + 1]
                        causal_v = values[chunk, head, :position + 1].astype(np.float64)
                        probabilities = attention_probabilities(q, causal_k)
                        reference = probabilities @ causal_v
                        reference_squared[row_index] = float(reference @ reference)
                        for method_index, selected in enumerate(selections):
                            validate_selection(selected, position)
                            selected_indices[method_index, row_index, :count] = selected
                            agreement[method_index, row_index] = np.array_equal(selected, recency)
                            overlap[method_index, row_index] = np.intersect1d(selected, recency, assume_unique=True).size / count
                            if METHODS[method_index].startswith("trie_"):
                                masses[method_index, row_index] = masses[method_index - 1, row_index]
                                errors[method_index, row_index] = errors[method_index - 1, row_index]
                            else:
                                mass = float(probabilities[selected].sum())
                                if not 0 <= mass <= 1 + 1e-12:
                                    raise AssertionError("kept mass is outside its probability range")
                                masses[method_index, row_index] = mass
                                restricted = attention_probabilities(q, causal_k[selected])
                                difference = restricted @ causal_v[selected] - reference
                                errors[method_index, row_index] = float(difference @ difference)
                            for stratum in (0, 1) if position >= BUDGET else (0,):
                                selected_positions[method_index, head, stratum] += np.bincount(selected, minlength=length)
                                selected_lags[method_index, head, stratum] += np.bincount(position - selected, minlength=length)
                        row_index += 1
            if row_index != query_count:
                raise AssertionError("raw query index and evaluation split disagree")
            np.savez_compressed(directory / "rawstats.npz", methods=np.asarray(METHODS), identifiers=identifiers,
                                identifier_columns=np.asarray(["chunk", "head", "position"]),
                                selected_indices=selected_indices, selected_counts=np.minimum(identifiers[:, 2] + 1, BUDGET),
                                kept_attention_mass=masses, output_error_squared=errors,
                                output_reference_squared=reference_squared, recency_exact_agreement=agreement,
                                recency_selected_overlap=overlap, primes=np.asarray(PRIMES),
                                causal_depth_one_match_counts=match_counts,
                                optional_older_depth_one_match_counts=optional_match_counts,
                                selected_position_counts=selected_positions, selected_lag_counts=selected_lags,
                                distribution_strata=np.asarray(["all_queries", "affected_queries"]))
            variant = {"name": name, "projection_kind": kind, "seed": seed,
                       "embedding_sha256": array_hash(reference_book.center, reference_book.projection),
                       "calibration_geometry": calibration_geometry(queries[:2], keys[:2], calibration_mask,
                                                                     reference_book.projection),
                       "code_occupancy": code_occupancy, "memory_accounting": memory,
                       "summaries": summarize_variant(identifiers, masses, errors, reference_squared,
                                                       agreement, overlap, match_counts, optional_match_counts, heads),
                       "checks": {"same_real_embeddings_exact": True, "packed_roundtrips": 4,
                                  "padic_trie_exact_selection_checks": 2 * query_count,
                                  "unique_causal_equal_budget_and_window": True}}
            write_json(directory / "results.json", variant)
            variant_results.append(variant)
            check_time()
        if file_hash(input_path) != trace_hash:
            raise RuntimeError("input trace changed during execution")
        elapsed = time.perf_counter() - start
        result = {"status": "alignment_diagnostic_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "config": config, "variants": variant_results,
                  "checks": {"padic_trie_exact_selection_checks": total_tree_checks,
                             "packed_code_roundtrips": 16, "same_embedding_checks": 4},
                  "functional_elapsed_seconds": elapsed,
                  "process_cpu_seconds": time.process_time() - cpu_start,
                  "scope": "Fixed reused-validation component diagnostic; no model forward/training, token NLL, confirmation or native speed claim.",
                  "memory_scope": {"real_keys_trace_array_bytes": int(keys.nbytes), "real_values_trace_array_bytes": int(values.nbytes),
                                   "all_real_kv_retained": True, "kv_compression": False,
                                   "python_tree_index_memory_measured": False, "physical_gqa_io_measured": False,
                                   "state_bytes_definition": "actual NumPy numeric buffers, padding and fit metadata included; Python/file-container overhead excluded",
                                   "trace_storage_note": "repeated per-query-head float32 capture is not the physical model KV-cache footprint"},
                  "timing_scope": "functional NumPy gate including calibration, controls and intermediate artifact writes; no efficiency claim",
                  "interpretation": "Finite balls exactly equal matched tries. Shared PCA separation is a tested hypothesis, not an assumed diagnosis."}
        write_json(output / "results.json", result)
        check_time()
        artifacts = sorted(path for path in output.rglob("*") if path.is_file())
        completed = {"status": result["status"], "created_utc": datetime.now(timezone.utc).isoformat(),
                     "trace_sha256": trace_hash, "code_sha256": config["code_sha256"],
                     "artifact_sha256": {str(path.relative_to(output)): file_hash(path) for path in artifacts},
                     "artifact_bytes": {str(path.relative_to(output)): path.stat().st_size for path in artifacts},
                     "functional_elapsed_seconds": time.perf_counter() - start,
                     "process_cpu_seconds": time.process_time() - cpu_start,
                     "completion_digest_location": "completed.sha256; digest excludes this manifest and its sidecar from artifact_sha256"}
        check_time()
        write_json(output / "completed.json", completed)
        result["completion_sha256"] = file_hash(output / "completed.json")
        (output / "completed.sha256").write_text(result["completion_sha256"] + "\n")
        return result
    except Exception as error:
        (output / "completed.json").unlink(missing_ok=True)
        (output / "completed.sha256").unlink(missing_ok=True)
        write_json(output / "failed.json", {"status": "alignment_diagnostic_failed", "error_type": type(error).__name__,
                                           "message": str(error), "trace_sha256": trace_hash,
                                           "functional_elapsed_seconds": time.perf_counter() - start,
                                           "process_cpu_seconds": time.process_time() - cpu_start})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0,
                        help="CPU and wall-time guards, each at most 600 seconds")
    args = parser.parse_args()
    result = run(args.input, args.output, max_seconds=args.max_seconds)
    print(json.dumps({"output": str(args.output.resolve()), "status": result["status"],
                      "completion_sha256": result["completion_sha256"], "checks": result["checks"],
                      "variants": [{"name": variant["name"], "summaries": [
                          {"method": row["method"], **row["all_heads"]} for row in variant["summaries"]
                          if row["stratum"] == "affected_queries"]} for variant in result["variants"]]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
