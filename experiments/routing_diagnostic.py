"""Component-only causal routing diagnostic on captured pretrained Q/K(/V).

No model is modified; retained attention mass and optional value-output errors
are diagnostics, not token NLL, accuracy or native speed measurements.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from padic_lm.routing import (
    PrefixTree, QuantileCodebook, angular_codes, angular_hyperplanes,
    attention_probabilities, digit_layout, pack_codes, select_causal_angular,
    select_causal_padic, select_causal_random, select_from_scores, unpack_codes,
)

METHODS = ("padic_balls", "prefix_tree", "recency", "seeded_random", "angular_lsh", "oracle_attention")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def load_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as trace:
        if "queries" not in trace or "keys" not in trace:
            raise ValueError("trace must contain queries and keys")
        queries, keys = np.array(trace["queries"]), np.array(trace["keys"])
        values = np.array(trace["values"]) if "values" in trace else None
        if queries.ndim != 4 or queries.shape != keys.shape or not all(queries.shape):
            raise ValueError("Q/K must match [chunks,heads,seq,head_dim]; repeat GQA keys before capture")
        if queries.dtype.kind != "f" or keys.dtype.kind != "f":
            raise TypeError("Q/K must contain real floating-point vectors")
        if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(keys)):
            raise ValueError("Q/K must be finite")
        if values is not None and (values.ndim != 4 or values.shape[:3] != queries.shape[:3]
                                   or not values.shape[-1] or values.dtype.kind != "f"
                                   or not np.all(np.isfinite(values))):
            raise ValueError("values must be finite floating [chunks,heads,seq,value_dim]")
        chunks, _, length, _ = queries.shape
        lengths = np.full(chunks, length, dtype=np.int64)
        if "valid_lengths" in trace:
            raw_lengths = trace["valid_lengths"]
            if raw_lengths.shape != (chunks,) or raw_lengths.dtype.kind not in "iu":
                raise ValueError("valid_lengths must be an integer [chunks] vector")
            lengths = raw_lengths.astype(np.int64)
        if "attention_mask" in trace:
            mask = np.asarray(trace["attention_mask"])
            if mask.shape != (chunks, length) or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("attention_mask must be binary [chunks,seq]")
            mask_lengths = mask.sum(axis=1).astype(np.int64)
            expected = np.arange(length)[None, :] < mask_lengths[:, None]
            if not np.array_equal(mask.astype(bool), expected):
                raise ValueError("diagnostic supports only contiguous valid prefixes, not holes or left padding")
            if "valid_lengths" in trace and not np.array_equal(lengths, mask_lengths):
                raise ValueError("valid_lengths and attention_mask disagree")
            lengths = mask_lengths
        if np.any(lengths < 1) or np.any(lengths > length):
            raise ValueError("each chunk must have between one and seq valid tokens")
        metadata: dict = {}
        if "metadata" in trace:
            raw_metadata = trace["metadata"]
            if raw_metadata.size == 1 and raw_metadata.dtype.kind in "US":
                raw_text = raw_metadata.reshape(()).item()
                if isinstance(raw_text, bytes):
                    raw_text = raw_text.decode("utf-8")
                try:
                    parsed = json.loads(raw_text)
                    metadata = parsed if isinstance(parsed, dict) else {"source_metadata": parsed}
                except json.JSONDecodeError:
                    metadata = {"source_metadata_text": raw_text}
        return queries, keys, values, lengths, metadata


def summarize(records: list[dict], budgets: list[int], *, has_values: bool) -> list[dict]:
    summaries = []
    for budget in budgets:
        for method in METHODS:
            for stratum in ("all_positions", "full_budget_positions"):
                chosen = [record for record in records if record["budget"] == budget
                          and record["method"] == method
                          and (stratum == "all_positions" or record["position"] + 1 >= budget)]
                if not chosen:
                    continue
                mass = np.array([record["kept_attention_mass"] for record in chosen])
                # Chunks, not independent tokens, are the natural replication
                # unit; these statistics intentionally are descriptive only.
                chunk_means = {str(chunk): float(np.mean([record["kept_attention_mass"]
                               for record in chosen if record["chunk"] == chunk]))
                               for chunk in sorted({record["chunk"] for record in chosen})}
                row = {
                    "budget": budget, "method": method, "stratum": stratum,
                    "queries": len(chosen), "chunks": len(chunk_means),
                    "kept_mass_mean": float(mass.mean()), "kept_mass_median": float(np.median(mass)),
                    "kept_mass_p10": float(np.quantile(mass, .1)),
                    "kept_mass_p90": float(np.quantile(mass, .9)),
                    "per_chunk_kept_mass_means": chunk_means,
                    "selected_keys_mean": float(np.mean([record["selected_count"] for record in chosen])),
                    "full_context_fraction": float(np.mean([record["selected_count"] == record["position"] + 1
                                                             for record in chosen])),
                }
                if has_values:
                    error_sq = sum(record["output_error_squared"] for record in chosen)
                    reference_sq = sum(record["output_reference_squared"] for record in chosen)
                    row["value_output_nrmse"] = float(np.sqrt(error_sq / reference_sq)) if reference_sq else (0.0 if not error_sq else None)
                    row["value_output_mean_l2_error"] = float(np.mean([record["output_l2_error"] for record in chosen]))
                summaries.append(row)
    return summaries


def run(
    input_path: Path, output: Path, *, calibration_chunks: int = 2,
    budgets: list[int] | None = None, recent_window: int = 8, digits: int = 4,
    projections: int = 2, seed: int = 17, query_stride: int = 1,
    projection_kind: str = "pca", prime: int = 2, code_bit_budget: int | None = None,
) -> dict:
    start = time.perf_counter()
    source_bytes = {path.name: path.read_bytes() for path in
                    (Path(__file__).resolve(), PROJECT / "src" / "padic_lm" / "routing.py")}
    input_path, output = input_path.resolve(), output.resolve()
    queries, keys, values, lengths, source_metadata = load_trace(input_path)
    chunks, heads, length, width = queries.shape
    budgets = sorted(set([64] if budgets is None else budgets))
    if not budgets or any(isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1 for budget in budgets):
        raise ValueError("budgets must be positive integers")
    budgets = [int(budget) for budget in budgets]
    if not 1 <= calibration_chunks < chunks:
        raise ValueError("reserve at least one calibration and one evaluation chunk")
    if query_stride < 1 or recent_window < 0:
        raise ValueError("query_stride must be positive and recent_window nonnegative")
    calibration_mask = np.arange(length)[None, :] < lengths[:calibration_chunks, None]
    book = QuantileCodebook.fit(queries[:calibration_chunks], keys[:calibration_chunks],
                               coordinates=projections, digits=digits, seed=seed,
                               mask=calibration_mask, projection_kind=projection_kind,
                               prime=prime, code_bit_budget=code_bit_budget)
    layout = digit_layout(book.digits, projections, prime)
    joint_alphabet = prime ** sum(layout)
    query_codes, key_codes = book.encode(queries), book.encode(keys)
    packed_q, packed_k = pack_codes(query_codes, layout, prime), pack_codes(key_codes, layout, prime)
    np.testing.assert_array_equal(unpack_codes(packed_q, projections, layout, prime), query_codes)
    np.testing.assert_array_equal(unpack_codes(packed_k, projections, layout, prime), key_codes)
    lsh_bits = packed_k.shape[-1] * 8
    planes = angular_hyperplanes(heads, width, lsh_bits, seed)
    lsh_queries, lsh_keys = angular_codes(queries, planes), angular_codes(keys, planes)
    records: list[dict] = []
    raw_identifiers, raw_selected, raw_mass, raw_error_sq, raw_ref_sq = [], [], [], [], []
    tree_checks = 0
    for chunk in range(calibration_chunks, chunks):
        valid_length = int(lengths[chunk])
        for head in range(heads):
            tree = PrefixTree(projections, layout, prime=prime)
            for position in range(valid_length):
                tree.append(key_codes[chunk, head, position])
                if position % query_stride:
                    continue
                probabilities = attention_probabilities(queries[chunk, head, position],
                                                          keys[chunk, head, :position + 1])
                causal_values = values[chunk, head, :position + 1].astype(np.float64) if values is not None else None
                full_output = probabilities @ causal_values if causal_values is not None else None
                reference_sq = float(np.dot(full_output, full_output)) if full_output is not None else np.nan
                for budget in budgets:
                    count = min(budget, position + 1)
                    chosen = {
                        "padic_balls": select_causal_padic(query_codes[chunk, head, position], key_codes[chunk, head],
                                                            position, budget, digits=layout, recent_window=recent_window, prime=prime),
                        "prefix_tree": tree.select(query_codes[chunk, head, position], budget, recent_window),
                        "recency": np.arange(position + 1 - count, position + 1, dtype=np.int64),
                        "seeded_random": select_causal_random(position, budget, seed=seed, chunk=chunk,
                                                               head=head, recent_window=recent_window),
                        "angular_lsh": select_causal_angular(lsh_queries[chunk, head, position], lsh_keys[chunk, head],
                                                              position, budget, recent_window),
                        # Oracle knows the actual full-attention scores. It is
                        # an upper bound at this budget/window, not deployable
                        # without the work routing seeks to avoid.
                        "oracle_attention": select_from_scores(probabilities, budget, recent_window),
                    }
                    if not np.array_equal(chosen["padic_balls"], chosen["prefix_tree"]):
                        raise AssertionError("independent tree and finite p-adic selection disagree")
                    tree_checks += 1
                    selected_row = np.full((len(METHODS), max(budgets)), -1, dtype=np.int64)
                    mass_row, error_row = [], []
                    for method_index, method in enumerate(METHODS):
                        selected = chosen[method]
                        if len(selected) != count or len(np.unique(selected)) != count or np.any(selected > position) or np.any(selected < 0):
                            raise AssertionError("routing violated unique causal equal-budget selection")
                        selected_row[method_index, :count] = selected
                        mass = float(probabilities[selected].sum())
                        if not 0.0 <= mass <= 1.0 + 1e-12:
                            raise AssertionError("retained mass is outside its probability range")
                        record = {"chunk": chunk, "head": head, "position": position,
                                  "context_keys": position + 1, "budget": budget, "method": method,
                                  "selected_count": count, "kept_attention_mass": mass}
                        error_squared = np.nan
                        if full_output is not None:
                            # Recompute selected softmax stably. Dividing full
                            # probabilities by kept mass can underflow to 0/0
                            # when the strongest key is excluded.
                            restricted_probabilities = attention_probabilities(queries[chunk, head, position], keys[chunk, head, selected])
                            selected_output = restricted_probabilities @ causal_values[selected]
                            error = selected_output - full_output
                            error_squared = float(np.dot(error, error))
                            record.update(output_error_squared=error_squared,
                                          output_reference_squared=reference_sq,
                                          output_l2_error=float(np.sqrt(error_squared)))
                        records.append(record)
                        mass_row.append(mass)
                        error_row.append(error_squared)
                    # Oracle is the maximizer among subsets that obey the
                    # same mandatory recent window.
                    if any(mass > mass_row[-1] + 1e-12 for mass in mass_row):
                        raise AssertionError("oracle retained mass failed to upper-bound a candidate")
                    raw_identifiers.append((chunk, head, position, budget))
                    raw_selected.append(selected_row)
                    raw_mass.append(mass_row)
                    raw_error_sq.append(error_row)
                    raw_ref_sq.append(reference_sq)
    if not records:
        raise ValueError("no evaluation queries remain")
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(input_path), "trace_sha256": file_hash(input_path),
        "code_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()},
        "code_snapshots": {name: f"sources/{name}" for name in source_bytes},
        "source_metadata": source_metadata, "qk_shape": list(queries.shape),
        "q_dtype": str(queries.dtype), "k_dtype": str(keys.dtype),
        "has_values": values is not None, "value_shape": list(values.shape) if values is not None else None,
        "valid_lengths": lengths.tolist(), "calibration_chunks": list(range(calibration_chunks)),
        "evaluation_chunks": list(range(calibration_chunks, chunks)),
        "calibration_token_positions": book.calibration_tokens,
        "calibration_qk_sha256": array_hash(queries[:calibration_chunks], keys[:calibration_chunks], calibration_mask),
        "budgets": budgets, "recent_window": recent_window, "digits": list(layout),
        "prime": prime, "coordinate_moduli": [prime ** count for count in layout],
        "joint_alphabet": joint_alphabet, "requested_code_bit_budget": code_bit_budget,
        "coordinates": projections,
        "product_metric": "max_coordinate_finite_padic_distance; equal_finite_coordinate_has_zero_distance",
        "tree_levels": "one base-prime digit from every still-active coordinate; exhausted coordinates use fixed sentinel",
        "encoding": "shared_centered_projection_quantiles_base_prime_digits_reversed",
        "projection_kind": projection_kind, "quantile_method": "linear", "bin_tie_rule": "right",
        "seed": seed, "query_stride": query_stride,
        "selection_tie_rule": "latest_key_first", "includes_current_key": True,
        "methods": list(METHODS), "lsh_hyperplanes_per_head": lsh_bits,
        "lsh_comparison": "same_serialized_code_bytes; more_real_projections_than_quantile_encoder",
        "attention_definition": "softmax(Q K^T / sqrt(head_dim)) on causal prefix, no additional bias",
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    memory = {
        "code_alphabet_entropy_bits_per_key_per_head": math.log2(joint_alphabet),
        "minimum_fixed_code_bits_per_key_per_head": (joint_alphabet - 1).bit_length(),
        "allocated_code_bits_per_key_per_head": packed_k.shape[-1] * 8,
        "serialized_code_bytes_per_key_per_head": packed_k.shape[-1],
        "numpy_unpacked_code_bytes_per_key_per_head": key_codes.dtype.itemsize * projections,
        "all_key_code_array_packed_bytes": int(packed_k.nbytes),
        "all_key_code_array_unpacked_bytes": int(key_codes.nbytes),
        "real_encoder_center_bytes": int(book.center.nbytes),
        "real_encoder_projection_bytes": int(book.projection.nbytes),
        "real_encoder_threshold_bytes": int(book.thresholds.nbytes),
        "lsh_real_hyperplane_bytes": int(planes.nbytes),
        "input_keys_bytes": int(keys.nbytes),
        "input_values_bytes": int(values.nbytes) if values is not None else None,
        "index_memory_measured": False,
        "kv_compression": False,
        "note": "All full real K/V remain present. Tree/Python index overhead and model weights are not measured here.",
    }
    source_directory = output / "sources"
    source_directory.mkdir(exist_ok=True)
    for name, content in source_bytes.items():
        (source_directory / name).write_bytes(content)
    result = {
        "status": "component_diagnostic_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config, "summaries": summarize(records, budgets, has_values=values is not None),
        "equivalence_checks": {"padic_prefix_exact_selection_checks": tree_checks,
                               "packed_code_roundtrips": 2,
                               "equal_budget_causal_selection": True},
        "memory_accounting": memory,
        "functional_elapsed_seconds": elapsed,
        "scope": "Development component diagnostic; no model conversion, token NLL, inferential claim or native speed claim.",
        "timing_scope": "Dense NumPy reference including calibration and all controls, excludes trace capture and output writes.",
        "interpretation": "Finite p-adic balls are exactly prefix-tree equivalent. Oracle uses unavailable full attention scores.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with (output / "per_query.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / "rawstats.npz",
                        methods=np.array(METHODS), identifiers=np.asarray(raw_identifiers, dtype=np.int64),
                        identifier_columns=np.array(["chunk", "head", "position", "budget"]),
                        selected_indices=np.stack(raw_selected), kept_attention_mass=np.asarray(raw_mass),
                        output_error_squared=np.asarray(raw_error_sq), output_reference_squared=np.asarray(raw_ref_sq),
                        valid_lengths=lengths, query_codes=query_codes, key_codes=key_codes,
                        packed_query_codes=packed_q, packed_key_codes=packed_k,
                        encoder_center=book.center, encoder_projection=book.projection,
                        encoder_thresholds=book.thresholds, lsh_hyperplanes=planes,
                        packed_lsh_query_codes=np.packbits(lsh_queries, axis=-1, bitorder="little"),
                        packed_lsh_key_codes=np.packbits(lsh_keys, axis=-1, bitorder="little"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-chunks", type=int, default=2)
    parser.add_argument("--budget", type=int, nargs="+", default=[64], help="one or several fixed candidate budgets")
    parser.add_argument("--recent-window", type=int, default=8)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--prime", type=int, choices=(2, 3, 5), default=2)
    parser.add_argument("--code-bit-budget", type=int, default=None,
                        help="joint entropy cap; distributes total base-prime digits across projections, overriding --digits")
    parser.add_argument("--projections", type=int, default=2)
    parser.add_argument("--projection-kind", choices=("pca", "random"), default="pca")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--query-stride", type=int, default=1)
    args = parser.parse_args()
    result = run(args.input, args.output, calibration_chunks=args.calibration_chunks,
                 budgets=args.budget, recent_window=args.recent_window, digits=args.digits,
                 projections=args.projections, seed=args.seed, query_stride=args.query_stride,
                 projection_kind=args.projection_kind, prime=args.prime, code_bit_budget=args.code_bit_budget)
    print(json.dumps({"output": str(args.output.resolve()), "checks": result["equivalence_checks"],
                      "scope": result["scope"], "prime": result["config"]["prime"],
                      "digits": result["config"]["digits"], "joint_alphabet": result["config"]["joint_alphabet"],
                      "summaries": [{key: value for key, value in row.items()
                                     if key in {"budget", "method", "kept_mass_mean", "value_output_nrmse", "queries"}}
                                    for row in result["summaries"] if row["stratum"] == "full_budget_positions"]},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
