"""Secondary logical GQA selection unions for a completed alignment diagnostic.

All real K/V remain retained. These counts describe selected position unions,
not actual device reads, physical storage, cache savings or native speed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from numbers import Integral
from pathlib import Path
import platform
import sys

import numpy as np


FAMILIES = ("pca-17", "random-17", "random-29", "random-43")
METHODS = ("recency", "flat_euclidean", "flat_dot_product", "padic_2", "trie_2", "padic_3", "trie_3")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _integer(value, name, *, minimum=1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _hex_digest(value, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(symbol in "0123456789abcdef" for symbol in value)


def verify_completion(directory: Path) -> tuple[dict, str]:
    """Verify every completed alignment artifact before trusting its arrays."""
    directory = directory.resolve()
    digest = file_hash(directory / "completed.json")
    if (directory / "completed.sha256").read_text().strip() != digest:
        raise ValueError("alignment completion sidecar does not match")
    completed = json.loads((directory / "completed.json").read_text())
    if completed.get("status") != "alignment_diagnostic_completed":
        raise ValueError("alignment input must be successfully completed")
    hashes, sizes = completed.get("artifact_sha256"), completed.get("artifact_bytes")
    if not isinstance(hashes, dict) or not hashes or not isinstance(sizes, dict) or set(hashes) != set(sizes):
        raise ValueError("completion must declare matching artifact hashes and sizes")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
              and path.relative_to(directory).as_posix() not in {"completed.json", "completed.sha256"}}
    if actual != set(hashes):
        raise ValueError("alignment files differ from the completed artifact set")
    for name, expected in hashes.items():
        relative = Path(name)
        path = directory / relative
        if relative.is_absolute() or ".." in relative.parts or not path.resolve().is_relative_to(directory):
            raise ValueError("artifact path must remain inside the alignment directory")
        if path.stat().st_size != sizes[name] or file_hash(path) != expected:
            raise ValueError(f"alignment artifact hash/size mismatch: {name}")
    required = {"config.json", "results.json", *(f"{family}/rawstats.npz" for family in FAMILIES)}
    if not required.issubset(hashes):
        raise ValueError("completed alignment omits required config/results/raw arrays")
    return completed, digest


def validate_gqa_capture(keys: np.ndarray, values: np.ndarray, model_manifest: dict,
                         *, head_to_kv: np.ndarray | None = None) -> np.ndarray:
    """Validate the pinned stock repeat_kv layout and every repeated K/V byte.

    Llama repeat_kv expands physical heads as [kv0 repeated G times, kv1
    repeated G times, ...]. The model manifest provides Q/KV head counts and
    G; exact captured arrays corroborate that documented layout. We neither
    infer physical heads from arbitrary query similarities nor accept a
    regrouping of heads chosen after observing selection results.
    """
    query_heads = _integer(model_manifest.get("query_heads"), "query_heads")
    physical_heads = _integer(model_manifest.get("physical_kv_heads"), "physical_kv_heads")
    group_size = _integer(model_manifest.get("gqa_groups"), "gqa_groups")
    if query_heads != physical_heads * group_size:
        raise ValueError("manifest Q/KV head counts and repeat factor disagree")
    keys, values = np.asarray(keys), np.asarray(values)
    if (keys.ndim != 4 or values.ndim != 4 or not all(keys.shape) or not all(values.shape)
            or keys.shape[:3] != values.shape[:3] or keys.shape[1] != query_heads
            or keys.dtype.kind != "f" or values.dtype.kind != "f"
            or not np.all(np.isfinite(keys)) or not np.all(np.isfinite(values))):
        raise ValueError("capture must contain matching finite repeated [chunks,Qheads,seq,dim] K/V")
    expected = np.repeat(np.arange(physical_heads, dtype=np.int32), group_size)
    for supplied_mapping in (head_to_kv, model_manifest.get("query_head_to_physical_kv_head")):
        if supplied_mapping is None:
            continue
        supplied = np.asarray(supplied_mapping)
        if supplied.dtype.kind not in "iu" or supplied.shape != expected.shape or not np.array_equal(supplied, expected):
            raise ValueError("supplied head grouping does not match pinned stock repeat_kv layout")
    for physical_head in range(physical_heads):
        heads = np.flatnonzero(expected == physical_head)
        for capture in (keys, values):
            reference = capture[:, heads[0]].tobytes(order="C")
            if any(capture[:, head].tobytes(order="C") != reference for head in heads[1:]):
                raise ValueError("captured K/V are not bitwise identical within a pinned repeat_kv group")
    return expected


def collect_unions(identifiers: np.ndarray, selected_indices: np.ndarray,
                   selected_counts: np.ndarray, *, lengths: np.ndarray,
                   evaluation_chunks: list[int], head_to_kv: np.ndarray,
                   budget: int, recent_window: int) -> dict[str, np.ndarray]:
    """Join head rows by their IDs, validate selections, then count group unions."""
    budget = _integer(budget, "budget")
    recent_window = _integer(recent_window, "recent_window", minimum=0)
    identifiers, selected_indices, selected_counts = map(np.asarray, (identifiers, selected_indices, selected_counts))
    lengths, head_to_kv = np.asarray(lengths), np.asarray(head_to_kv)
    if (identifiers.ndim != 2 or identifiers.shape[1] != 3 or identifiers.dtype.kind not in "iu"
            or selected_indices.ndim != 3 or selected_indices.shape[1:] != (len(identifiers), budget)
            or selected_indices.dtype.kind not in "iu" or selected_counts.shape != (len(identifiers),)
            or selected_counts.dtype.kind not in "iu"):
        raise ValueError("raw IDs, selected indices and selected counts have invalid shapes/dtypes")
    if (lengths.ndim != 1 or lengths.dtype.kind not in "iu" or np.any(lengths < 1)
            or head_to_kv.ndim != 1 or head_to_kv.dtype.kind not in "iu" or not len(head_to_kv)):
        raise ValueError("valid lengths and head mapping must be nonempty integer vectors")
    if np.any(head_to_kv < 0):
        raise ValueError("head mapping cannot contain negative physical head IDs")
    physical_heads = int(head_to_kv.max()) + 1
    if len(head_to_kv) % physical_heads or not np.array_equal(
            head_to_kv, np.repeat(np.arange(physical_heads), len(head_to_kv) // physical_heads)):
        raise ValueError("head mapping must have the validated contiguous repeat_kv layout")
    evaluation_chunks = [_integer(chunk, "evaluation chunk", minimum=0) for chunk in evaluation_chunks]
    if not evaluation_chunks or len(set(evaluation_chunks)) != len(evaluation_chunks) or max(evaluation_chunks) >= len(lengths):
        raise ValueError("evaluation chunk list must be unique and address valid lengths")
    expected_ids = {(chunk, head, position) for chunk in evaluation_chunks for head in range(len(head_to_kv))
                    for position in range(int(lengths[chunk]))}
    row_index = {tuple(map(int, identity)): index for index, identity in enumerate(identifiers)}
    if len(row_index) != len(identifiers) or set(row_index) != expected_ids:
        raise ValueError("raw IDs must cover every declared evaluation query/head exactly once")
    method_count = selected_indices.shape[0]
    if method_count < 1:
        raise ValueError("at least one method is required")
    actual_counts = np.empty((method_count, len(identifiers)), dtype=np.int32)
    for row, (_, _, position) in enumerate(identifiers):
        position = int(position)
        count = min(position + 1, budget)
        recent = min(recent_window, count)
        if int(selected_counts[row]) != count:
            raise ValueError("declared per-head selection count disagrees with exact causal budget")
        for method in range(method_count):
            padded = selected_indices[method, row]
            if np.any(padded < -1):
                raise ValueError("selection padding must use only -1")
            selected = padded[padded != -1]
            if (len(selected) != count or np.any(selected > position)
                    or np.any(np.diff(selected.astype(np.int64)) <= 0)):
                raise ValueError("each selection must be chronological, unique, causal and exact-budget")
            if not np.array_equal(selected[-recent:] if recent else [],
                                  np.arange(position + 1 - recent, position + 1)):
                raise ValueError("selection must include the identical mandatory recent window")
            actual_counts[method, row] = len(selected)
    union_ids = np.asarray(sorted((chunk, kv_head, position) for chunk in evaluation_chunks
                                 for kv_head in range(physical_heads)
                                 for position in range(int(lengths[chunk]))), dtype=np.int32)
    unions = np.empty((method_count, len(union_ids)), dtype=np.int32)
    bounds = np.empty(len(union_ids), dtype=np.int32)
    group_size = len(head_to_kv) // physical_heads
    for row, (chunk, physical_head, position) in enumerate(union_ids):
        chunk, physical_head, position = map(int, (chunk, physical_head, position))
        count = min(position + 1, budget)
        recent = min(recent_window, count)
        bounds[row] = min(position + 1, recent + group_size * (count - recent))
        query_rows = [row_index[(chunk, int(head), position)] for head in np.flatnonzero(head_to_kv == physical_head)]
        for method in range(method_count):
            selected = selected_indices[method, query_rows].reshape(-1)
            unions[method, row] = len(np.unique(selected[selected != -1]))
            if not count <= unions[method, row] <= bounds[row]:
                raise AssertionError("logical union exceeds its validated shared-recent bound")
    return {"union_identifiers": union_ids, "union_counts": unions, "shared_recent_union_bounds": bounds,
            "query_identifiers": identifiers, "per_query_head_selected_counts": actual_counts,
            "query_head_to_physical_kv_head": head_to_kv.astype(np.int32)}


def summarize(methods: tuple[str, ...], raw: dict[str, np.ndarray], affected_start: int) -> list[dict]:
    rows = []
    for stratum in ("all_queries", "affected_queries"):
        union_mask = np.ones(len(raw["union_identifiers"]), dtype=bool)
        query_mask = np.ones(len(raw["query_identifiers"]), dtype=bool)
        if stratum == "affected_queries":
            union_mask = raw["union_identifiers"][:, 2] >= affected_start
            query_mask = raw["query_identifiers"][:, 2] >= affected_start
        for index, method in enumerate(methods):
            counts = raw["union_counts"][index, union_mask]
            head_counts = raw["per_query_head_selected_counts"][index, query_mask]
            rows.append({"method": method, "stratum": stratum,
                         "logical_group_query_count": int(len(counts)),
                         "selected_position_union_mean": float(counts.mean()) if len(counts) else None,
                         "selected_position_union_median": float(np.median(counts)) if len(counts) else None,
                         "selected_position_union_max": int(counts.max()) if len(counts) else None,
                         "per_query_head_count": int(len(head_counts)),
                         "per_query_head_selected_positions_mean": float(head_counts.mean()) if len(head_counts) else None,
                         "shared_recent_union_bound_max": int(raw["shared_recent_union_bounds"][union_mask].max())
                             if len(counts) else None})
    return rows


def run(alignment_directory: Path, trace_path: Path, model_manifest_path: Path, output: Path) -> dict:
    alignment_directory, trace_path, model_manifest_path, output = (
        Path(path).resolve() for path in (alignment_directory, trace_path, model_manifest_path, output))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    sources = {Path(__file__).name: Path(__file__).read_bytes()}
    completed, completion_digest = verify_completion(alignment_directory)
    config = json.loads((alignment_directory / "config.json").read_text())
    parent_result = json.loads((alignment_directory / "results.json").read_text())
    model_manifest = json.loads(model_manifest_path.read_text())
    trace_digest, model_manifest_digest = file_hash(trace_path), file_hash(model_manifest_path)
    if (parent_result.get("status") != completed["status"] or parent_result.get("config") != config
            or any(value != trace_digest for value in
                   (completed.get("trace_sha256"), config.get("trace_sha256"), model_manifest.get("trace_sha256")))):
        raise ValueError("completed alignment, model manifest and original trace provenance disagree")
    if config.get("budget") != 32 or config.get("recent_window") != 8 or tuple(config.get("methods", ())) != METHODS:
        raise ValueError("secondary accounting requires the frozen budget32/recent8 alignment methods")
    if config.get("calibration_chunks") != [0, 1] or config.get("evaluation_chunks") != list(range(2, 32)):
        raise ValueError("secondary accounting requires the fixed calibration/evaluation split")
    if (not model_manifest.get("model") or not _hex_digest(model_manifest.get("model_revision"), 40)
            or not _hex_digest(model_manifest.get("model_loaded_state_sha256"), 64)
            or model_manifest.get("layer_index") != 0 or model_manifest.get("sequence_length") != 128
            or model_manifest.get("calibration_chunk_indices") != config["calibration_chunks"]
            or model_manifest.get("evaluation_chunk_indices") != config["evaluation_chunks"]):
        raise ValueError("model manifest must pin the original first-layer checkpoint/state")
    expected_source_hashes = config.get("code_sha256", {})
    if completed.get("code_sha256") != expected_source_hashes or not expected_source_hashes:
        raise ValueError("completed alignment source hashes disagree")
    for name, digest in expected_source_hashes.items():
        relative = config.get("code_snapshots", {}).get(name)
        if relative not in completed["artifact_sha256"] or completed["artifact_sha256"][relative] != digest:
            raise ValueError("alignment source snapshot/hash is not pinned by completion")
    with np.load(trace_path, allow_pickle=False) as trace:
        queries, keys, values = (np.array(trace[name]) for name in ("queries", "keys", "values"))
        if (queries.ndim != 4 or not all(queries.shape) or queries.shape != keys.shape
                or queries.dtype.kind != "f" or not np.all(np.isfinite(queries))):
            raise ValueError("original query/key capture must match and be finite")
        lengths = np.array(trace["valid_lengths"]) if "valid_lengths" in trace else np.full(queries.shape[0], queries.shape[2], dtype=np.int64)
        if "attention_mask" in trace:
            mask = np.array(trace["attention_mask"])
            if mask.shape != (queries.shape[0], queries.shape[2]) or not np.all((mask == 0) | (mask == 1)):
                raise ValueError("capture attention mask must be binary [chunks,seq]")
            mask_lengths = mask.sum(axis=1).astype(np.int64)
            if (not np.array_equal(mask.astype(bool), np.arange(queries.shape[2])[None] < mask_lengths[:, None])
                    or ("valid_lengths" in trace and not np.array_equal(lengths, mask_lengths))):
                raise ValueError("capture must have matching contiguous valid prefixes")
            lengths = mask_lengths
    if (list(queries.shape) != config.get("qk_shape") or list(values.shape) != config.get("value_shape")
            or lengths.tolist() != config.get("valid_lengths") or queries.shape[0] != 32
            or queries.shape[2] != 128 or lengths.shape != (32,) or lengths.dtype.kind not in "iu"
            or np.any(lengths < 1) or np.any(lengths > queries.shape[2])):
        raise ValueError("original capture shapes/valid lengths disagree with the fixed alignment input")
    head_to_kv = validate_gqa_capture(keys, values, model_manifest)
    variants, raw_output = [], {"query_head_to_physical_kv_head": head_to_kv}
    for family in FAMILIES:
        with np.load(alignment_directory / family / "rawstats.npz", allow_pickle=False) as data:
            methods = tuple(data["methods"].tolist())
            if methods != METHODS or tuple(data["identifier_columns"].tolist()) != ("chunk", "head", "position"):
                raise ValueError("raw method or identifier ordering metadata differs from the frozen schema")
            if data["selected_indices"].shape[0] != len(methods):
                raise ValueError("raw selection method count differs from its metadata")
            raw = collect_unions(data["identifiers"], data["selected_indices"], data["selected_counts"],
                                 lengths=lengths, evaluation_chunks=config["evaluation_chunks"],
                                 head_to_kv=head_to_kv, budget=config["budget"], recent_window=config["recent_window"])
        variants.append({"name": family, "summaries": summarize(methods, raw, config["budget"])})
        raw_output.update({f"{family.replace('-', '_')}_{name}": value for name, value in raw.items()
                           if name != "query_head_to_physical_kv_head"})
    # Recheck read-only inputs before creating a fresh output directory.
    if (file_hash(trace_path) != trace_digest or file_hash(model_manifest_path) != model_manifest_digest
            or verify_completion(alignment_directory)[1] != completion_digest
            or Path(__file__).read_bytes() != sources[Path(__file__).name]):
        raise RuntimeError("accounting inputs changed during verification")
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources").mkdir()
    for name, content in sources.items():
        (output / "sources" / name).write_bytes(content)
    np.savez_compressed(output / "union_counts.npz", methods=np.asarray(METHODS),
                        union_identifier_columns=np.asarray(["chunk", "physical_kv_head", "position"]),
                        query_identifier_columns=np.asarray(["chunk", "query_head", "position"]), **raw_output)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "alignment_directory": str(alignment_directory),
                "alignment_completion_sha256": completion_digest, "trace_path": str(trace_path), "trace_sha256": trace_digest,
                "alignment_config_sha256": completed["artifact_sha256"]["config.json"],
                "rawstats_sha256": {family: completed["artifact_sha256"][f"{family}/rawstats.npz"] for family in FAMILIES},
                "model_manifest_path": str(model_manifest_path), "model_manifest_sha256": model_manifest_digest,
                "model": model_manifest["model"], "model_revision": model_manifest["model_revision"],
                "model_loaded_state_sha256": model_manifest["model_loaded_state_sha256"],
                "query_heads": model_manifest["query_heads"], "physical_kv_heads": model_manifest["physical_kv_heads"],
                "gqa_group_size": model_manifest["gqa_groups"], "query_head_to_physical_kv_head": head_to_kv.tolist(),
                "group_mapping_basis": "pinned stock Llama repeat_kv contiguous layout, corroborated by bitwise-equal captured K and V",
                "budget_per_query_head": 32, "mandatory_recent_inside_budget": 8,
                "affected_query_definition": "zero_based_query_position >= 32",
                "shared_recent_union_bound": "min(causal_prefix, r + group_size * (min(causal_prefix,budget) - r)), r=min(recent,budget,causal_prefix)",
                "scope": "Secondary logical selected-position unions; all real K/V remain retained. No actual reads, storage savings, bytes or speed are measured.",
                "python_version": platform.python_version(), "numpy_version": np.__version__, "command": list(sys.argv),
                "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
                "source_snapshots": {name: f"sources/{name}" for name in sources},
                "checks": {"input_artifact_hashes_verified": True, "trace_and_pinned_model_hash_agreement": True,
                           "captured_repeat_kv_groups_bitwise_exact": True, "raw_ids_joined_without_order_assumptions": True,
                           "exact_budget_causal_chronological_recent_selections": True, "shared_recent_union_bound_verified": True}}
    write_json(output / "manifest.json", manifest)
    result = {"status": "alignment_access_completed", "manifest": manifest, "variants": variants}
    write_json(output / "results.json", result)
    artifacts = sorted(path for path in output.rglob("*") if path.is_file())
    completion = {"status": result["status"], "artifact_sha256": {
        str(path.relative_to(output)): file_hash(path) for path in artifacts},
        "artifact_bytes": {str(path.relative_to(output)): path.stat().st_size for path in artifacts},
        "alignment_completion_sha256": completion_digest, "trace_sha256": trace_digest}
    write_json(output / "completed.json", completion)
    digest = file_hash(output / "completed.json")
    (output / "completed.sha256").write_text(digest + "\n")
    result["completion_sha256"] = digest
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.alignment, args.trace, args.model_manifest, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "completion_sha256": result["completion_sha256"], "variants": result["variants"]}, indent=2))


if __name__ == "__main__":
    main()
