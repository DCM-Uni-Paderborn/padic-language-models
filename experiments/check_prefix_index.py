"""Frozen first-article complete-fixture equivalence for causal finite-prefix index."""
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
from padic_lm import prefix_index
from padic_lm.prefix_index import CausalPrefixIndex

PARENT = "ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8"
AUDIT = "28b682bdf864333c6161606cfc4e6c67260551cab60fe5eb6f162676a6439a36"
CONTEXT = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def retained_object_graph_bytes(obj, seen=None):
    """Recursive retained Python graph, not process RSS or native packed memory."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(retained_object_graph_bytes(k, seen) + retained_object_graph_bytes(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        total += sum(retained_object_graph_bytes(v, seen) for v in obj)
    elif hasattr(obj, "__dict__"):
        total += retained_object_graph_bytes(vars(obj), seen)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh correctness output required")
    if digest(args.development / "completed.json") != PARENT or digest(args.development_audit) != AUDIT:
        raise ValueError("requires frozen development and passed full audit")
    parent = json.loads((args.development / "completed.json").read_text())
    parent_audit = json.loads(args.development_audit.read_text())
    assert parent_audit["status"] == "shared_budget_audit_passed"
    if (args.development / "failed.json").exists():
        raise ValueError("failed input")
    args.output.mkdir(parents=True)
    CONTEXT.update(started=started, output=args.output)
    def check_time():
        if time.monotonic() - started >= 3600:
            raise TimeoutError("separate prefix-index3600-second process guard reached")
    protocol = args.protocol.read_bytes()
    paths = [Path(__file__), Path(prefix_index.__file__)]
    sources = {p.name: p.read_bytes() for p in paths}
    (args.output / "sources").mkdir()
    for name, content in sources.items():
        (args.output / "sources" / name).write_bytes(content)
    (args.output / "prospective_protocol.txt").write_bytes(protocol)
    records = []
    input_hashes = {}
    mask_rows = 0
    for length in (512, 2048):
        name = f"windows/length-{length}-article-00.npz"
        path = args.development / name
        if path.stat().st_size != parent["artifact_bytes"][name] or digest(path) != parent["artifact_sha256"][name]:
            raise ValueError("selected complete input fixture differs")
        input_hashes[name] = parent["artifact_sha256"][name]
        with np.load(path, allow_pickle=False) as raw:
            for seed in (17, 29, 43):
                for kind, prime, digits, coarsen in (("p2", 2, (4, 4), 1), ("p3", 3, (3, 2), 1), ("coarsened_p2", 2, (4, 4), 2)):
                    check_time()
                    family = "p3" if kind == "p3" else "p2"
                    queries = raw[f"s{seed}_{family}_query_codes"][..., 0]
                    keys = raw[f"s{seed}_{family}_key_codes"][..., 0]
                    expected = np.unpackbits(raw[f"s{seed}_{kind}_group_selection_bits"], axis=-1, bitorder="little")[..., :length].astype(bool)
                    assert queries.shape == keys.shape == (15, length) and expected.shape == (5, length, length)
                    indices = [CausalPrefixIndex(prime, digits, 3, coarsen) for _ in range(5)]
                    counters = {k: 0 for k in ("optional_keys", "head_list_entries_yielded", "posting_entries_examined", "distinct_candidates_scored")}
                    endings = {}
                    append_seconds = query_seconds = 0.
                    for position in range(length):
                        check_time()
                        for group, index in enumerate(indices):
                            selected = np.zeros(length, dtype=bool)
                            if position < 128:
                                selected[:position + 1] = True
                            else:
                                stop = position + 1 - 8
                                clock = time.monotonic()
                                while index.keys < stop:
                                    index.append(keys[group * 3:group * 3 + 3, index.keys])
                                append_seconds += time.monotonic() - clock
                                clock = time.monotonic()
                                chosen, info = index.query(queries[group * 3:group * 3 + 3, position], 120)
                                query_seconds += time.monotonic() - clock
                                selected[chosen] = True
                                selected[stop:position + 1] = True
                                for key in counters:
                                    counters[key] += info[key]
                                endings[info["termination"]] = endings.get(info["termination"], 0) + 1
                            if not np.array_equal(selected, expected[group, position]):
                                raise AssertionError(f"complete-fixture mask mismatch: {length}, {seed}, {kind}, {position}, {group}")
                            assert not selected[position + 1:].any() and selected.sum() == min(position + 1, 128)
                            assert selected[max(0, position - 7):position + 1].all()
                            mask_rows += 1
                    record = {"length": length, "seed": seed, "kind": kind, "group_mask_rows_exact": 5 * length,
                              "affected_group_queries": 5 * (length - 128), **counters,
                              "termination_counts": endings, "append_wall_seconds": append_seconds, "query_wall_seconds": query_seconds,
                              "original_key_code_array_bytes": int(keys.nbytes), "prefix_tables_bytes": sum(i.prefix.nbytes for i in indices),
                              "logical_posting_positions": sum(i.stored_position_entries() for i in indices),
                              "hypothetical_uint32_postings_bytes": 4 * sum(i.stored_position_entries() for i in indices),
                              "retained_python_index_graph_bytes": retained_object_graph_bytes(indices),
                              "work_scope": "Counts have different units; not hardware bytes/FLOPs. Recursive graph excludes original KV/window/model and transient per-query scratch."}
                    records.append(record)
                    print(json.dumps({"length": length, "seed": seed, "kind": kind, "rows_exact": 5 * length}), flush=True)
    assert mask_rows == 115200 and len(records) == 18
    if args.protocol.read_bytes() != protocol or any(p.read_bytes() != sources[p.name] for p in paths):
        raise RuntimeError("executed source or protocol changed")
    check_time()
    output = {"status": "causal_prefix_index_complete_fixtures_exact", "created_utc": datetime.now(timezone.utc).isoformat(),
              "development_completion_sha256": PARENT, "development_audit_sha256": AUDIT,
              "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
              "protocol_sha256": hashlib.sha256(protocol).hexdigest(), "selected_input_sha256": input_hashes,
              "group_mask_rows_exact": mask_rows, "complete_article_context_fixtures": 2, "records": records,
              "python_version": sys.version, "numpy_version": np.__version__, "platform": platform.platform(),
              "process_wall_seconds": time.monotonic() - started,
              "scope": "Exact selector identity, source-pinned first development article at both full lengths/all seeds/finite controls. CPU Python prototype; no new model forward, native speed or worst-case sublinear guarantee."}
    (args.output / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    artifacts = sorted(p for p in args.output.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(args.output)): digest(p) for p in artifacts}
    check_time()
    completion = {"status": output["status"], "artifact_sha256": hashes,
                  "artifact_bytes": {str(p.relative_to(args.output)): p.stat().st_size for p in artifacts},
                  "process_wall_seconds": time.monotonic() - started}
    (args.output / "completed.json").write_text(json.dumps(completion, indent=2) + "\n")
    (args.output / "completed.sha256").write_text(digest(args.output / "completed.json") + "\n")
    print(json.dumps({"status": output["status"], "rows_exact": mask_rows, "process_wall_seconds": completion["process_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT:
            output = CONTEXT["output"]
            (output / "completed.json").unlink(missing_ok=True)
            (output / "completed.sha256").unlink(missing_ok=True)
            (output / "failed.json").write_text(json.dumps({"status": "causal_prefix_index_correctness_failed", "type": type(error).__name__,
                "message": str(error), "process_wall_seconds": time.monotonic() - CONTEXT["started"]}, indent=2) + "\n")
        raise
