"""Frozen paired-document quality screen; requires complete numerical verification."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from padic_lm import heldout

SEEDS = (17, 29, 43)
KINDS = ("p2", "p3", "coarsened_p2", "grid_p2", "grid_p3", "flat", "gaussian", "gaussian_local64")
METHODS = ("stock", "full_control", "recency", "sink_recency", "uniform", "mass_upper") + tuple(
    f"s{s}_{k}" for s in SEEDS for k in KINDS)
ENDPOINTS = ("all", "affected")
CONTEXT = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify_manifest(directory):
    completion_path = directory / "completed.json"
    if (directory / "failed.json").exists():
        raise ValueError("failed parent cannot be analyzed")
    if digest(completion_path) != (directory / "completed.sha256").read_text().strip():
        raise ValueError("parent completion digest differs")
    completion = json.loads(completion_path.read_text())
    for name, expected in completion["artifact_sha256"].items():
        path = directory / name
        if path.stat().st_size != completion["artifact_bytes"][name] or digest(path) != expected:
            raise ValueError(f"parent artifact differs: {name}")
    return completion


def comparison_plan():
    primary = [("primary", f"p{p}", f"s{s}_p{p}", ref, endpoint)
               for p in (2, 3) for s in SEEDS
               for ref in ("stock", "uniform", f"s{s}_coarsened_p2") for endpoint in ENDPOINTS]
    secondary = [("secondary_vs_stock", "", name, "stock", endpoint)
                 for name in METHODS[1:] for endpoint in ENDPOINTS]
    grids = [("secondary_matched_grid", "", f"s{s}_p{p}", f"s{s}_grid_p{p}", endpoint)
             for s in SEEDS for p in (2, 3) for endpoint in ENDPOINTS]
    return primary + secondary + grids


def family_decisions(comparisons, article_count, margin):
    """Require the complete intersection; equality and one failed seed do not pass."""
    output = {}
    for family in ("p2", "p3"):
        required = {(f"s{s}_{family}", ref, endpoint) for s in SEEDS
                    for ref in ("stock", "uniform", f"s{s}_coarsened_p2") for endpoint in ENDPOINTS}
        rows = [r for r in comparisons if r["role"] == "primary" and r["family"] == family]
        actual = [(r["candidate"], r["reference"], r["endpoint"]) for r in rows]
        if len(rows) != 18 or set(actual) != required or len(set(actual)) != len(actual):
            raise ValueError("all18 distinct frozen primary comparisons per family required")
        if any(not np.isfinite(r["upper_975"]) for r in rows):
            raise ValueError("finite upper bounds required")
        failed = [list(key) for key, row in zip(actual, rows) if row["upper_975"] >= margin]
        eligible = article_count >= 40
        output[family] = {"required_comparisons": 18, "comparisons_with_upper_below_margin": 18 - len(failed),
                          "minimum_document_count_met": eligible, "failed_or_inconclusive_comparisons": failed,
                          "passes_frozen_screen": eligible and not failed,
                          "decision_scope": "Proposed fixed-three-seed first-layer quality screen; requires independent statistical audit"}
    return output


def extract_article_sums(directory, config):
    n = len(config["articles"])
    sums = np.empty((len(METHODS), n, 2), dtype=np.float64)
    counts = np.broadcast_to(np.array([2047, 1919], dtype=np.int64), (n, 2)).copy()
    for index in range(n):
        path = directory / "windows" / f"length-2048-article-{index:02d}.npz"
        with np.load(path, allow_pickle=False) as window:
            stock = window["reference_target_nll"]
            if stock.shape != (2047,) or not np.all(np.isfinite(stock)):
                raise ValueError("complete finite2047-target reference required")
            for method_index, name in enumerate(METHODS):
                loss = stock if name == "stock" else window[name + "_target_nll"]
                if loss.shape != stock.shape or not np.all(np.isfinite(loss)) or not np.array_equal(loss[:128], stock[:128]):
                    raise ValueError(f"target loss or unchanged-prefix invariant differs: {index}, {name}")
                loss = loss.astype(np.float64)
                sums[method_index, index] = [loss.sum(), loss[128:].sum()]
    if not np.array_equal(sums[0], sums[1]):
        raise ValueError("full-control document sums differ from stock")
    return sums, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--numerical-audit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh statistical output required")
    protocol = args.protocol.read_bytes()
    audit = json.loads(args.numerical_audit.read_text())
    parent_hash = digest(args.input / "completed.json")
    if (hashlib.sha256(protocol).hexdigest() != heldout.PROTOCOL_HASH
            or audit["status"] != "heldout_quality_audit_passed"
            or audit["evaluation_completion_sha256"] != parent_hash):
        raise ValueError("frozen protocol and complete matching numerical audit required")
    prior = audit["cumulative_gate_seconds"]
    if not np.isfinite(prior) or prior < 0 or prior >= 10800:
        raise ValueError("valid remaining frozen gate wall allowance required")
    args.output.mkdir(parents=True)
    CONTEXT.update(started=started, prior=prior, output=args.output)
    def check_time():
        if prior + time.monotonic() - started >= 10800:
            raise TimeoutError("held-out cumulative10800-second ceiling reached")
    check_time()
    completion = verify_manifest(args.input)
    result = json.loads((args.input / "results.json").read_text())
    config = result["config"]
    n = len(config["articles"])
    if (completion["status"] != "heldout_quality_evaluation_complete"
            or result["status"] != completion["status"] or result["completed_windows"] != n
            or audit["windows_verified"] != n or audit["full_budget_target_loss_identity_windows"] != n
            or audit["complete_model_fixture_readouts_including_references"] != 30
            or not result["original_model_state_unchanged"] or result["new_fitting_or_tuning"]
            or not result["test_split_read"] or config["methods"] != list(METHODS[1:])
            or config["seeds"] != list(SEEDS) or config["lengths"] != [2048]
            or config["layer"] != 0 or config["group_budget"] != 128
            or config["protocol_sha256"] != heldout.PROTOCOL_HASH
            or config["development_completion_sha256"] != heldout.DEVELOPMENT_HASH
            or config["development_audit_sha256"] != heldout.DEVELOPMENT_AUDIT_HASH
            or len({r["article_id"] for r in config["articles"]}) != n):
        raise ValueError("complete frozen all-article/all-method test evidence required")
    paths = [Path(__file__), Path(heldout.__file__)]
    sources = {p.name: p.read_bytes() for p in paths}
    (args.output / "sources").mkdir()
    for name, content in sources.items():
        (args.output / "sources" / name).write_bytes(content)
    (args.output / "prospective_protocol.txt").write_bytes(protocol)
    sums, counts = extract_article_sums(args.input, config)
    indices = np.random.Generator(np.random.PCG64(heldout.RESAMPLE_SEED)).integers(
        0, n, size=(heldout.RESAMPLES, n), dtype=np.int64)
    np.savez_compressed(args.output / "document_bootstrap.npz", article_sums=sums,
                        target_counts=counts, resample_indices=indices,
                        methods=np.asarray(METHODS), endpoints=np.asarray(ENDPOINTS),
                        article_ids=np.asarray([r["article_id"] for r in config["articles"]]))
    comparisons = []
    for role, family, candidate, reference, endpoint in comparison_plan():
        check_time()
        m, r, e = METHODS.index(candidate), METHODS.index(reference), ENDPOINTS.index(endpoint)
        estimate, _ = heldout.paired_document_bootstrap(sums[m, :, e], sums[r, :, e], counts[:, e], indices)
        comparisons.append({"role": role, "family": family, "candidate": candidate, "reference": reference,
                            "endpoint": endpoint, **estimate,
                            "paired_ppl_ratio": float(np.exp(estimate["paired_delta_nll"])),
                            "ppl_ratio_lower_025": float(np.exp(estimate["lower_025"])),
                            "ppl_ratio_upper_975": float(np.exp(estimate["upper_975"])),
                            "upper_below_quality_margin": bool(estimate["upper_975"] < heldout.QUALITY_MARGIN)})
    with (args.output / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    article_records = []
    for i, article in enumerate(config["articles"]):
        article_records.append({"article_id": article["article_id"], "source_order": i,
            "target_counts": dict(zip(ENDPOINTS, counts[i].tolist())),
            "loss_sums": {method: dict(zip(ENDPOINTS, sums[m, i].tolist())) for m, method in enumerate(METHODS)}})
    report = {"status": "heldout_quality_statistics_complete_pending_audit",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_completion_sha256": parent_hash, "numerical_audit_sha256": digest(args.numerical_audit),
              "protocol_sha256": heldout.PROTOCOL_HASH, "source_sha256": {name: hashlib.sha256(b).hexdigest() for name, b in sources.items()},
              "articles": n, "resamples": heldout.RESAMPLES, "resample_seed": heldout.RESAMPLE_SEED,
              "random_generator": "NumPy PCG64; one common paired document-index matrix",
              "resampling_unit": "complete eligible test article; no token/head/seed resampling",
              "resample_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
              "quantile_method": "linear", "quality_margin_nats": float(heldout.QUALITY_MARGIN),
              "targets_by_endpoint": dict(zip(ENDPOINTS, counts.sum(0).tolist())),
              "primary_comparisons": 36, "secondary_vs_stock_comparisons": 58, "secondary_matched_grid_comparisons": 12,
              "comparisons": comparisons, "proposed_family_decisions": family_decisions(comparisons, n, heldout.QUALITY_MARGIN),
              "article_records": article_records,
              "coverage_scope": "Approximate paired-document percentile bootstrap; fixed three seeds. Two-family nominal5% screening allocation assumes valid constituent bounds; no exact finite-sample FWER or seed-population inference.",
              "quality_scope": "One first layer, common128/2048 logical-position budget, real KV retained; one-percent PPL screening allowance. No equality, superiority, p-adic-specific mechanism or native resource advantage.",
              "secondary_scope": "Two-sided2.5/97.5% paired-document estimates; no new superiority or favorable multiple-comparison selection.",
              "numpy_version": np.__version__, "statistical_analysis_wall_seconds": time.monotonic() - started}
    if (args.protocol.read_bytes() != protocol or any(p.read_bytes() != sources[p.name] for p in paths)
            or digest(args.input / "completed.json") != parent_hash):
        raise RuntimeError("frozen inputs or executed sources changed")
    write_json(args.output / "results.json", report)
    artifacts = sorted(p for p in args.output.rglob("*") if p.is_file())
    hashes = {str(p.relative_to(args.output)): digest(p) for p in artifacts}
    sizes = {str(p.relative_to(args.output)): p.stat().st_size for p in artifacts}
    check_time()
    write_json(args.output / "completed.json", {"status": report["status"], "evaluation_completion_sha256": parent_hash,
               "cumulative_gate_seconds": prior + time.monotonic() - started,
               "artifact_sha256": hashes, "artifact_bytes": sizes})
    (args.output / "completed.sha256").write_text(digest(args.output / "completed.json") + "\n")
    check_time()
    print(json.dumps({"status": report["status"], "articles": n, "comparisons": len(comparisons),
                      "cumulative_gate_seconds": prior + time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT:
            out = CONTEXT["output"]
            (out / "completed.json").unlink(missing_ok=True)
            (out / "completed.sha256").unlink(missing_ok=True)
            write_json(out / "failed.json", {"status": "heldout_quality_statistics_failed", "type": type(error).__name__,
                "message": str(error), "cumulative_gate_seconds": CONTEXT["prior"] + time.monotonic() - CONTEXT["started"]})
        raise
