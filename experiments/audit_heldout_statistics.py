"""Independent document-frequency/bootstrap-order-statistic audit; no production imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

PROTOCOL_HASH = "9a0095eb8e079773e63d259c292a4d7ad10b932ded90df58026bbf90be2048d0"
SEEDS = (17, 29, 43)
KINDS = ("p2", "p3", "coarsened_p2", "grid_p2", "grid_p3", "flat", "gaussian", "gaussian_local64")
METHODS = ["stock", "full_control", "recency", "sink_recency", "uniform", "mass_upper"] + [
    f"s{s}_{kind}" for s in SEEDS for kind in KINDS]
CONTEXT = {}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for data in iter(lambda: source.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def verify_manifest(directory):
    assert not (directory / "failed.json").exists(), "failed parent"
    assert digest(directory / "completed.json") == (directory / "completed.sha256").read_text().strip()
    record = json.loads((directory / "completed.json").read_text())
    for name, expected in record["artifact_sha256"].items():
        path = directory / name
        assert path.stat().st_size == record["artifact_bytes"][name] and digest(path) == expected, name
    return record


def linear_order_statistic(samples, probability):
    """Explicit interpolated order statistic, independent of np.quantile."""
    ordered = sorted(float(x) for x in samples)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    fraction = location - lower
    return ordered[lower] + fraction * (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower])


def frequency_bootstrap(candidate, reference, counts, frequencies):
    """Multiplicity-weighted document sums rather than gathering loss vectors."""
    differences = np.asarray(candidate) - np.asarray(reference)
    estimates = (frequencies @ differences) / (frequencies @ np.asarray(counts))
    point = math.fsum(float(x) for x in differences) / math.fsum(float(x) for x in counts)
    return {"paired_delta_nll": point, "lower_025": linear_order_statistic(estimates, .025),
            "upper_975": linear_order_statistic(estimates, .975)}, estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--numerical-audit", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh statistical-audit output required")
    completion = json.loads((args.statistics / "completed.json").read_text())
    prior = completion["cumulative_gate_seconds"]
    assert np.isfinite(prior) and 0 <= prior < 10800
    args.output.mkdir(parents=True)
    CONTEXT.update(started=started, prior=prior, output=args.output)
    def check_time():
        if prior + time.monotonic() - started >= 10800:
            raise TimeoutError("held-out cumulative10800-second ceiling reached")
    source = Path(__file__).read_bytes()
    (args.output / "audit_source.py").write_bytes(source)
    protocol = args.protocol.read_bytes()
    assert hashlib.sha256(protocol).hexdigest() == PROTOCOL_HASH
    check_time()
    verify_manifest(args.evaluation)
    verify_manifest(args.statistics)
    report = json.loads((args.statistics / "results.json").read_text())
    evaluation = json.loads((args.evaluation / "results.json").read_text())
    audit = json.loads(args.numerical_audit.read_text())
    parent_hash = digest(args.evaluation / "completed.json")
    assert audit["status"] == "heldout_quality_audit_passed" and audit["evaluation_completion_sha256"] == parent_hash
    assert report["status"] == "heldout_quality_statistics_complete_pending_audit"
    assert completion["evaluation_completion_sha256"] == report["evaluation_completion_sha256"] == parent_hash
    assert report["numerical_audit_sha256"] == digest(args.numerical_audit)
    assert report["protocol_sha256"] == evaluation["config"]["protocol_sha256"] == PROTOCOL_HASH
    assert report["resamples"] == 20000 and report["resample_seed"] == 514203 and report["quantile_method"] == "linear"
    assert report["quality_margin_nats"] == np.log(1.01)
    articles = evaluation["config"]["articles"]
    count = len(articles)
    assert count == report["articles"] == audit["windows_verified"] == evaluation["completed_windows"]
    assert evaluation["config"]["methods"] == METHODS[1:] and evaluation["config"]["seeds"] == list(SEEDS)
    assert evaluation["config"]["lengths"] == [2048] and evaluation["config"]["layer"] == 0
    assert evaluation["original_model_state_unchanged"] and not evaluation["new_fitting_or_tuning"] and evaluation["test_split_read"]
    for name, expected in report["source_sha256"].items():
        assert digest(args.statistics / "sources" / name) == expected
    with np.load(args.statistics / "document_bootstrap.npz", allow_pickle=False) as saved:
        sums = saved["article_sums"].copy()
        targets = saved["target_counts"].copy()
        indices = saved["resample_indices"].copy()
        assert saved["methods"].tolist() == METHODS and saved["endpoints"].tolist() == ["all", "affected"]
        assert saved["article_ids"].tolist() == [a["article_id"] for a in articles]
    assert sums.shape == (30, count, 2) and targets.shape == (count, 2)
    assert np.issubdtype(targets.dtype, np.integer) and np.all(targets == np.array([2047, 1919]))
    assert np.all(np.isfinite(sums)) and indices.dtype == np.dtype("int64")
    expected_draws = np.random.Generator(np.random.PCG64(514203)).integers(0, count, size=(20000, count), dtype=np.int64)
    assert np.array_equal(indices, expected_draws)
    assert hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest() == report["resample_indices_sha256"]
    frequencies = np.stack([np.bincount(row, minlength=count) for row in indices])
    assert np.all(frequencies.sum(1) == count)
    maximum_sum_difference = 0.
    reconstructed = np.empty_like(sums)
    assert len(report["article_records"]) == count
    for i, article in enumerate(articles):
        check_time()
        row = report["article_records"][i]
        assert row["article_id"] == article["article_id"] and row["source_order"] == i
        assert row["target_counts"] == {"all": 2047, "affected": 1919}
        with np.load(args.evaluation / "windows" / f"length-2048-article-{i:02d}.npz", allow_pickle=False) as raw:
            stock = raw["reference_target_nll"]
            assert stock.shape == (2047,) and np.all(np.isfinite(stock))
            for m, method in enumerate(METHODS):
                loss = stock if method == "stock" else raw[method + "_target_nll"]
                assert loss.shape == (2047,) and np.all(np.isfinite(loss)) and np.array_equal(loss[:128], stock[:128])
                values = (math.fsum(float(x) for x in loss), math.fsum(float(x) for x in loss[128:]))
                for e, endpoint in enumerate(("all", "affected")):
                    reconstructed[m, i, e] = values[e]
                    error = abs(values[e] - sums[m, i, e])
                    maximum_sum_difference = max(maximum_sum_difference, error)
                    assert error < 4e-10 and row["loss_sums"][method][endpoint] == sums[m, i, e]
    assert np.array_equal(reconstructed[0], reconstructed[1])
    assert report["targets_by_endpoint"] == {"all": count * 2047, "affected": count * 1919}
    expected = []
    for prime in (2, 3):
        for seed in SEEDS:
            for comparator in ("stock", "uniform", f"s{seed}_coarsened_p2"):
                for endpoint in ("all", "affected"):
                    expected.append(("primary", f"p{prime}", f"s{seed}_p{prime}", comparator, endpoint))
    for method in METHODS[1:]:
        for endpoint in ("all", "affected"):
            expected.append(("secondary_vs_stock", "", method, "stock", endpoint))
    for seed in SEEDS:
        for prime in (2, 3):
            for endpoint in ("all", "affected"):
                expected.append(("secondary_matched_grid", "", f"s{seed}_p{prime}", f"s{seed}_grid_p{prime}", endpoint))
    assert len(expected) == 106 and len(report["comparisons"]) == 106
    assert (report["primary_comparisons"], report["secondary_vs_stock_comparisons"], report["secondary_matched_grid_comparisons"]) == (36, 58, 12)
    maximum_bound_difference = 0.
    independently_reconstructed = []
    for identity, row in zip(expected, report["comparisons"]):
        check_time()
        assert tuple(row[k] for k in ("role", "family", "candidate", "reference", "endpoint")) == identity
        _, _, candidate, reference, endpoint = identity
        m, r, e = METHODS.index(candidate), METHODS.index(reference), ("all", "affected").index(endpoint)
        truth, _ = frequency_bootstrap(reconstructed[m, :, e], reconstructed[r, :, e], targets[:, e], frequencies)
        for key, value in truth.items():
            error = abs(value - row[key])
            maximum_bound_difference = max(maximum_bound_difference, error)
            assert error < 3e-13, (identity, key, error)
        for output_key, original_key in (("paired_ppl_ratio", "paired_delta_nll"), ("ppl_ratio_lower_025", "lower_025"), ("ppl_ratio_upper_975", "upper_975")):
            assert abs(row[output_key] - math.exp(truth[original_key])) < 4e-13
        below = truth["upper_975"] < np.log(1.01)
        assert below == row["upper_below_quality_margin"]
        independently_reconstructed.append({**row, **truth, "independent_upper_below_margin": below})
    verified_families = {}
    for family in ("p2", "p3"):
        rows = [row for row in independently_reconstructed if row["role"] == "primary" and row["family"] == family]
        assert len(rows) == 18
        failed = [[r["candidate"], r["reference"], r["endpoint"]] for r in rows if not r["independent_upper_below_margin"]]
        passed = count >= 40 and not failed
        proposed = report["proposed_family_decisions"][family]
        assert proposed["required_comparisons"] == 18 and proposed["comparisons_with_upper_below_margin"] == 18 - len(failed)
        assert proposed["minimum_document_count_met"] == (count >= 40)
        assert proposed["failed_or_inconclusive_comparisons"] == failed and proposed["passes_frozen_screen"] == passed
        verified_families[family] = {"passes_frozen_screen": passed, "required_comparisons": 18,
            "comparisons_with_upper_below_margin": 18 - len(failed), "minimum_document_count_met": count >= 40,
            "failed_or_inconclusive_comparisons": failed,
            "classification": "quality non-inferior in the frozen three-seed first-layer study" if passed else "non-inferiority not established by the frozen screen"}
    assert Path(__file__).read_bytes() == source and args.protocol.read_bytes() == protocol
    assert digest(args.evaluation / "completed.json") == parent_hash
    assert digest(args.statistics / "completed.json") == (args.statistics / "completed.sha256").read_text().strip()
    check_time()
    output = {"status": "heldout_quality_statistics_audit_passed", "created_utc": datetime.now(timezone.utc).isoformat(),
              "evaluation_completion_sha256": parent_hash, "numerical_audit_sha256": digest(args.numerical_audit),
              "statistics_completion_sha256": digest(args.statistics / "completed.json"), "audit_source_sha256": digest(__file__),
              "protocol_sha256": PROTOCOL_HASH, "articles_verified": count, "raw_target_loss_elements": 30 * count * 2047,
              "document_endpoint_sums_verified": 30 * count * 2, "resampling_indices_exact": 20000 * count,
              "comparison_intervals_verified": 106, "maximum_document_sum_difference": maximum_sum_difference,
              "maximum_point_or_bound_difference": maximum_bound_difference, "verified_family_decisions": verified_families,
              "audit_wall_seconds": time.monotonic() - started, "cumulative_gate_seconds": prior + time.monotonic() - started,
              "algorithm_scope": "No production imports; math.fsum raw-document reconstruction; document-multiplicity matrix products; explicitly interpolated sorted order statistics; exact PCG64 draw identity.",
              "inference_scope": "Approximate paired-document bootstrap; fixed three seeds; first layer on one corpus/model. Passing quality screen does not establish equality, superiority, p-adic-specific mechanism or native resource advantage."}
    (args.output / "audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    check_time()
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT:
            (CONTEXT["output"] / "audit.json").unlink(missing_ok=True)
            (CONTEXT["output"] / "failed.json").write_text(json.dumps({"status": "heldout_quality_statistics_audit_failed",
                "type": type(error).__name__, "message": str(error),
                "cumulative_gate_seconds": CONTEXT["prior"] + time.monotonic() - CONTEXT["started"]}, indent=2) + "\n")
        raise
