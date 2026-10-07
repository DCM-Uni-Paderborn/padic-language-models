"""Regenerate the frozen PG-19 bootstrap comparisons from compact book endpoints."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from padic_lm.heldout import paired_document_bootstrap


def calculate(rows):
    methods = ["eager_original", "native_full", "recency", "uniform"]
    methods += [f"s{s}_{name}" for s in (17, 29, 43) for name in ("coarsened_p2", "p2", "p3")]
    if {r["method"] for r in rows} != set(methods):
        raise ValueError("requires the original thirteen readouts")
    table = {}
    book_ids = {}
    for length in (2048, 4096):
        for method in methods:
            for endpoint in ("all", "affected"):
                selected = sorted((r for r in rows if r["length"] == length and r["method"] == method and r["endpoint"] == endpoint), key=lambda r: r["book_index"])
                if [r["book_index"] for r in selected] != list(range(40)):
                    raise ValueError("requires all forty book endpoints")
                counts = np.array([r["targets"] for r in selected], dtype=np.int64)
                expected = length - 1 - (128 if endpoint == "affected" else 0)
                if not np.all(counts == expected):
                    raise ValueError("target accounting differs")
                ids = [r["book_id"] for r in selected]
                if length in book_ids and ids != book_ids[length]:
                    raise ValueError("book alignment differs")
                book_ids[length] = ids
                table[length, method, endpoint] = (np.array([r["loss_sum"] for r in selected]), counts)
    if book_ids[2048] != book_ids[4096] or len(rows) != 40*13*2*2:
        raise ValueError("nested corpus or endpoint count differs")
    draws = np.random.Generator(np.random.PCG64(514203)).integers(0, 40, size=(20000, 40), dtype=np.int64)
    primary, secondary, absolute = [], [], []
    def compare(length, method, reference, endpoint):
        sums, counts = table[length, method, endpoint]
        refs, other_counts = table[length, reference, endpoint]
        if not np.array_equal(counts, other_counts): raise ValueError("paired counts differ")
        interval, _ = paired_document_bootstrap(sums, refs, counts, draws)
        return {"length": length, "method": method, "reference": reference, "endpoint": endpoint, **interval,
                "point_relative_ppl_percent": float(100*np.expm1(interval["paired_delta_nll"])),
                "upper_relative_ppl_percent": float(100*np.expm1(interval["upper_975"]))}
    for length in (2048, 4096):
        for method in methods:
            for endpoint in ("all", "affected"):
                sums, counts = table[length, method, endpoint]
                nll = float(sums.sum()/counts.sum())
                absolute.append({"length": length, "method": method, "endpoint": endpoint, "nll": nll, "subword_ppl": float(np.exp(nll))})
                secondary.append(compare(length, method, "native_full", endpoint))
                if method != "eager_original": secondary.append(compare(length, method, "eager_original", endpoint))
        for family in ("p2", "p3"):
            for seed in (17, 29, 43):
                for reference in ("native_full", "uniform", f"s{seed}_coarsened_p2"):
                    for endpoint in ("all", "affected"):
                        row = compare(length, f"s{seed}_{family}", reference, endpoint)
                        row.update(family=family, seed=seed, passed=row["upper_975"] < float(np.log(1.01)))
                        primary.append(row)
    families = {f: {"passed_comparisons": sum(r["passed"] for r in primary if r["family"] == f),
                    "required_comparisons": 36, "screen_passed": all(r["passed"] for r in primary if r["family"] == f)} for f in ("p2", "p3")}
    return {"families": families, "primary": primary, "secondary": secondary, "absolute": absolute,
            "bootstrap_seed": 514203, "draws": 20000, "allowance_nats": float(np.log(1.01))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoints", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError("fresh output required")
    result = calculate(json.loads(a.endpoints.read_text()))
    a.output.mkdir(parents=True)
    (a.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    rows = [{"kind": kind, **r} for kind in ("primary", "secondary") for r in result[kind]]
    with (a.output / "all-paired-comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (a.output / "absolute-endpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(result["absolute"][0])); writer.writeheader(); writer.writerows(result["absolute"])
    print(json.dumps(result["families"]))


if __name__ == "__main__": main()
