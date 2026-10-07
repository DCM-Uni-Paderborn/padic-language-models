"""Display every frozen quality comparison after both real-result audits pass."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {"full_control": "Full causal control", "recency": "Recency", "sink_recency": "4 sinks + recency",
          "uniform": "Uniform + recent", "mass_upper": "Native mass upper control",
          "p2": "Binary balls", "p3": "Ternary balls", "coarsened_p2": "Coarsened binary hierarchy",
          "grid_p2": "Binary quantile grid", "grid_p3": "Ternary quantile grid", "flat": "Flat centroids",
          "gaussian": "Gaussian", "gaussian_local64": "Gaussian, local/global64/64"}
ENDPOINTS = ("all", "affected")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def display(name):
    if name.startswith("s") and name != "sink_recency":
        seed, kind = name.split("_", 1)
        return f"{LABELS[kind]} / seed {seed[1:]}"
    return LABELS[name]


def interval_axis(ax, rows, labels, margin):
    y = np.arange(len(rows))
    for position, row in enumerate(rows):
        lo, point, hi = (row[k] for k in ("lower_025", "paired_delta_nll", "upper_975"))
        if not np.all(np.isfinite([lo, point, hi])) or lo > hi:
            raise ValueError("invalid audited interval")
        color = "#bc642c" if row["candidate"].endswith("_p3") else "#286493"
        ax.plot([lo, hi], [position, position], color=color, lw=1.4)
        ax.scatter([point], [position], color=color, s=22, zorder=3)
    ax.axvline(0, color="#777777", lw=.8)
    ax.axvline(margin, color="#a32635", lw=1.1, ls="--")
    ax.set_yticks(y, labels)
    ax.set_ylim(len(rows) - .5, -.5)
    ax.grid(axis="x", alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Paired delta NLL (nats/target)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh figure destination required")
    completed = json.loads((args.statistics / "completed.json").read_text())
    completed_hash = digest(args.statistics / "completed.json")
    assert completed_hash == (args.statistics / "completed.sha256").read_text().strip()
    for name, value in completed["artifact_sha256"].items():
        path = args.statistics / name
        assert path.stat().st_size == completed["artifact_bytes"][name] and digest(path) == value
    result = json.loads((args.statistics / "results.json").read_text())
    audit = json.loads(args.audit.read_text())
    assert audit["status"] == "heldout_quality_statistics_audit_passed"
    assert audit["statistics_completion_sha256"] == completed_hash
    assert audit["evaluation_completion_sha256"] == result["evaluation_completion_sha256"]
    assert audit["protocol_sha256"] == result["protocol_sha256"]
    comparisons = result["comparisons"]
    assert len(comparisons) == audit["comparison_intervals_verified"] == 106
    lookup = {(r["role"], r["candidate"], r["reference"], r["endpoint"]): r for r in comparisons}
    assert len(lookup) == 106
    # Keep the protocol's complete order, without ranking/filtering by results.
    names = [r["candidate"] for r in comparisons if r["role"] == "secondary_vs_stock" and r["endpoint"] == ENDPOINTS[0]]
    assert len(names) == 29 and len(set(names)) == 29
    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 10.5), sharey=True)
    titles = (f"All {result['targets_by_endpoint']['all']:,} targets", f"Affected {result['targets_by_endpoint']['affected']:,} targets")
    for ax, endpoint, title in zip(axes, ENDPOINTS, titles):
        rows = [lookup[("secondary_vs_stock", name, "stock", endpoint)] for name in names]
        interval_axis(ax, rows, [display(n) for n in names], result["quality_margin_nats"])
        ax.set_title(title)
    axes[1].tick_params(labelleft=False)
    fig.suptitle(f"Untouched {result['articles']}-article test: all 29 readouts versus stock", fontsize=14)
    fig.text(.02, .014, "20,000 paired-document draws; approximate two-sided 95% percentile intervals. Red line: frozen 1% PPL margin.\n"
             "All 29 stock comparisons are secondary; family screening also requires uniform/coarsened comparators, all seeds and both endpoints.\n"
             "First layer, 128/2048 logical positions; original real KV retained. No native efficiency or superiority claim.", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .96))
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"heldout-all-readouts.{ext}", dpi=170)
    plt.close(fig)
    candidates = [f"s{s}_{family}" for s in (17, 29, 43) for family in ("p2", "p3")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for e, endpoint in enumerate(ENDPOINTS):
        for column, ref_kind in enumerate(("stock", "uniform", "coarsened_p2")):
            rows = []
            for name in candidates:
                reference = name.split("_", 1)[0] + "_coarsened_p2" if ref_kind == "coarsened_p2" else ref_kind
                rows.append(lookup[("primary", name, reference, endpoint)])
            ax = axes[e, column]
            interval_axis(ax, rows, [display(n) for n in candidates], result["quality_margin_nats"])
            ax.set_title(f"{endpoint.replace('_', ' ')} / {ref_kind.replace('_', ' ')}")
            if column:
                ax.tick_params(labelleft=False)
    fig.suptitle("All 36 frozen primary comparisons", fontsize=14)
    fig.text(.02, .013, "Shown: paired-document 2.5/97.5% bounds; the screening uses the 97.5% upper bound strictly below log(1.01).\n"
             "Each family requires all 18 seed/comparator/endpoint bounds and at least 40 articles. Coverage is approximate; seeds are fixed.\n"
             "A quality pass concerns this single first-layer study; it does not establish a competitive resource operating point.", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, .95))
    for ext in ("png", "svg"):
        fig.savefig(args.output / f"heldout-primary-screen.{ext}", dpi=170)
    plt.close(fig)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest = {"status": "audited_heldout_comparisons_plotted_without_selection",
                "statistics_completion_sha256": completed_hash, "statistical_audit_sha256": digest(args.audit),
                "plot_source_sha256": digest(__file__), "readouts": names, "primary_comparisons": 36,
                "secondary_stock_intervals": 58, "family_decisions": audit["verified_family_decisions"],
                "files_sha256": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
