"""Plot every declared shared-budget readout after completion and verification."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("fresh figure directory required")
    complete_hash = sha(args.input / "completed.json")
    assert complete_hash == (args.input / "completed.sha256").read_text().strip()
    audit = json.loads(args.audit.read_text())
    assert audit["status"] == "shared_budget_audit_passed" and audit["development_completion_sha256"] == complete_hash
    result = json.loads((args.input / "results.json").read_text())
    complete = json.loads((args.input / "completed.json").read_text())
    assert sha(args.input / "results.json") == complete["artifact_sha256"]["results.json"]
    names = result["config"]["methods"]
    assert len(names) == 29
    labels = {"full_control": "Full causal control", "recency": "Recency", "sink_recency": "4 sinks + recency",
              "uniform": "Uniform + recent", "mass_upper": "Native mass upper control",
              "p2": "Binary balls", "p3": "Ternary balls", "coarsened_p2": "Coarsened binary hierarchy",
              "grid_p2": "Binary quantile grid", "grid_p3": "Ternary quantile grid", "flat": "Flat centroids",
              "gaussian": "Gaussian", "gaussian_local64": "Gaussian, local/global 64/64"}
    display = []
    for name in names:
        if name.startswith("s") and name != "sink_recency":
            seed, kind = name.split("_", 1)
            display.append(f"{labels[kind]} · seed {seed[1:]}")
        else:
            display.append(labels[name])
    fig, axes = plt.subplots(1, 2, figsize=(13, 10.3), sharey=True)
    y = np.arange(len(names))
    for length, color, marker in ((512, "#24649a", "o"), (2048, "#d56a29", "s")):
        methods = result["length_results"][str(length)]["methods"]
        changes = np.array([100 * (methods[n]["perplexity_ratio"] - 1) for n in names])
        errors = np.array([methods[n]["affected_projected_nrmse"] for n in names])
        for ax, values in zip(axes, (changes, errors)):
            ax.scatter(values, y, c=color, marker=marker, s=33, label=f"{length:,} tokens", zorder=3)
    axes[0].set_xscale("symlog", linthresh=.02, linscale=1)
    axes[0].axvline(0, color="#555555", linewidth=.8)
    axes[0].set_xlabel("Relative PPL change vs stock (%)\nSymmetric log scale; linear within ±0.02%")
    axes[1].set_xlabel("Native projected NRMSE\nAffected query positions 128…L−1")
    axes[0].set_yticks(y, display)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=.22)
        for boundary in (4.5, 12.5, 20.5):
            ax.axhline(boundary, color="#aaaaaa", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Shared GQA budget: all 29 frozen readouts", fontsize=15)
    fig.text(.02, .015, "43 development articles; nested prefixes; first-layer intervention. Sparse group budget 128; full control uses all causal keys.\n"
             "Original real KV remain resident. Descriptive quality/error; no confirmation, native-I/O or speed claim.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .965))
    args.output.mkdir(parents=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"shared-quality.{extension}", dpi=180)
    plt.close(fig)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    manifest = {"development_completion_sha256": complete_hash, "audit_sha256": sha(args.audit),
                "plot_source_sha256": sha(__file__), "methods": names,
                "figures_sha256": {f.name: sha(f) for f in args.output.glob("shared-quality.*")},
                "scope": "All29 declared readouts, two correlated context lengths, descriptive development"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
