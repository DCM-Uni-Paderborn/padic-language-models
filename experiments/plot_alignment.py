"""Export scientific figures from a completed, hash-verified alignment gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="completed alignment run directory")
    parser.add_argument("--trace", type=Path, required=True, help="original cached Q/K/V trace")
    parser.add_argument("--output", type=Path, required=True, help="fresh figure directory")
    args = parser.parse_args()
    run, trace, output = args.input.resolve(), args.trace.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("use a fresh figure directory")
    done = json.loads((run / "completed.json").read_text())
    if done["status"] != "alignment_diagnostic_completed":
        raise ValueError("the alignment diagnostic must have completed")
    artifacts = done["artifact_sha256"]

    def verified(relative: str) -> Path:
        path = run / relative
        if file_hash(path) != artifacts[relative]:
            raise ValueError(f"archived artifact hash mismatch: {relative}")
        return path

    results = json.loads(verified("results.json").read_text())
    if file_hash(trace) != results["config"]["trace_sha256"]:
        raise ValueError("trace differs from the executed input")
    states_path = verified("pca-17/encoder_state.npz")
    with np.load(states_path, allow_pickle=False) as state:
        center, projection = state["p2_center"], state["p2_projection"]
        thresholds, digits = state["p2_thresholds"], state["p2_digits"]
    with np.load(trace, allow_pickle=False) as arrays:
        calibration = results["config"]["calibration_chunks"]
        queries, keys = arrays["queries"][calibration], arrays["keys"][calibration]
    if queries.shape != keys.shape or queries.shape[1] != len(center):
        raise ValueError("trace and archived encoder head layouts disagree")
    lengths = np.asarray(results["config"]["valid_lengths"])[calibration]
    valid = np.arange(queries.shape[2])[None, :] < lengths[:, None]

    output.mkdir()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    heads = len(center)
    columns, rows = 3, (heads + 2) // 3
    fig, axes = plt.subplots(rows, columns, figsize=(10.5, 3.0 * rows), squeeze=False)
    handles = None
    for head, ax in enumerate(axes.flat):
        if head >= heads:
            ax.set_visible(False)
            continue
        q = (queries[:, head][valid].astype(np.float64) - center[head]) @ projection[head]
        k = (keys[:, head][valid].astype(np.float64) - center[head]) @ projection[head]
        qpoints = ax.scatter(q[:, 0], q[:, 1], s=9, alpha=.45, color="#2366ab", label="Queries")
        kpoints = ax.scatter(k[:, 0], k[:, 1], s=9, alpha=.45, color="#c75445", label="Keys")
        # The first reversed binary digit is the original bin's highest bit.
        cuts = [thresholds[head, j, 2 ** int(digits[j]) // 2 - 1] for j in (0, 1)]
        vertical = ax.axvline(cuts[0], color="#555555", linestyle="--", linewidth=.8, label="Coarse binary boundary")
        ax.axhline(cuts[1], color="#555555", linestyle="--", linewidth=.8)
        ax.set_title(f"Head {head}")
        ax.set_xlabel("Shared PCA coordinate 1")
        ax.set_ylabel("Shared PCA coordinate 2")
        handles = [qpoints, kpoints, vertical]
    fig.suptitle("First-layer calibration Q/K and coarse binary partitions\nArchived shared PCA; all heads, first two chunks", y=1.01)
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.035))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"calibration-pca.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    variants = results["variants"]
    if [v["name"] for v in variants] != ["pca-17", "random-17", "random-29", "random-43"]:
        raise ValueError("plot expects all four prescribed embedding families")
    methods = [("padic_2", "p = 2", "#2366ab"), ("padic_3", "p = 3", "#d88b29"),
               ("flat_euclidean", "Flat Euclidean", "#368c61"),
               ("flat_dot_product", "Flat dot product", "#895da3")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    x, bar_width = np.arange(len(variants)), .18
    recency = []
    for variant in variants:
        rows_by_method = {row["method"]: row["all_heads"] for row in variant["summaries"]
                          if row["stratum"] == "affected_queries"}
        recency.append(rows_by_method["recency"])
    for index, (method, label, color) in enumerate(methods):
        values = [next(row["all_heads"] for row in v["summaries"]
                       if row["method"] == method and row["stratum"] == "affected_queries") for v in variants]
        for ax, metric in zip(axes, ("value_output_nrmse", "kept_mass_mean")):
            ax.bar(x + (index - 1.5) * bar_width, [row[metric] for row in values],
                   width=bar_width, color=color, label=label)
    for ax, metric in zip(axes, ("value_output_nrmse", "kept_mass_mean")):
        reference_values = [row[metric] for row in recency]
        if not np.array_equal(reference_values, np.repeat(reference_values[0], len(reference_values))):
            raise ValueError("recency control changed between embedding families")
        ax.axhline(reference_values[0], color="#333333", linestyle="--", linewidth=1.1, label="Recency")
        ax.set_xticks(x, ["PCA", "Random 17", "Random 29", "Random 43"])
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pre-projection value-output NRMSE")
    axes[0].set_title("Value reconstruction error")
    axes[1].set_ylabel("Retained dense attention mass")
    axes[1].set_title("Attention mass")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    fig.suptitle("Affected queries 32–127; budget 32 including recent 8\nOne-byte addresses; encoder and centroid metadata differ", y=1.03)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(.5, -.065))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"encoding-comparison.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    shutil.copyfile(Path(__file__), output / "plot_alignment.py")
    record = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "run": str(run), "trace_sha256": file_hash(trace),
              "input_completion_sha256": file_hash(run / "completed.json"),
              "results_sha256": file_hash(run / "results.json"),
              "encoder_state_sha256": file_hash(states_path),
              "source_sha256": file_hash(Path(__file__)), "matplotlib": matplotlib.__version__,
              "numpy": np.__version__, "scope": "Descriptive archived-state figures; no fit or new model evaluation",
              "artifact_sha256": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "figure-manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
