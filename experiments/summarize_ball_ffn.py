"""Verify closed P1 archives and export every development result without selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError("fresh summary required")
    completion = json.loads((args.pilot/"completed.json").read_text())
    for name, pin in completion["artifact_sha256"].items():
        if digest(args.pilot/name) != pin or (args.pilot/name).stat().st_size != completion["artifact_bytes"][name]:
            raise ValueError("pilot artifact changed: "+name)
    audit = json.loads((args.audit/"audit.json").read_text())
    if audit["status"] != "ball_ffn_development_pilot_independently_verified" or audit["pilot_completion_sha256"] != digest(args.pilot/"completed.json"):
        raise ValueError("independent audit lineage differs")
    data = json.loads((args.pilot/"results.json").read_text())
    out.mkdir(parents=True)
    fields = ("seed", "method", "stage", "training_nrmse", "development_nrmse", "development_bf16_nrmse",
              "development_feature_rank", "development_constant_features", "nll", "delta_nll", "accepted_moves",
              "proposals", "candidate_array_bytes", "archived_bundle_bytes")
    rows = []
    for r in data["results"]:
        row = {name:r.get(name, "") for name in fields}
        row.update(training_nrmse=r.get("training", {}).get("nrmse", ""),
            development_nrmse=r.get("development", {}).get("nrmse", ""),
            development_bf16_nrmse=r.get("development", {}).get("bf16_output_nrmse", ""),
            development_feature_rank=r.get("development", {}).get("feature_rank", ""),
            development_constant_features=r.get("development", {}).get("constant_features", ""))
        rows.append(row)
    with (out/"all-results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    aliases = dict(reverse_ball="Reversed balls", ordered_ball="Ordered balls", shuffled_ball="Shuffled balls",
        reverse_signed="Signed powers", real_threshold="Real thresholds", unmixed_ball="Unmixed balls",
        real_linear="Real linear", zero="Zero FFN", mean="Mean FFN")
    lines = []
    for r in data["results"]:
        if r["stage"] != "final" or r["seed"] is None:
            continue
        lines.append(f'{aliases[r["method"]]} & {r["seed"]} & {r["development"]["nrmse"]:.6f} & {r["delta_nll"]:.6f} & {r["accepted_moves"]} \\\\')
    for r in data["results"]:
        if r["seed"] is None:
            lines.append(f'{aliases[r["method"]]} & -- & -- & {r["delta_nll"]:.6f} & -- \\\\')
    (out/"table-rows.tex").write_text("\n".join(lines)+"\n")
    primary = [r for r in data["results"] if r["method"] == "reverse_ball" and r["stage"] == "final"]
    summary = dict(status="all_ball_ffn_development_results_exported", stock_nll=data["stock_nll"],
        records=len(rows), pilot_completion_sha256=digest(args.pilot/"completed.json"), audit_sha256=digest(args.audit/"audit.json"),
        pilot_files=len(completion["artifact_sha256"])+1,
        pilot_bytes=sum(completion["artifact_bytes"].values())+(args.pilot/"completed.json").stat().st_size,
        primary_development_nrmse_range=[min(r["development"]["nrmse"] for r in primary),max(r["development"]["nrmse"] for r in primary)],
        primary_delta_nll_range=[min(r["delta_nll"] for r in primary),max(r["delta_nll"] for r in primary)],
        primary_training_sse_unchanged=all(r["accepted_moves"] == 0 and r["ring_weight_changes"] == 0 for r in primary),
        progression_all_seeds=audit["progression_all_seeds"], secondary_process_seconds=completion["elapsed_process_seconds"]+audit["elapsed_process_seconds"],
        audit_residue_outputs=audit["residue_outputs_checked"], audit_fresh_lm_windows=audit["fresh_lm_windows"],
        source_sha256=digest(__file__), scope=data["scope"])
    (out/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    (out/"summarize_ball_ffn.py").write_bytes(Path(__file__).read_bytes())
    artifacts = sorted(out.iterdir())
    (out/"manifest.json").write_text(json.dumps(dict(artifact_sha256={p.name:digest(p) for p in artifacts},
        artifact_bytes={p.name:p.stat().st_size for p in artifacts}), indent=2, sort_keys=True)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
