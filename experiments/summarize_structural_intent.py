"""Export every fixed structural-probe outcome after its independent audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if a.output.exists(): raise FileExistsError("fresh summary path required")
    completion = json.loads((a.input/"completed.json").read_text())
    for name, digest in completion["artifact_sha256"].items():
        assert sha(a.input/name) == digest and (a.input/name).stat().st_size == completion["artifact_bytes"][name]
    audit = json.loads((a.audit/"audit.json").read_text())
    assert audit["status"] == "structural_intent_independently_verified"
    assert audit["producer_completion_sha256"] == sha(a.input/"completed.json")
    records = json.loads((a.input/"results.json").read_text())
    comparisons = json.loads((a.audit/"comparisons.json").read_text())
    progression = json.loads((a.audit/"progression.json").read_text())
    assert len(records) == 12 and len(comparisons) == 9 and len(progression) == 3
    a.output.mkdir(parents=True)
    rows = []
    for r in records:
        for split, values in r["splits"].items():
            rows.append(dict(seed=r["seed"], method=r["method"], split=split,
                learned_parameters=r["learned_parameter_count"], **values))
    fields = ["seed", "method", "split", "learned_parameters", "intent_accuracy", "intent_nll", "domain_accuracy", "domain_nll", "probability_sum_max_error"]
    with (a.output/"all-metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n");w.writeheader();w.writerows(rows)
    with (a.output/"all-comparisons.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comparisons[0]), lineterminator="\n");w.writeheader();w.writerows(comparisons)
    names = {"semantic_prefix":"Semantic prefix", "shuffled_prefix":"Shuffled prefix", "flat_joint":"Flat joint", "direct_domain":"Direct domain"}
    tex = []
    for r in records:
        seen, unseen = r["splits"]["seen_test"], r["splits"]["unseen_test"]
        fine = f"{100*seen['intent_accuracy']:.2f} & {seen['intent_nll']:.5f}" if "intent_accuracy" in seen else "--- & ---"
        tex.append(f"{names[r['method']]} & {r['seed']} & {fine} & {100*unseen['domain_accuracy']:.2f} & {unseen['domain_nll']:.5f}\\\\")
    (a.output/"model-rows.tex").write_text("\n".join(tex)+"\n")
    tex = []
    for r in comparisons:
        tex.append(f"{r['seed']} & {names[r['comparator']]} & {100*r['domain_accuracy_difference']:+.2f} & {r['domain_nll_difference']:+.5f} & {r['domain_nll_difference_95pct_upper']:+.5f}\\\\")
    (a.output/"comparison-rows.tex").write_text("\n".join(tex)+"\n")
    effect = []
    for seed in (17, 29, 43):
        data = {r["method"]:r for r in records if r["seed"] == seed}
        effect.append(dict(seed=seed, semantic_minus_shuffled_seen_intent_accuracy_pp=
            100*(data["semantic_prefix"]["splits"]["seen_test"]["intent_accuracy"]-data["shuffled_prefix"]["splits"]["seen_test"]["intent_accuracy"])))
    summary = dict(status="all_fixed_structural_probe_results_exported", producer_sha256=sha(a.input/"completed.json"),
        audit_sha256=sha(a.audit/"audit.json"), records=12, split_rows=48, comparison_rows=9,
        primary_progression=progression, descriptive_semantic_topology_effect=effect,
        scope="Known-intent semantic versus shuffled gains are descriptive; all primary unseen-transfer progression rules fail. No unique p-adic advantage.")
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (a.output/"summarize_structural_intent.py").write_bytes(Path(__file__).read_bytes())
    files = sorted(p for p in a.output.iterdir() if p.is_file())
    manifest = dict(artifact_sha256={p.name:sha(p) for p in files}, artifact_bytes={p.name:p.stat().st_size for p in files})
    (a.output/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
