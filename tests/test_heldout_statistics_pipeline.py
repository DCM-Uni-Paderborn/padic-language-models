"""Synthetic black-box checks; fixture statuses are not scientific evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_heldout_quality import METHODS


def manifest(directory):
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.name not in ("completed.json", "completed.sha256"))
    completion = {"status": "heldout_quality_evaluation_complete",
                  "artifact_sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  "artifact_bytes": {str(p.relative_to(directory)): p.stat().st_size for p in paths}}
    (directory / "completed.json").write_text(json.dumps(completion))
    value = hashlib.sha256((directory / "completed.json").read_bytes()).hexdigest()
    (directory / "completed.sha256").write_text(value + "\n")
    return value


class PipelineTests(unittest.TestCase):
    def test_black_box_synthetic_analysis_and_independent_audit(self):
        with tempfile.TemporaryDirectory(prefix="heldout-statistics-fixture-") as temporary:
            base = Path(temporary)
            evaluation, statistics, verified = [base / name for name in ("evaluation", "statistics", "verification")]
            (evaluation / "windows").mkdir(parents=True)
            protocol = ROOT / "research" / "heldout-quality-protocol.md"
            config = {"articles": [{"article_id": "fixture-one"}, {"article_id": "fixture-two"}],
                      "methods": list(METHODS[1:]), "seeds": [17, 29, 43], "lengths": [2048], "layer": 0,
                      "group_budget": 128,
                      "protocol_sha256": "9a0095eb8e079773e63d259c292a4d7ad10b932ded90df58026bbf90be2048d0",
                      "development_completion_sha256": "ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8",
                      "development_audit_sha256": "28b682bdf864333c6161606cfc4e6c67260551cab60fe5eb6f162676a6439a36"}
            stock = np.linspace(1, 2, 2047, dtype=np.float32)
            for i in range(2):
                arrays = {"reference_target_nll": stock}
                for m, method in enumerate(METHODS[1:]):
                    loss = stock.copy()
                    if method != "full_control":
                        loss[128:] += np.float32((i + 1) * (m + 1) / 2000)
                    arrays[method + "_target_nll"] = loss
                np.savez_compressed(evaluation / "windows" / f"length-2048-article-{i:02d}.npz", **arrays)
            (evaluation / "results.json").write_text(json.dumps({"status": "heldout_quality_evaluation_complete",
                "config": config, "completed_windows": 2, "original_model_state_unchanged": True,
                "new_fitting_or_tuning": False, "test_split_read": True}))
            parent = manifest(evaluation)
            audit_path = base / "synthetic-numerical-audit.json"
            audit_path.write_text(json.dumps({"status": "heldout_quality_audit_passed", "evaluation_completion_sha256": parent,
                "windows_verified": 2, "full_budget_target_loss_identity_windows": 2,
                "complete_model_fixture_readouts_including_references": 30, "cumulative_gate_seconds": 1.}))
            analyze = [sys.executable, str(ROOT / "experiments" / "analyze_heldout_quality.py"),
                       "--input", str(evaluation), "--numerical-audit", str(audit_path),
                       "--protocol", str(protocol), "--output", str(statistics)]
            checked = subprocess.run(analyze, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            data = json.loads((statistics / "results.json").read_text())
            self.assertEqual(len(data["comparisons"]), 106)
            self.assertEqual(data["targets_by_endpoint"], {"all": 4094, "affected": 3838})
            self.assertFalse(any(r["passes_frozen_screen"] for r in data["proposed_family_decisions"].values()))
            command = [sys.executable, str(ROOT / "experiments" / "audit_heldout_statistics.py"),
                       "--evaluation", str(evaluation), "--numerical-audit", str(audit_path),
                       "--statistics", str(statistics), "--protocol", str(protocol), "--output", str(verified)]
            checked = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            independently_checked = json.loads((verified / "audit.json").read_text())
            self.assertEqual(independently_checked["resampling_indices_exact"], 40000)
            self.assertEqual(independently_checked["comparison_intervals_verified"], 106)
            self.assertFalse(any(r["passes_frozen_screen"] for r in independently_checked["verified_family_decisions"].values()))
            # A modified published bound, even with a refreshed artifact hash, must fail.
            data["comparisons"][0]["upper_975"] += .001
            (statistics / "results.json").write_text(json.dumps(data))
            completion_path = statistics / "completed.json"
            completion = json.loads(completion_path.read_text())
            changed_path = statistics / "results.json"
            completion["artifact_sha256"]["results.json"] = hashlib.sha256(changed_path.read_bytes()).hexdigest()
            completion["artifact_bytes"]["results.json"] = changed_path.stat().st_size
            completion_path.write_text(json.dumps(completion))
            (statistics / "completed.sha256").write_text(hashlib.sha256(completion_path.read_bytes()).hexdigest() + "\n")
            command[-1] = str(base / "corrupt-bound-verification")
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertTrue((base / "corrupt-bound-verification" / "failed.json").exists())


if __name__ == "__main__":
    unittest.main()
