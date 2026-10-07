"""Repair only the result export from complete immutable fitted arrays."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main():
    root = Path(sys.argv[1]); results = []
    assert not (root/"summary.json").exists(), "never overwrite a prior completed summary"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    for modulus in (4, 9):
        data = np.load(root/f"mod{modulus}-priors.npz")
        files = sorted(root.glob(f"mod{modulus}-*-logits.npz"))
        assert len(files) == 15, "all stock, real, native and quotient outputs required"
        for p in files:
            method = p.name[len(f"mod{modulus}-"):-len("-logits.npz")]
            preds = np.load(p)
            assert set(preds.files) == {"train", "validation", "same_range", "large_numbers", "new_wording"}
            for part in preds.files:
                z = preds[part]; y = data[part+"_y"]
                centered = z-z.max(1)[:, None]
                ll = np.log(np.exp(centered).sum(1))-centered[np.arange(len(y)), y]
                assert np.all(np.isfinite(z)) and np.all(np.isfinite(ll))
                results.append(dict(modulus=modulus, method=method, split=part,
                    accuracy=float(np.mean(z.argmax(1) == y)), nll=float(ll.mean())))
    summary = dict(completed_export_utc=datetime.now(timezone.utc).isoformat(), results=results,
        primary_endpoint="constrained single digit conditional accuracy/NLL; not corpus token NLL",
        all_fits_completed=True, training_repeated=False,
        completion_note="Producer completed every fit and saved all raw logits, then failed JSON serialization on an underflowed real-raw OOD loss. Stable logsumexp exports the same immutable logits.",
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/"summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n")
    assert all(hashlib.sha256((root/k).read_bytes()).hexdigest() == v for k, v in before.items())
    (root/"export-repair.json").write_text(json.dumps(dict(original_sha256=before, output_sha256=hashlib.sha256((root/"summary.json").read_bytes()).hexdigest(), original_files_unchanged=True), indent=2, sort_keys=True)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
