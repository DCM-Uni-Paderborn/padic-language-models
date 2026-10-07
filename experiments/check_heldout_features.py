"""Verify instrumented test features against frozen development algorithms and arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import evaluate_shared_budget as original
import evaluate_heldout_quality as current


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("development", "training", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh feature-preflight output required")
    assert sha(args.development / "completed.json") == "ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8"
    original.verify_manifest(args.training, original.TRAIN_HASH)
    completion = json.loads((args.development / "completed.json").read_text())
    filename = "windows/length-2048-article-00.npz"
    assert sha(args.development / filename) == completion["artifact_sha256"][filename]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(args.development / filename, allow_pickle=False) as window:
        values = {}
        for role, field in (("q", "queries"), ("k", "keys")):
            tensor = torch.tensor(window[f"reference_{field}_bits"], dtype=torch.uint16,
                                  device="cuda").view(torch.bfloat16)[None]
            if role == "k":
                tensor = tensor.repeat_interleave(3, dim=1)
            restored = torch.empty_strided(tensor.shape,
                       tuple(int(v) for v in window[f"reference_{role}_fp32_stride"]),
                       dtype=torch.float32, device="cuda")
            restored.copy_(tensor.float())
            values[role] = restored
        frozen = original.freeze_books(args.training)
        with torch.inference_mode():
            a, saved_a = original.feature_states(values["q"], values["k"], frozen)
            b, saved_b, timing = current.feature_states(values["q"], values["k"], frozen)
        assert saved_a.keys() == saved_b.keys()
        for name in saved_a:
            assert np.array_equal(saved_a[name], saved_b[name]), name
            assert np.array_equal(saved_b[name], window[name]), name
        score_checks = 0
        for seed in (17, 29, 43):
            for kind in original.KINDS:
                old = original.score_provider(a[seed], kind)
                new = current.score_provider(b[seed], kind)
                for pos in (128, 511, 1023, 2047):
                    stop = pos + 1 - (64 if kind == "gaussian_local64" else 8)
                    assert np.array_equal(old(pos, stop), new(pos, stop)), (seed, kind, pos)
                    score_checks += 1
            assert all(np.isfinite(v) and v >= 0 for v in timing[seed].values())
    report = {"status": "heldout_feature_preflight_passed", "development_completion_sha256": sha(args.development / "completed.json"),
              "feature_arrays_exact_against_both_original_and_archive": len(saved_a),
              "score_rows_exact_against_original": score_checks, "test_driver_sha256": sha(Path(current.__file__)),
              "verification_source_sha256": sha(__file__), "wall_seconds": time.monotonic() - started,
              "scope": "One archived2048-token validation teacher; no test model output or tuning; same features and scoring, added timing only"}
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
