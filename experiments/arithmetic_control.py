"""Reproducible synthetic dot-product controls; this is not an LLM benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from padic_lm.arithmetic import (
    balanced_decode,
    dot_abs_bound,
    exact_dot,
    exponent_at_bit_budget,
    modular_dot,
    signed_recovery_exponent,
    symmetric_quantize,
    twos_complement_dot,
)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def normalized_rms(error: np.ndarray, reference: np.ndarray) -> float:
    denominator = rms(reference)
    numerator = rms(error)
    if not denominator:
        if numerator:
            raise ValueError("nonzero error with an all-zero reference")
        return 0.0
    return numerator / denominator


def plot_results(records: list[dict], baselines: list[dict], output: Path) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"created": False, "reason": str(exc)}

    dimensions = sorted({r["dimension"] for r in records})
    primes = sorted({r["p"] for r in records})
    budgets = sorted({r["bit_budget"] for r in records})
    fig, axes = plt.subplots(2, len(dimensions), figsize=(14, 7), sharex=True, sharey="row")
    colors = {2: "#255c99", 3: "#d66c18", 5: "#16805c"}
    for col, dimension in enumerate(dimensions):
        for p in primes:
            groups = [[r for r in records if r["dimension"] == dimension and r["p"] == p and r["bit_budget"] == b] for b in budgets]
            for row, key in enumerate(("wrap_fraction", "integer_reference_nrmse")):
                values = np.array([[r[key] for r in group] for group in groups])
                axes[row, col].plot(budgets, values.mean(axis=1), "o-", color=colors[p], label=f"p={p}")
                axes[row, col].fill_between(budgets, values.min(axis=1), values.max(axis=1), color=colors[p], alpha=0.12)
        axes[0, col].set_title(f"Dot length {dimension}")
        axes[1, col].set_xlabel("Residue entropy cap (bits)")
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xticks(budgets)
        axes[0, col].set_ylim(-0.04, 1.05)
        axes[1, col].set_ylim(-0.04, 1.05 * max(r["integer_reference_nrmse"] for r in records))
    axes[0, 0].set_ylabel("Incorrect signed recovery (fraction)", fontsize=10)
    axes[1, 0].set_ylabel("NRMSE vs exact quantized dot", fontsize=10)
    axes[0, -1].legend(loc="best")
    fig.suptitle("Finite residue arithmetic controls: Gaussian vectors, W4/A8\nLines: mean over 3 seeds; shading: seed range. Synthetic arithmetic only.")
    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    fig.text(0.5, 0.015, "p=2 is exactly fixed-width wrapping. Odd primes use the largest p^k under each common bit cap; effective entropies differ.", ha="center", fontsize=9)
    fig.savefig(output / "recovery-curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in sorted({b["seed"] for b in baselines}):
        rows = sorted([b for b in baselines if b["seed"] == seed], key=lambda b: b["dimension"])
        ax.plot([b["dimension"] for b in rows], [b["quantization_nrmse_vs_real"] for b in rows], "o-", label=f"Seed {seed}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dot length")
    ax.set_ylabel("W4/A8 NRMSE against unquantized real dot")
    ax.set_title("Quantization baseline, before any modular wrapping")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "quantization-baseline.png", dpi=180)
    plt.close(fig)
    return {"created": True, "matplotlib_version": matplotlib.__version__, "files": ["recovery-curves.png", "quantization-baseline.png"]}


def run(output: Path, samples: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    seeds = [17, 29, 43]
    dimensions = [16, 64, 256, 1024]
    primes = [2, 3, 5]
    bit_budgets = [8, 12, 16, 20, 24, 28]
    records, baselines, arrays = [], [], {}
    twos_checks = modular_checks = recovery_checks = 0

    for seed in seeds:
        for dimension in dimensions:
            # Each condition has its own deterministically derived RNG; order
            # changes do not change its samples.
            rng = np.random.default_rng(np.random.SeedSequence([seed, dimension]))
            weights = rng.standard_normal((samples, dimension)) / math.sqrt(dimension)
            activations = rng.standard_normal((samples, dimension))
            qw = [symmetric_quantize(row, 4) for row in weights]
            qx = [symmetric_quantize(row, 8) for row in activations]
            scale_products = np.array([w.scale * x.scale for w, x in zip(qw, qx)])
            integer_dots = np.array([exact_dot(w.values, x.values) for w, x in zip(qw, qx)], dtype=np.int64)
            real_dots = np.einsum("ij,ij->i", weights, activations)
            quantized_dots = integer_dots.astype(np.float64) * scale_products
            abs_bounds = [dot_abs_bound(w.values, x.values) for w, x in zip(qw, qx)]
            worst_case_bound = dimension * 7 * 127
            baselines.append({
                "seed": seed, "dimension": dimension, "samples": samples,
                "quantization_nrmse_vs_real": normalized_rms(quantized_dots - real_dots, real_dots),
                "real_dot_rms": rms(real_dots), "quantized_dot_rms": rms(quantized_dots),
                "max_abs_integer_dot": int(np.max(np.abs(integer_dots))),
                "max_sample_triangle_bound": max(abs_bounds),
                "worst_case_code_bound": worst_case_bound,
            })
            prefix = f"seed{seed}_d{dimension}"
            arrays[f"{prefix}_integer_dots"] = integer_dots
            arrays[f"{prefix}_scale_products"] = scale_products
            arrays[f"{prefix}_real_dots"] = real_dots
            arrays[f"{prefix}_weight_codes"] = np.stack([w.values for w in qw])
            arrays[f"{prefix}_activation_codes"] = np.stack([x.values for x in qx])

            for p in primes:
                guaranteed_k = signed_recovery_exponent(p, worst_case_bound)
                guaranteed_m = p**guaranteed_k
                guaranteed = [balanced_decode(modular_dot(w.values, x.values, p, guaranteed_k), guaranteed_m) for w, x in zip(qw, qx)]
                if guaranteed != integer_dots.tolist():
                    raise AssertionError("proven sufficient signed-recovery bound failed")
                recovery_checks += samples
                for bit_budget in bit_budgets:
                    k = exponent_at_bit_budget(p, bit_budget)
                    ring_modulus = p**k
                    residues = [modular_dot(w.values, x.values, p, k) for w, x in zip(qw, qx)]
                    decoded = np.array([balanced_decode(r, ring_modulus) for r in residues], dtype=np.int64)
                    expected_residues = [int(s) % ring_modulus for s in integer_dots]
                    if residues != expected_residues:
                        raise AssertionError("modular dot did not equal exact dot modulo M")
                    modular_checks += samples
                    if p == 2:
                        wrapping = [twos_complement_dot(w.values, x.values, bit_budget) for w, x in zip(qw, qx)]
                        if wrapping != decoded.tolist():
                            raise AssertionError("Z/(2^k) and two's-complement wrapping differed")
                        twos_checks += samples
                    error_codes = decoded - integer_dots
                    if np.any(error_codes % ring_modulus):
                        raise AssertionError("signed decode error was not an integer multiple of M")
                    recovered_dots = decoded.astype(np.float64) * scale_products
                    wrap_counts = (integer_dots - decoded) // ring_modulus
                    records.append({
                        "seed": seed, "dimension": dimension, "samples": samples,
                        "p": p, "k": k, "bit_budget": bit_budget, "modulus": ring_modulus,
                        "effective_entropy_bits": math.log2(ring_modulus),
                        "guaranteed_recovery_k": guaranteed_k,
                        "guaranteed_recovery_modulus": guaranteed_m,
                        "guaranteed_recovery_entropy_bits": math.log2(guaranteed_m),
                        "max_abs_integer_dot": int(np.max(np.abs(integer_dots))),
                        "wrap_fraction": float(np.mean(decoded != integer_dots)),
                        "wrapped_dots": int(np.sum(decoded != integer_dots)),
                        "max_abs_wrap_count": int(np.max(np.abs(wrap_counts))),
                        "integer_reference_nrmse": normalized_rms(recovered_dots - quantized_dots, quantized_dots),
                        "real_reference_nrmse": normalized_rms(recovered_dots - real_dots, real_dots),
                        "max_abs_integer_error": int(np.max(np.abs(error_codes))),
                        "modular_matches_exact_reduction": True,
                        "twos_complement_matches": True if p == 2 else None,
                    })
                    arrays[f"{prefix}_p{p}_b{bit_budget}_decoded"] = decoded

    sources = [PROJECT / "src/padic_lm/arithmetic.py", Path(__file__).resolve(), PROJECT / "tests/test_arithmetic.py"]
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT / "tests"), "-p", "test_arithmetic.py", "-v"]
    tests = subprocess.run(test_command, cwd=PROJECT, capture_output=True, text=True)
    if tests.returncode:
        raise AssertionError(f"arithmetic tests failed:\n{tests.stdout}\n{tests.stderr}")
    report = {
        "experiment": "finite-residue-arithmetic-control",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic Gaussian dot products only; no language model, no packed storage, no speed claim, no p-adic metric.",
        "configuration": {
            "seeds": seeds, "dimensions": dimensions, "primes": primes,
            "bit_budgets": bit_budgets, "samples_per_seed_dimension": samples,
            "weight_bits": 4, "weight_codes": [-7, 7], "activation_bits": 8,
            "activation_codes": [-127, 127], "quantizer": "per-vector symmetric max-absolute, ties-to-even, scales outside ring",
            "weight_distribution": "iid N(0,1/dimension)", "activation_distribution": "iid N(0,1)",
            "rng": "NumPy default_rng with SeedSequence([seed,dimension])",
            "matching": "Common entropy cap B; choose largest k with p**k <= 2**B. Effective entropy varies by prime.",
            "balanced_interval": "[-floor(M/2),ceil(M/2)-1]; even endpoint M/2 decodes negative",
            "sufficient_recovery_bound": "M > 2*d*7*127 guarantees every possible W4/A8 dot of length d",
            "normalization": "RMS error / RMS reference over samples within a seed/dimension; scales included",
        },
        "environment": {
            "python_version": sys.version, "python_executable": sys.executable,
            "numpy_version": np.__version__, "platform": platform.platform(),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "working_directory": str(Path.cwd()),
            "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        },
        "validation": {
            "modular_dot_exact_reduction_checks": modular_checks,
            "twos_complement_equivalence_checks": twos_checks,
            "sufficient_bound_recovery_checks": recovery_checks,
            "unit_tests": {"command": test_command, "returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
        },
        "baselines": baselines,
        "records": records,
    }
    report["plots"] = plot_results(records, baselines, output)
    with (output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output / "baselines.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0]))
        writer.writeheader()
        writer.writerows(baselines)
    np.savez_compressed(output / "raw-dots.npz", **arrays)
    artifact_names = ["records.csv", "baselines.csv", "raw-dots.npz", *report["plots"].get("files", [])]
    report["artifact_sha256"] = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifact_names}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/arithmetic-control")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    report = run(args.output.resolve(), args.samples)
    baseline_errors = [b["quantization_nrmse_vs_real"] for b in report["baselines"]]
    print(json.dumps({
        "output": str(args.output.resolve()), "conditions": len(report["records"]),
        "baseline_quantization_nrmse_range": [min(baseline_errors), max(baseline_errors)],
        "validation": {k: v for k, v in report["validation"].items() if k != "unit_tests"},
        "tests_passed": report["validation"]["unit_tests"]["returncode"] == 0,
        "plots": report["plots"],
    }, indent=2))


if __name__ == "__main__":
    main()
