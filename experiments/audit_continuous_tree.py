"""Fraction-based audit, with no imports from the producer or learner."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np


def magnitude(q, p):
    q = Fraction(q)
    if not q:
        return 0.
    n, d, v = abs(q.numerator), q.denominator, 0
    while n % p == 0:
        n //= p; v += 1
    while d % p == 0:
        d //= p; v -= 1
    return float(Fraction(p)**(-v))


def loss(state, x, y):
    p, den = state["p"], state["denominator"]
    cc = [Fraction(c, den) for c in state["centers_numerator"]]
    errors, losses = [], []
    for row, target in zip(x, y):
        pred = sum(int(a)*c for a, c in zip(row, cc))
        err = magnitude(pred-int(target), p)
        radius = max(magnitude(int(a), p)*r for a, r in zip(row, state["radii"]))
        errors.append(err); losses.append(max(radius, err)-radius/2)
    return float(np.mean(losses)), max(errors)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("directory", type=Path)
    args = ap.parse_args(); root = args.directory
    summary = json.loads((root/"summary.json").read_text())
    verified, checked = [], 0
    fig = json.loads((root/"figure-trajectory.json").read_text())
    for row in fig:
        measured, _ = loss(row["state"], [[1]], [23])
        assert abs(measured-row["loss"]) < 1e-12
        checked += 1
    assert fig[-1]["state"]["centers_numerator"] == [207]
    assert fig[-1]["state"]["radii"][0] <= 3**-6*(1+1e-12)
    for fit in summary["fits"]:
        seed = fit["seed"]
        data = np.load(root/f"regression-{seed}-data.npz")
        rng = np.random.default_rng(seed)
        theta = rng.integers(0, 243, size=3, dtype=np.int64)
        np.testing.assert_array_equal(theta, data["theta"])
        for part in ("train", "validation", "test"):
            inputs = np.column_stack((rng.integers(0, 81, size=(512, 2), dtype=np.int64), np.ones(512, dtype=np.int64)))
            targets = inputs@theta+243*rng.integers(0, 27, size=512, dtype=np.int64)
            np.testing.assert_array_equal(inputs, data[part+"_x"])
            np.testing.assert_array_equal(targets, data[part+"_y"])
            ll, ee = loss(fit["state"], inputs, targets)
            assert abs(ll-fit["metrics"][part]["direct_loss"]) < 1e-12
            # Fraction conversion and NumPy negative powers differ by one ULP.
            assert abs(ee-fit["metrics"][part]["max_point_error"]) < 1e-14
        trace = json.loads((root/f"regression-{seed}-trajectory.json").read_text())
        for row in trace:
            ll, _ = loss(row["state"], data["train_x"], data["train_y"])
            assert abs(ll-row["loss"]) < 1e-12
            checked += 512
        assert all(trace[j+1]["loss"] <= trace[j]["loss"]+1e-12 for j in range(len(trace)-1))
        den = fit["state"]["denominator"]
        assert all((int(c)-int(t)*den) % (243*den) == 0 for c, t in zip(fit["state"]["centers_numerator"], theta))
        assert fit["accepted_updates"] > 0 and fit["metrics"]["test"]["max_point_error"] <= 1/243*(1+1e-12)
        verified.append(seed)
    assert summary["producer_gate"]
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file() and p.name != "audit.json"}
    result = dict(passed=True, independent_fraction_loss_elements=checked, seeds=verified,
                  limitations="No convergence theorem or full upstream result reproduced; joint group enumeration differs from upstream coordinate queue.", sha256=pins)
    (root/"audit.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
