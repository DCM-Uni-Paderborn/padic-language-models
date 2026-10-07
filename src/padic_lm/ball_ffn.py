"""Finite-ring affine features and a real readout for a small FFN pilot.

Only the affine/ball stage is p-adic (at finite precision). PCA, quantile
encoding, ridge fitting and all objectives/readouts are explicitly real.
"""
from __future__ import annotations

import numpy as np

from .arithmetic import modulus


def integer_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array")
    return a


def finite_affine(codes, weights, bias, p=2, k=4):
    """Exact residues with a proven pre-matmul int64 accumulation bound."""
    m = modulus(p, k)
    x, w, b = (integer_array(a, n) for a, n in
               ((codes, "codes"), (weights, "weights"), (bias, "bias")))
    if x.ndim != 2 or w.ndim != 2 or b.shape != (w.shape[0],) or x.shape[1] != w.shape[1]:
        raise ValueError("affine shapes must be N x m, r x m and r")
    if m > np.iinfo(np.int64).max or x.shape[1]*(m-1)**2 + m-1 > np.iinfo(np.int64).max:
        raise OverflowError("int64 affine accumulation is unsafe at this precision")
    x, w, b = ((a % m).astype(np.int64) for a in (x, w, b))
    return (x @ w.T + b) % m


def ball_features(residues, p=2, k=4):
    """Row-major indicators of the balls p^ell Z_p, ell=1,...,k."""
    modulus(p, k)
    a = integer_array(residues, "residues")
    if a.ndim != 2:
        raise ValueError("residues must be a matrix")
    depths = np.asarray([p**ell for ell in range(1, k+1)], dtype=np.int64)
    return (a[..., None] % depths == 0).reshape(len(a), -1).astype(np.float64)


def reversed_digits(p=2, k=4):
    """Permutation sending leading ordinary digits to low p-adic digits."""
    m = modulus(p, k)
    result = np.zeros(m, dtype=np.int64)
    values = np.arange(m, dtype=np.int64)
    for _ in range(k):
        result = result*p + values % p
        values //= p
    return result.astype(np.uint8 if m <= 256 else np.uint64)


def fit_encoder(hidden, dimensions=16, bins=16):
    """Training-only PCA, followed by training-calibrated quantile bins."""
    h = np.asarray(hidden, dtype=np.float64)
    center = h.mean(axis=0).astype(np.float32)
    centered = h-center.astype(np.float64)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(h))
    projection = vectors[:, -dimensions:][:, ::-1].copy()
    for j in range(dimensions):
        if projection[np.argmax(np.abs(projection[:, j])), j] < 0:
            projection[:, j] *= -1
    projection = projection.astype(np.float32)
    projected = centered @ projection.astype(np.float64)
    thresholds = np.quantile(projected, np.arange(1, bins)/bins, axis=0, method="linear").T
    mean = projected.mean(axis=0)
    scale = projected.std(axis=0)
    if np.any(scale <= 0) or not np.isfinite(projected).all():
        raise ValueError("degenerate real encoder")
    return dict(center=center, projection=projection, thresholds=thresholds,
                mean=mean, scale=scale)


def encode(hidden, encoder):
    projected = (np.asarray(hidden, dtype=np.float64)-encoder["center"].astype(np.float64)) @ encoder["projection"].astype(np.float64)
    # Equality belongs to the upper bin; fit and hard replay use this contract.
    ordered = np.column_stack([np.searchsorted(t, projected[:, j], side="right")
                               for j, t in enumerate(encoder["thresholds"])]).astype(np.uint8)
    real = (projected-encoder["mean"])/encoder["scale"]
    return ordered, real


def row_features(values, kind):
    if kind not in ("ball", "signed", "real"):
        raise ValueError("unknown feature kind")
    a = np.asarray(values)
    if kind == "signed":
        s = np.where(a >= 8, a-16, a).astype(np.float64)/8
        return np.stack([s**j for j in range(1, 5)], axis=1)
    if kind == "real":
        return (a[:, None] <= np.asarray([-1., -.25, .25, 1.])).astype(np.float64)
    return ball_features(a[:, None])


def row_values(x, w, b, kind, real_scale=1.):
    if kind == "real":
        sw = np.where(w >= 8, w.astype(np.float64)-16, w).astype(np.float64)/8
        sb = (float(b)-16 if b >= 8 else float(b))/8
        return (x @ sw + sb)/real_scale
    return (x @ w.astype(np.int64) + int(b)) % 16


def features(x, weights, bias, kind="ball", real_scale=None):
    if kind not in ("ball", "signed", "real"):
        raise ValueError("unknown feature kind")
    if kind != "real":
        values = finite_affine(x, weights, bias)
        if kind == "ball":
            return ball_features(values)
        return np.concatenate([row_features(values[:, j], kind) for j in range(values.shape[1])], axis=1)
    return np.concatenate([row_features(row_values(x, w, b, kind, real_scale[j]), kind)
                           for j, (w, b) in enumerate(zip(weights, bias))], axis=1)


def ridge(feature, target, alpha=.001):
    """Minimize mean squared residual + alpha * squared readout norm."""
    x, y = np.asarray(feature, dtype=np.float64), np.asarray(target, dtype=np.float64)
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc, yc = x-xm, y-ym
    gram = xc.T @ xc / len(x)
    coefficient = np.linalg.solve(gram+alpha*np.eye(x.shape[1]), xc.T @ yc/len(x))
    return coefficient, ym-xm @ coefficient


def discrete_fit(x, target, weights, bias, kind="ball", real_scale=None,
                 sweeps=2, seed=17, check_time=lambda: None):
    """Exact hard forwards; binary digit alternatives, fixed readout per sweep.

    The objective is real FP64 training SSE. Analytic SSE differences avoid
    materializing a 960-dimensional prediction for each finite alternative.
    A fresh ridge fit follows each complete sweep; no development input is used.
    """
    w, b = weights.copy(), bias.copy()
    rng = np.random.Generator(np.random.PCG64(seed+8047))
    phi = features(x, w, b, kind, real_scale)
    coefficient, intercept = ridge(phi, target)
    initial = dict(weights=w.copy(), bias=b.copy(), coefficient=coefficient.copy(), intercept=intercept.copy())
    residual = target-phi @ coefficient-intercept
    history = [dict(sweep=-1, row=-1, proposals=0, accepted=0, sse=float(np.sum(residual*residual)))]
    for sweep in range(sweeps):
        for row in rng.permutation(len(w)):
            check_time()
            row = int(row)
            base = phi[:, 4*row:4*row+4].copy()
            readout = coefficient[4*row:4*row+4]
            gram = readout @ readout.T
            projected_residual = residual @ readout.T
            row_sse = float(np.sum(residual*residual))
            tolerance = max(1e-10, row_sse*1e-12)
            current_delta, accepted = 0., 0
            row_phi = base
            for coordinate in rng.permutation((w.shape[1]+1)*4):
                j, digit = divmod(int(coordinate), 4)
                proposed_w, proposed_b = w[row].copy(), int(b[row])
                if j == w.shape[1]:
                    proposed_b ^= 1 << digit
                else:
                    proposed_w[j] ^= 1 << digit
                scale = 1. if real_scale is None else real_scale[row]
                proposed_phi = row_features(row_values(x, proposed_w, proposed_b, kind, scale), kind)
                d = proposed_phi-base
                delta = float(np.einsum("ni,ij,nj->", d, gram, d, optimize=False)-2*np.sum(d*projected_residual))
                if delta < current_delta-tolerance:
                    w[row], b[row] = proposed_w, proposed_b
                    current_delta, row_phi = delta, proposed_phi
                    accepted += 1
            residual -= (row_phi-base) @ readout
            phi[:, 4*row:4*row+4] = row_phi
            actual = float(np.sum(residual*residual))
            if abs((actual-row_sse)-current_delta) > max(1e-7, row_sse*1e-9):
                raise ArithmeticError("analytic hard-forward SSE update disagrees with full residual")
            history.append(dict(sweep=sweep, row=row, proposals=(w.shape[1]+1)*4,
                                accepted=accepted, sse=actual))
        coefficient, intercept = ridge(phi, target)
        residual = target-phi @ coefficient-intercept
        history.append(dict(sweep=sweep, row=-1, proposals=0, accepted=0,
                            sse=float(np.sum(residual*residual))))
    final = dict(weights=w, bias=b, coefficient=coefficient, intercept=intercept)
    return initial, final, history
