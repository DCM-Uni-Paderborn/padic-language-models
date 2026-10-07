"""Small independent disk-path learner; not upstream training code.

Implements affine Gauss radii and one-sided slopes in the p-adic hull.
Centers use exact integers over p**2; radii and real losses use float64.
Joint incident directions are enumerated within one affine output group.
"""
from itertools import product
import math
import numpy as np


def norm_int(a, p):
    a = np.abs(np.asarray(a, dtype=np.int64))
    zero = a == 0
    work = np.where(zero, 1, a).copy()
    v = np.zeros(a.shape, dtype=np.int64)
    while np.any(work % p == 0):
        div = work % p == 0
        work = np.where(div, work // p, work)
        v += div
    return np.where(zero, 0., np.power(float(p), -v))


def softmax(z):
    q = np.exp(z - np.max(z, axis=1, keepdims=True))
    return q / q.sum(axis=1, keepdims=True)


def metrics(z, y):
    q = softmax(z)
    centered = z-np.max(z, axis=1, keepdims=True)
    nll = np.log(np.exp(centered).sum(axis=1))-centered[np.arange(len(y)), y]
    return dict(accuracy=float(np.mean(np.argmax(q, axis=1) == y)),
                nll=float(nll.mean()))


class AffineDisk:
    def __init__(self, p, dimensions, radius=None, depth=6):
        if p not in (2, 3):
            raise ValueError("this bounded implementation supports p=2 or3")
        self.p, self.denominator, self.depth = p, p**2, depth
        self.c = np.zeros(dimensions, dtype=np.int64)
        self.r = np.full(dimensions, float(p**2 if radius is None else radius))

    def state(self):
        return dict(p=self.p, denominator=self.denominator, depth=self.depth,
                    centers_numerator=self.c.tolist(), radii=self.r.tolist())

    def forward(self, x):
        terms = norm_int(x, self.p) * self.r
        radius = terms.max(axis=1)
        center_norm = self.denominator * norm_int(x @ self.c, self.p)
        score = np.maximum(radius, center_norm)
        return score, radius, center_norm

    def candidates(self):
        options = []
        for c, r in zip(self.c, self.r):
            log = math.log(r, self.p)
            vertex = abs(log-round(log)) < 1e-11
            opts = [(0, int(c), float("inf"))]
            if r < self.p**2*(1-1e-12):
                upper = r*self.p if vertex else float(self.p**math.ceil(log))
                opts.append((1, int(c), upper-r))
            if r > self.p**(-self.depth)*(1+1e-12):
                lower = r/self.p if vertex else float(self.p**math.floor(log))
                if vertex:
                    n = -int(round(log))
                    unit = self.p**(n+2)
                    for digit in range(self.p):
                        opts.append((-1, int(c % unit + digit*unit), r-lower))
                else:
                    opts.append((-1, int(c), r-lower))
            options.append(opts)
        combos = [cc for cc in product(*options) if any(t[0] for t in cc)]
        if not combos:
            return np.empty((0, len(self.c))), np.empty((0, len(self.c)), dtype=np.int64), np.empty(0)
        dr = np.array([[t[0] for t in cc] for cc in combos], dtype=np.float64)
        centers = np.array([[t[1] for t in cc] for cc in combos], dtype=np.int64)
        caps = np.array([min(t[2] for t in cc) for cc in combos])
        return dr, centers, caps

    def slopes(self, x, kind, *, y=None, derivative=None):
        dr, centers, caps = self.candidates()
        if len(dr) == 0:
            return np.empty(0), dr, centers, caps
        nx = norm_int(x, self.p)
        terms = nx*self.r
        radius = terms.max(axis=1)
        active = np.isclose(terms, radius[:, None], rtol=1e-12, atol=0.)
        # Max, rather than sum, of rates at tied active monomials.
        rate = np.max(np.where(active[None, :, :], dr[:, None, :]*nx[None, :, :], -np.inf), axis=2)
        raw = centers @ x.T
        if kind == "direct":
            raw -= np.asarray(y)[None, :]*self.denominator
        cn = self.denominator*norm_int(raw, self.p)
        equal = np.isclose(cn, radius[None, :], rtol=1e-12, atol=0.)
        ds = np.where(equal, np.maximum(rate, 0.), np.where(cn < radius[None, :], rate, 0.))
        if kind == "direct":
            slopes = np.mean(ds-.5*rate, axis=1)
        elif kind == "classification":
            score = np.maximum(cn, radius[None, :])
            slopes = np.mean(np.asarray(derivative)[None, :]*(-ds/score), axis=1)
        else:
            raise ValueError(kind)
        return slopes, dr, centers, caps

    def update(self, x, kind, rng, alpha, loss, *, y=None, derivative=None):
        slopes, dr, centers, caps = self.slopes(x, kind, y=y, derivative=derivative)
        if len(slopes) == 0 or slopes.min() >= -1e-12:
            return dict(moved=False, slope=float(slopes.min()) if len(slopes) else 0., step=0.)
        best = np.flatnonzero(np.isclose(slopes, slopes.min(), rtol=1e-11, atol=1e-12))
        i = int(rng.choice(best))
        old_c, old_r = self.c.copy(), self.r.copy()
        before = float(loss())
        step = min(float(alpha*(-slopes[i])), float(caps[i]))
        for backtrack in range(20):
            self.c = centers[i].copy()
            self.r = old_r+step*dr[i]
            for j, r in enumerate(self.r):
                log = math.log(r, self.p)
                if abs(log-round(log)) < 1e-11:
                    n = -int(round(log))
                    self.r[j] = float(self.p**(-n))
                    self.c[j] %= self.p**(n+2)
            after = float(loss())
            if after <= before-1e-14:
                return dict(moved=True, slope=float(slopes[i]), step=step,
                            before=before, after=after, backtracks=backtrack,
                            direction=dr[i].astype(int).tolist(), state=self.state())
            step /= 2
        self.c, self.r = old_c, old_r
        return dict(moved=False, slope=float(slopes[i]), step=0., backtracks=20)


def direct_loss(model, x, y):
    _, radius, _ = model.forward(x)
    error = model.denominator*norm_int(x@model.c-y*model.denominator, model.p)
    return float(np.mean(np.maximum(error, radius)-radius/2))


def quotient_logits(x, states):
    """Independent ordinary congruence-depth evaluation, without norm_int.

    This is an equivalent finite quotient-tree control, not a second fit.
    """
    rows = []
    for state in states:
        p, den = state["p"], state["denominator"]
        cs, rs = state["centers_numerator"], state["radii"]
        for_row = []
        for inputs in x.tolist():
            out = sum(int(a)*int(b) for a, b in zip(inputs, cs))
            if out == 0:
                center_score = 0.
            else:
                k = 0
                while out % (p**(k+1)) == 0:
                    k += 1
                center_score = den/float(p**k)
            radius_terms = []
            for a, r in zip(inputs, rs):
                if a == 0:
                    radius_terms.append(0.)
                else:
                    k = 0
                    while int(a) % (p**(k+1)) == 0:
                        k += 1
                    radius_terms.append(r/float(p**k))
            for_row.append(-math.log(max(center_score, max(radius_terms))))
        rows.append(for_row)
    return np.array(rows).T
