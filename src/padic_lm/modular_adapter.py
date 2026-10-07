"""Typed-operand adapters for constrained next-digit LLM answers."""
import numpy as np
from padic_lm.continuous_tree import softmax


def features(x, modulus, kind, location=None, scale=None):
    a = np.asarray(x[:, :2], dtype=np.float64)
    if kind == "raw":
        if location is None:
            location = a.mean(0); scale = a.std(0)+1e-9
        return np.column_stack(((a-location)/scale, np.ones(len(a)))), location, scale
    if kind == "fourier":
        angles = (x[:, :2] % modulus)*(2*np.pi/modulus)
        ca, cb = np.cos(angles[:, 0]), np.cos(angles[:, 1])
        sa, sb = np.sin(angles[:, 0]), np.sin(angles[:, 1])
        return np.column_stack((ca*cb, ca*sb, sa*cb, sa*sb, np.ones(len(a)))), None, None
    raise ValueError(kind)


def fit_real(f, base, y, steps=300):
    w = np.zeros((f.shape[1], base.shape[1]), dtype=np.float64)
    first = np.zeros_like(w); second = np.zeros_like(w)
    history = []
    for t in range(1, steps+1):
        q = softmax(base+f@w)
        g = q.copy(); g[np.arange(len(y)), y] -= 1
        grad = f.T@g/len(y)
        first = .9*first+.1*grad; second = .999*second+.001*grad*grad
        w -= .03*(first/(1-.9**t))/(np.sqrt(second/(1-.999**t))+1e-8)
        history.append(float(-np.log(softmax(base+f@w)[np.arange(len(y)), y]).mean()))
    return w, history
