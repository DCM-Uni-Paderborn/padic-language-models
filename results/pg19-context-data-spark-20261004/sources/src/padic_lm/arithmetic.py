"""Exact arithmetic controls for finite p-adic precision experiments.

The computations here are over Z/(p**k), with real quantization scales outside
the ring. They simulate mathematical behavior using Python integers, not packed
low-bit storage or accelerated low-bit kernels. No p-adic metric is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isqrt
from numbers import Integral
from typing import Iterable

import numpy as np


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return int(value)


def _prime(p: int) -> int:
    p = _integer(p, "p")
    if p < 2 or any(p % d == 0 for d in range(2, isqrt(p) + 1)):
        raise ValueError("p must be prime")
    return p


def modulus(p: int, k: int) -> int:
    """Return p**k for prime p and positive finite precision k."""
    p = _prime(p)
    k = _integer(k, "k")
    if k < 1:
        raise ValueError("k must be positive")
    return p**k


def balanced_decode(residue: int, ring_modulus: int) -> int:
    """Decode into [-floor(M/2), ceil(M/2)-1].

    For even M the tie M/2 decodes to -M/2. For odd M the interval is
    symmetric. Any integer representative of the residue is accepted.
    """
    residue = _integer(residue, "residue")
    ring_modulus = _integer(ring_modulus, "ring_modulus")
    if ring_modulus < 2:
        raise ValueError("ring_modulus must be at least two")
    canonical = residue % ring_modulus
    return canonical - ring_modulus if 2 * canonical >= ring_modulus else canonical


def _paired_integers(
    weights: Iterable[int], activations: Iterable[int]
) -> list[tuple[int, int]]:
    w = list(weights)
    x = list(activations)
    if len(w) != len(x):
        raise ValueError("weights and activations must have equal lengths")
    return [(_integer(a, "weight"), _integer(b, "activation")) for a, b in zip(w, x)]


def exact_dot(weights: Iterable[int], activations: Iterable[int]) -> int:
    """Compute an integer dot product without NumPy fixed-width overflow."""
    return sum(a * b for a, b in _paired_integers(weights, activations))


def modular_dot(
    weights: Iterable[int], activations: Iterable[int], p: int, k: int
) -> int:
    """Compute a canonical dot-product residue in Z/(p**k).

    Reduction occurs after each addition. Inputs are promoted to arbitrary-
    precision Python integers before multiplication, including np.int64 inputs.
    """
    ring_modulus = modulus(p, k)
    accumulator = 0
    for a, b in _paired_integers(weights, activations):
        accumulator = (accumulator + (a % ring_modulus) * (b % ring_modulus)) % ring_modulus
    return accumulator


def dot_abs_bound(weights: Iterable[int], activations: Iterable[int]) -> int:
    """Return sum_i |w_i| |x_i|, a proven bound on |sum_i w_i x_i|."""
    return sum(abs(a) * abs(b) for a, b in _paired_integers(weights, activations))


def signed_recovery_exponent(p: int, abs_bound: int) -> int:
    """Smallest positive k with p**k > 2*abs_bound.

    If |s| <= abs_bound, then -M/2 < s < M/2 for M=p**k. Thus the balanced
    representative of s mod M is exactly s. The strict inequality handles the
    even-modulus endpoint, where +M/2 would decode to -M/2.
    """
    p = _prime(p)
    abs_bound = _integer(abs_bound, "abs_bound")
    if abs_bound < 0:
        raise ValueError("abs_bound must be nonnegative")
    k, ring_modulus = 1, p
    while ring_modulus <= 2 * abs_bound:
        k += 1
        ring_modulus *= p
    return k


def exponent_at_bit_budget(p: int, bit_budget: int) -> int:
    """Largest positive k satisfying log2(p**k) <= bit_budget.

    Integer comparisons avoid floating-point log errors at powers of two.
    This is a cap on residue-alphabet entropy, not a storage-packing claim.
    """
    p = _prime(p)
    bit_budget = _integer(bit_budget, "bit_budget")
    if bit_budget < 1:
        raise ValueError("bit_budget must be positive")
    cap = 1 << bit_budget
    k, ring_modulus = 0, 1
    while ring_modulus * p <= cap:
        k += 1
        ring_modulus *= p
    if not k:
        raise ValueError("bit budget cannot fit one base-p digit")
    return k


def twos_complement_dot(
    weights: Iterable[int], activations: Iterable[int], bits: int
) -> int:
    """Simulate signed fixed-width wrapping accumulation with Python ints.

    This independently uses a mask after every accumulation. It is exactly
    equivalent to balanced decoding over Z/(2**bits), including overflows.
    """
    bits = _integer(bits, "bits")
    if bits < 1:
        raise ValueError("bits must be positive")
    mask = (1 << bits) - 1
    accumulator = 0
    for a, b in _paired_integers(weights, activations):
        accumulator = (accumulator + a * b) & mask
    return accumulator - (1 << bits) if accumulator & (1 << (bits - 1)) else accumulator


@dataclass(frozen=True)
class QuantizedArray:
    values: np.ndarray
    scale: float
    bits: int

    def dequantize(self) -> np.ndarray:
        return self.values.astype(np.float64) * self.scale


def symmetric_quantize(values: Iterable[float] | np.ndarray, bits: int) -> QuantizedArray:
    """Per-array symmetric max-absolute quantization with ties-to-even rounding.

    A nominal b-bit code uses [-(2**(b-1)-1), +(2**(b-1)-1)]; the most negative
    two's-complement code is unused. Returned codes occupy int64 arrays. The
    scale is a float64 value outside the residue ring. All-zero arrays use 1.
    """
    bits = _integer(bits, "bits")
    if not 2 <= bits <= 16:
        raise ValueError("bits must be between 2 and 16")
    data = np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise ValueError("quantization inputs must be finite")
    qmax = (1 << (bits - 1)) - 1
    maximum = float(np.max(np.abs(data))) if data.size else 0.0
    scale = maximum / qmax if maximum else 1.0
    # A positive subnormal maximum can underflow when divided by qmax.
    if scale == 0.0:
        scale = maximum
    codes = np.clip(np.rint(data / scale), -qmax, qmax).astype(np.int64)
    return QuantizedArray(codes, scale, bits)
