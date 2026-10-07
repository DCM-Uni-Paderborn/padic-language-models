"""Finite p-adic ball routing and independently implemented tree controls.

These routines are NumPy diagnostics, not sublinear indexes or optimized kernels.
The finite product ultrametric is exactly equivalent to a radix prefix tree.
Real PCA/quantile encoding is an explicit bridge, fitted on calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt
from numbers import Integral

import numpy as np


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return value


def _digits(value: int) -> int:
    value = _positive_integer(value, "digits")
    if value > 16:
        raise ValueError("diagnostic supports at most 16 digits per coordinate")
    return value


def _prime(value: int) -> int:
    value = _positive_integer(value, "prime")
    if value < 2 or any(value % divisor == 0 for divisor in range(2, isqrt(value) + 1)):
        raise ValueError("prime must be prime")
    return value


def digit_layout(digits: int | tuple[int, ...], coordinates: int, prime: int = 2) -> tuple[int, ...]:
    """Allow mixed precision while keeping each canonical coordinate in uint16."""
    prime = _prime(prime)
    coordinates = _positive_integer(coordinates, "coordinates")
    layout = tuple(_digits(value) for value in digits) if isinstance(digits, (tuple, list)) else (_digits(digits),) * coordinates
    if len(layout) != coordinates:
        raise ValueError("digit layout must match the coordinate count")
    if any(prime ** count > 65536 for count in layout):
        raise ValueError("each diagnostic coordinate alphabet must fit uint16")
    return layout


def digits_at_bit_budget(prime: int, bit_budget: int, coordinates: int) -> tuple[int, ...]:
    """Allocate total digits under p**total <= 2**budget using exact integers.

    Earlier coordinates get one extra digit when division is uneven. The
    resulting joint alphabet may not fill the entire byte/bit budget.
    """
    prime = _prime(prime)
    bit_budget = _positive_integer(bit_budget, "bit_budget")
    coordinates = _positive_integer(coordinates, "coordinates")
    total, alphabet = 0, 1
    while alphabet * prime <= (1 << bit_budget):
        total, alphabet = total + 1, alphabet * prime
    if total < coordinates:
        raise ValueError("bit budget cannot give each coordinate one base-prime digit")
    base, extra = divmod(total, coordinates)
    return digit_layout(tuple(base + (coordinate < extra) for coordinate in range(coordinates)), coordinates, prime)


def _codes(values: np.ndarray, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError("codes must have a nonempty coordinate dimension")
    if values.dtype.kind not in "iu":
        raise TypeError("codes must be integer arrays")
    layout = digit_layout(digits, values.shape[-1], prime)
    alphabets = np.array([prime ** count for count in layout], dtype=np.uint64)
    if np.any(values < 0) or np.any(values >= alphabets):
        raise ValueError("codes must be canonical residues for each coordinate modulus")
    return values.astype(np.uint16, copy=False)


def reverse_bits(values: np.ndarray, digits: int) -> np.ndarray:
    """Reverse K bits: a real coarse-to-fine path becomes low-to-high digits."""
    return reverse_digits(values, digits, prime=2)


def reverse_digits(values: np.ndarray, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    """Reverse each coordinate's base-prime path into finite p-adic digits."""
    values = _codes(values, digits, prime)
    layout = digit_layout(digits, values.shape[-1], prime)
    reversed_values = np.zeros_like(values)
    for coordinate, count in enumerate(layout):
        for level in range(count):
            digit = (values[..., coordinate] // (prime ** level)) % prime
            reversed_values[..., coordinate] += digit * (prime ** (count - level - 1))
    return reversed_values


def common_padic_depth(query: np.ndarray, keys: np.ndarray, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    """Number of shared low-order digits, minimized across coordinates.

    For distinct scalar residues this equals v_p(q-k). A coordinate that is
    equal at its finite precision contributes zero distance. Identical vector
    codes receive maximum depth, hence zero in the finite quotient metric.
    """
    query, keys = _codes(query, digits, prime), _codes(keys, digits, prime)
    if query.ndim != 1 or keys.ndim != 2 or query.shape[0] != keys.shape[1]:
        raise ValueError("query must be [coordinates] and keys [keys, coordinates]")
    layout = digit_layout(digits, len(query), prime)
    differences = keys.astype(np.int64) - query.astype(np.int64)
    depth = np.zeros(len(keys), dtype=np.int16)
    for level in range(1, max(layout) + 1):
        depth += np.all(differences % (prime ** level) == 0, axis=1)
    return depth


def finite_padic_distances(query: np.ndarray, keys: np.ndarray, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    """Product ultrametric max_j d_K(q_j,k_j), with equal residues at distance 0."""
    depth = common_padic_depth(query, keys, digits, prime)
    return np.where(depth == max(digit_layout(digits, len(query), prime)), 0.0,
                    float(prime) ** (-depth.astype(np.float64)))


def _selection_sizes(length: int, budget: int, recent_window: int) -> tuple[int, int]:
    budget = _positive_integer(budget, "budget")
    recent_window = _positive_integer(recent_window, "recent_window", allow_zero=True)
    count = min(length, budget)
    return count, min(recent_window, count)


def select_from_scores(scores: np.ndarray, budget: int, recent_window: int = 0) -> np.ndarray:
    """Select high scores, break ties by recency, return chronological indices.

    The input must contain only the causal prefix, including the current key.
    A mandatory recent window takes priority for every diagnostic method.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("scores must be a finite vector")
    count, recent = _selection_sizes(len(scores), budget, recent_window)
    if count == len(scores):
        return np.arange(count, dtype=np.int64)
    mandatory = np.arange(len(scores) - recent, len(scores), dtype=np.int64)
    pool = np.arange(len(scores) - recent, dtype=np.int64)
    ranking = np.lexsort((-pool, -scores[pool]))
    return np.sort(np.concatenate((mandatory, pool[ranking[:count - recent]])))


def select_causal_padic(
    query: np.ndarray, key_codes: np.ndarray, position: int, budget: int,
    *, digits: int | tuple[int, ...], recent_window: int = 0, prime: int = 2,
) -> np.ndarray:
    """Nearest finite p-adic balls, then recency; never reads future keys."""
    position = _positive_integer(position, "position", allow_zero=True)
    key_codes = np.asarray(key_codes)
    if key_codes.ndim != 2 or position >= len(key_codes):
        raise ValueError("position must address a key in a [keys, coordinates] array")
    depth = common_padic_depth(query, key_codes[:position + 1], digits, prime)
    return select_from_scores(depth, budget, recent_window)


@dataclass
class _TreeNode:
    indices: list[int] = field(default_factory=list)
    children: dict[tuple[int, ...], "_TreeNode"] = field(default_factory=dict)


class PrefixTree:
    """Append-only causal radix tree; does not call the p-adic distance routine.

    Each edge contains one low-order base-prime digit from every active
    coordinate. With m coordinates there are up to p**m children. Stored lists
    and traversal are deliberately simple reference implementations.
    """

    def __init__(self, coordinates: int, digits: int | tuple[int, ...], prime: int = 2):
        self.coordinates = _positive_integer(coordinates, "coordinates")
        self.prime = _prime(prime)
        self.digits = digits
        self.layout = digit_layout(digits, coordinates, prime)
        self.root = _TreeNode()
        self.length = 0

    def _path(self, code: np.ndarray) -> list[tuple[int, ...]]:
        code = _codes(code, self.digits, self.prime)
        if code.shape != (self.coordinates,):
            raise ValueError("tree code has the wrong coordinate shape")
        return [tuple((int(value) // self.prime ** level) % self.prime
                      if level < self.layout[coordinate] else -1
                      for coordinate, value in enumerate(code))
                for level in range(max(self.layout))]

    def append(self, code: np.ndarray) -> int:
        path = self._path(code)
        index = self.length
        node = self.root
        node.indices.append(index)
        for symbol in path:
            node = node.children.setdefault(symbol, _TreeNode())
            node.indices.append(index)
        self.length += 1
        return index

    def select(self, query: np.ndarray, budget: int, recent_window: int = 0) -> np.ndarray:
        path = self._path(query)
        count, recent = _selection_sizes(self.length, budget, recent_window)
        if count == self.length:
            return np.arange(count, dtype=np.int64)
        nodes = [self.root]
        for symbol in path:
            child = nodes[-1].children.get(symbol)
            if child is None:
                break
            nodes.append(child)
        mandatory = set(range(self.length - recent, self.length))
        selected = list(mandatory)
        if len(selected) == count:
            return np.array(sorted(selected), dtype=np.int64)
        # Deepest matching subtree first, then progressively more distant
        # sibling subtrees. Within a subtree, later causal keys win ties.
        for level in range(len(nodes) - 1, -1, -1):
            closer = set(nodes[level + 1].indices) if level + 1 < len(nodes) else set()
            for index in reversed(nodes[level].indices):
                if index not in closer and index not in mandatory:
                    selected.append(index)
                    if len(selected) == count:
                        return np.array(sorted(selected), dtype=np.int64)
        raise AssertionError("tree traversal failed to fill the causal budget")


@dataclass(frozen=True)
class QuantileCodebook:
    """Shared per-head real encoder, fitted exclusively to calibration Q/K."""

    center: np.ndarray                  # [heads, head_dim]
    projection: np.ndarray              # [heads, head_dim, coordinates]
    thresholds: np.ndarray              # [heads, coordinates, max(p**K_j)-1], padded
    digits: int | tuple[int, ...]
    projection_kind: str
    seed: int
    calibration_tokens: int
    prime: int = 2

    @classmethod
    def fit(
        cls, queries: np.ndarray, keys: np.ndarray, *, coordinates: int = 2,
        digits: int | tuple[int, ...] = 4, seed: int = 17, mask: np.ndarray | None = None,
        projection_kind: str = "pca", prime: int = 2, code_bit_budget: int | None = None,
    ) -> "QuantileCodebook":
        prime = _prime(prime)
        coordinates = _positive_integer(coordinates, "coordinates")
        layout = (digit_layout(digits, coordinates, prime) if code_bit_budget is None
                  else digits_at_bit_budget(prime, code_bit_budget, coordinates))
        if code_bit_budget is not None:
            digits = layout
        seed = _positive_integer(seed, "seed", allow_zero=True)
        queries, keys = np.asarray(queries, dtype=np.float64), np.asarray(keys, dtype=np.float64)
        if queries.ndim != 4 or queries.shape != keys.shape:
            raise ValueError("calibration Q/K must have identical [chunks,heads,seq,dim] shapes")
        chunks, heads, length, width = queries.shape
        if coordinates > width or not all(queries.shape):
            raise ValueError("coordinates must fit a nonempty head dimension")
        if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(keys)):
            raise ValueError("calibration Q/K must be finite")
        mask = np.ones((chunks, length), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (chunks, length) or not np.any(mask):
            raise ValueError("calibration mask must match chunks/seq and retain tokens")
        if projection_kind not in {"pca", "random"}:
            raise ValueError("projection_kind must be pca or random")
        centers, projections, thresholds = [], [], []
        max_alphabet = max(prime ** count for count in layout)
        for head in range(heads):
            pooled = np.concatenate((queries[:, head][mask], keys[:, head][mask]), axis=0)
            center = pooled.mean(axis=0)
            centered = pooled - center
            if projection_kind == "pca":
                covariance = centered.T @ centered / len(pooled)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                projection = eigenvectors[:, np.argsort(eigenvalues, kind="stable")[::-1][:coordinates]]
            else:
                rng = np.random.default_rng(np.random.SeedSequence([seed, head, 1103]))
                projection, _ = np.linalg.qr(rng.normal(size=(width, coordinates)), mode="reduced")
            # Eigenvector/QR signs have no geometric meaning; fix them to make
            # the explicit quantile representation reproducible.
            anchors = np.argmax(np.abs(projection), axis=0)
            signs = np.where(projection[anchors, np.arange(coordinates)] < 0, -1.0, 1.0)
            projection = projection * signs
            projected = centered @ projection
            cutpoints = np.empty((coordinates, max_alphabet - 1), dtype=np.float64)
            for coordinate, count in enumerate(layout):
                alphabet = prime ** count
                quantiles = np.arange(1, alphabet, dtype=np.float64) / alphabet
                cuts = np.quantile(projected[:, coordinate], quantiles, method="linear")
                cutpoints[coordinate, :alphabet - 1] = cuts
                # Padding is excluded by encode; retaining finite padding
                # makes artifacts easy to inspect and its bytes are counted.
                cutpoints[coordinate, alphabet - 1:] = cuts[-1]
            centers.append(center)
            projections.append(projection)
            thresholds.append(cutpoints)
        return cls(np.stack(centers), np.stack(projections), np.stack(thresholds),
                   digits, projection_kind, seed, int(mask.sum()), prime)

    def encode(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 4 or values.shape[1] != len(self.center) or values.shape[-1] != self.center.shape[-1]:
            raise ValueError("encoder input must match calibrated heads and head_dim")
        if not np.all(np.isfinite(values)):
            raise ValueError("encoder input must be finite")
        result = np.empty(values.shape[:-1] + (self.projection.shape[-1],), dtype=np.uint16)
        layout = digit_layout(self.digits, self.projection.shape[-1], self.prime)
        for head in range(values.shape[1]):
            projected = (values[:, head] - self.center[head]) @ self.projection[head]
            for coordinate in range(self.projection.shape[-1]):
                # Duplicate cutpoints are intentional. Constant calibration
                # data remains deterministic, though its alphabet collapses.
                result[:, head, :, coordinate] = np.searchsorted(
                    self.thresholds[head, coordinate, :self.prime ** layout[coordinate] - 1],
                    projected[..., coordinate], side="right")
        return reverse_digits(result, self.digits, self.prime)


def angular_hyperplanes(heads: int, width: int, bits: int, seed: int) -> np.ndarray:
    """Seeded origin hyperplanes for standard angular/sign LSH, not data-fit."""
    heads, width, bits = (_positive_integer(x, n) for x, n in
                          ((heads, "heads"), (width, "width"), (bits, "bits")))
    seed = _positive_integer(seed, "seed", allow_zero=True)
    result = []
    for head in range(heads):
        rng = np.random.default_rng(np.random.SeedSequence([seed, head, 2207]))
        planes = rng.normal(size=(width, bits))
        result.append(planes / np.linalg.norm(planes, axis=0))
    return np.stack(result)


def angular_codes(values: np.ndarray, planes: np.ndarray) -> np.ndarray:
    values, planes = np.asarray(values, dtype=np.float64), np.asarray(planes, dtype=np.float64)
    if values.ndim != 4 or planes.ndim != 3 or values.shape[1] != planes.shape[0] or values.shape[-1] != planes.shape[1]:
        raise ValueError("angular input/planes shapes disagree")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(planes)):
        raise ValueError("angular input/planes must be finite")
    return np.einsum("chts,hsp->chtp", values, planes) >= 0


def select_causal_angular(query: np.ndarray, keys: np.ndarray, position: int,
                          budget: int, recent_window: int = 0) -> np.ndarray:
    query, keys = np.asarray(query, dtype=bool), np.asarray(keys, dtype=bool)
    position = _positive_integer(position, "position", allow_zero=True)
    if query.ndim != 1 or keys.ndim != 2 or keys.shape[1] != len(query) or position >= len(keys):
        raise ValueError("angular query/keys/position disagree")
    return select_from_scores(np.sum(keys[:position + 1] == query, axis=1), budget, recent_window)


def select_causal_random(position: int, budget: int, *, seed: int,
                         chunk: int = 0, head: int = 0, recent_window: int = 0) -> np.ndarray:
    position = _positive_integer(position, "position", allow_zero=True)
    identifiers = [_positive_integer(x, n, allow_zero=True) for x, n in
                   ((seed, "seed"), (chunk, "chunk"), (head, "head"))]
    length = position + 1
    count, recent = _selection_sizes(length, budget, recent_window)
    if count == length:
        return np.arange(length, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(identifiers + [position, budget, 3301]))
    mandatory = np.arange(length - recent, length, dtype=np.int64)
    sampled = rng.choice(length - recent, size=count - recent, replace=False)
    return np.sort(np.concatenate((mandatory, sampled)))


def attention_probabilities(query: np.ndarray, causal_keys: np.ndarray) -> np.ndarray:
    query, causal_keys = np.asarray(query, dtype=np.float64), np.asarray(causal_keys, dtype=np.float64)
    if query.ndim != 1 or causal_keys.ndim != 2 or causal_keys.shape[1] != len(query) or not len(query) or not len(causal_keys):
        raise ValueError("attention needs a nonempty query and causal key matrix")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(causal_keys)):
        raise ValueError("attention inputs must be finite")
    logits = causal_keys @ query / np.sqrt(len(query))
    weights = np.exp(logits - np.max(logits))
    return weights / weights.sum()


def pack_codes(codes: np.ndarray, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    """Pack joint mixed-radix residues into the minimum fixed-width bytes.

    Unlike separate coordinate containers this does not impose extra rounding
    for odd-prime digits. The joint alphabet may leave some byte codes unused.
    This diagnostic supports joint alphabets up to 2**64.
    """
    codes = _codes(codes, digits, prime)
    layout = digit_layout(digits, codes.shape[-1], prime)
    alphabet = prime ** sum(layout)
    bits = (alphabet - 1).bit_length()
    if bits > 64:
        raise ValueError("joint diagnostic packing supports at most 64 bits")
    joint = np.zeros(codes.shape[:-1], dtype=np.uint64)
    multiplier = 1
    for coordinate, count in enumerate(layout):
        joint += codes[..., coordinate].astype(np.uint64) * np.uint64(multiplier)
        multiplier *= prime ** count
    offsets = np.arange((bits + 7) // 8, dtype=np.uint64) * 8
    return ((joint[..., None] >> offsets) & 255).astype(np.uint8)


def unpack_codes(packed: np.ndarray, coordinates: int, digits: int | tuple[int, ...], prime: int = 2) -> np.ndarray:
    coordinates = _positive_integer(coordinates, "coordinates")
    packed = np.asarray(packed)
    layout = digit_layout(digits, coordinates, prime)
    alphabet = prime ** sum(layout)
    bits = (alphabet - 1).bit_length()
    if bits > 64:
        raise ValueError("joint diagnostic packing supports at most 64 bits")
    if packed.dtype != np.uint8 or packed.ndim < 1 or packed.shape[-1] != (bits + 7) // 8:
        raise ValueError("packed codes must have uint8 dtype and the exact byte width")
    offsets = np.arange(packed.shape[-1], dtype=np.uint64) * 8
    joint = np.sum(packed.astype(np.uint64) << offsets, axis=-1, dtype=np.uint64)
    if alphabet < (1 << 64) and np.any(joint >= alphabet):
        raise ValueError("packed codes contain unused alphabet or padding values")
    result = np.empty(packed.shape[:-1] + (coordinates,), dtype=np.uint16)
    for coordinate, count in enumerate(layout):
        modulus = prime ** count
        result[..., coordinate] = joint % modulus
        joint //= modulus
    return result
