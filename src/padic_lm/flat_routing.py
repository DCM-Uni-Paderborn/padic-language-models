"""Calibration-only, one-byte flat key-centroid routing diagnostic.

The real center/projection are supplied by the corresponding shared encoder;
this module never refits them or observes queries during centroid fitting.
Queries remain real two-dimensional vectors. The primary Euclidean control
ranks distance to a key's assigned centroid; a secondary projected-dot score
uses the same codes and centers. Neither guarantees approximation quality
for the model's original, full-dimensional QK attention scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .routing import select_from_scores


def _integer(value: int, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{name} is outside its supported range")
    return result


def _canonical(points: np.ndarray) -> np.ndarray:
    """Lexicographic exact unique points; normalize signed-zero duplicates."""
    points = np.array(points, dtype=np.float64, copy=True)
    points[points == 0] = 0.0
    return np.unique(points, axis=0)


def _squared_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.sum((points[..., None, :] - centroids) ** 2, axis=-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("squared centroid distances must remain finite in float64")
    return result


def _fit_head(points: np.ndarray, maximum: int, seed: int, head: int,
              max_iterations: int) -> tuple[np.ndarray, int, int]:
    points = np.array(points, dtype=np.float64, copy=True)
    points[points == 0] = 0.0
    unique, weights = np.unique(points, axis=0, return_counts=True)
    count = min(len(unique), maximum)
    if count == len(unique):
        # At 256 distinct calibration keys this deliberately memorizes their
        # projected locations; it is not evidence of out-of-sample quality.
        return unique, len(unique), 0
    rng = np.random.default_rng(np.random.SeedSequence([seed, head, 4409]))
    centroids = unique[np.sort(rng.choice(len(unique), count, replace=False))].copy()
    for iteration in range(1, max_iterations + 1):
        labels = np.argmin(_squared_distances(unique, centroids), axis=-1)
        updated = []
        for index in range(len(centroids)):
            selected = labels == index
            if not np.any(selected):
                # Empty centers carry no code meaning and are dropped. No
                # extra random restart or evaluation-dependent choice occurs.
                continue
            with np.errstate(over="ignore", invalid="ignore"):
                mean = np.sum(unique[selected] * weights[selected, None], axis=0) / weights[selected].sum()
            if not np.all(np.isfinite(mean)):
                raise ValueError("fitted centroids must remain finite in float64")
            updated.append(mean)
        updated = _canonical(np.stack(updated))
        if np.array_equal(updated, centroids):
            return updated, len(unique), iteration
        centroids = updated
    return centroids, len(unique), max_iterations


@dataclass(frozen=True)
class FlatKeyCodebook:
    """Per-head flat centers with uint8 key codes and a fixed shared 2D map.

    Exact centroid-distance ties use the lowest lexicographic centroid ID.
    Key-ranking ties use recency through ``select_from_scores``. Lloyd updates
    use exact duplicate counts in canonical order, drop empty/duplicate
    centers, and stop on an exactly unchanged state or the declared cap.
    """

    center: np.ndarray                  # float64 [heads, head_dim]
    projection: np.ndarray              # float64 [heads, head_dim, 2]
    centroids: np.ndarray               # float64 [heads, max_active_centers, 2], zero padded
    centroid_counts: np.ndarray         # uint16 [heads], including the possible count 256
    unique_key_counts: np.ndarray       # uint64 [heads], projected calibration uniqueness
    calibration_counts: np.ndarray      # uint64 [heads], retained calibration key count
    iterations: np.ndarray              # uint8 [heads], zero for exact unique-point storage
    fit_parameters: np.ndarray          # uint64 [configured max_centroids, seed, max_iterations]

    @classmethod
    def fit(
        cls, keys: np.ndarray, *, center: np.ndarray, projection: np.ndarray,
        max_centroids: int = 256, seed: int = 17, max_iterations: int = 20,
        mask: np.ndarray | None = None,
    ) -> "FlatKeyCodebook":
        """Fit key-only centroids on the caller's declared calibration chunks.

        Masked padding is excluded before projection and finite checks. The
        caller must provide only calibration chunks; no temporal split can be
        inferred from an arbitrary array. The prospective diagnostic uses
        chunks 0/1 exclusively and a fixed maximum of 20 Lloyd iterations.
        """
        maximum = _integer(max_centroids, "max_centroids", minimum=1, maximum=256)
        seed = _integer(seed, "seed", maximum=np.iinfo(np.uint64).max)
        cap = _integer(max_iterations, "max_iterations", minimum=1, maximum=20)
        keys = np.asarray(keys, dtype=np.float64)
        center = np.array(center, dtype=np.float64, copy=True)
        projection = np.array(projection, dtype=np.float64, copy=True)
        if keys.ndim != 4 or not all(keys.shape):
            raise ValueError("calibration keys must be nonempty [chunks,heads,seq,dim]")
        chunks, heads, length, width = keys.shape
        if width < 2 or center.shape != (heads, width) or projection.shape != (heads, width, 2):
            raise ValueError("shared center/projection must match heads/head_dim and two coordinates")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(projection)):
            raise ValueError("shared center/projection must be finite")
        if mask is None:
            mask = np.ones((chunks, length), dtype=bool)
        else:
            mask = np.asarray(mask)
            if mask.dtype.kind != "b":
                raise TypeError("calibration mask must be boolean")
        if mask.shape != (chunks, length) or not np.any(mask):
            raise ValueError("calibration mask must match chunks/seq and retain keys")
        fitted, unique_counts, iterations = [], [], []
        for head in range(heads):
            retained = keys[:, head][mask]
            if not np.all(np.isfinite(retained)):
                raise ValueError("retained calibration keys must be finite")
            with np.errstate(over="ignore", invalid="ignore"):
                projected = (retained - center[head]) @ projection[head]
            if not np.all(np.isfinite(projected)):
                raise ValueError("projected calibration keys must remain finite in float64")
            centers, unique_count, steps = _fit_head(projected, maximum, seed, head, cap)
            fitted.append(centers)
            unique_counts.append(unique_count)
            iterations.append(steps)
        counts = np.array([len(values) for values in fitted], dtype=np.uint16)
        padded = np.zeros((heads, int(counts.max()), 2), dtype=np.float64)
        for head, values in enumerate(fitted):
            padded[head, :len(values)] = values
        result = cls(center, projection, padded, counts,
                     np.array(unique_counts, dtype=np.uint64),
                     np.full(heads, int(mask.sum()), dtype=np.uint64),
                     np.array(iterations, dtype=np.uint8),
                     np.array([maximum, seed, cap], dtype=np.uint64))
        for values in result.state_arrays.values():
            values.setflags(write=False)
        return result

    @property
    def seed(self) -> int:
        return int(self.fit_parameters[1])

    @property
    def max_centroids(self) -> int:
        return int(self.fit_parameters[0])

    @property
    def max_iterations(self) -> int:
        return int(self.fit_parameters[2])

    @property
    def state_arrays(self) -> dict[str, np.ndarray]:
        """Every stored NumPy buffer, including padding and fit metadata."""
        return {name: getattr(self, name) for name in
                ("center", "projection", "centroids", "centroid_counts",
                 "unique_key_counts", "calibration_counts", "iterations", "fit_parameters")}

    @property
    def state_bytes(self) -> int:
        """Actual ndarray payload bytes, excluding Python/file-format overhead.

        This counts full padded centroids, duplicated shared encoder arrays
        and diagnostic fit metadata. One uint8 key code per head/token is a
        separate payload; masks, indexes and real KV arrays are not compressed.
        """
        return sum(values.nbytes for values in self.state_arrays.values())

    def project_queries(self, values: np.ndarray) -> np.ndarray:
        """Apply the supplied map without quantizing or refitting queries."""
        values = np.asarray(values, dtype=np.float64)
        if (values.ndim != 4 or values.shape[1] != len(self.center)
                or values.shape[-1] != self.center.shape[-1]):
            raise ValueError("input must match calibrated [chunks,heads,seq,head_dim]")
        if not np.all(np.isfinite(values)):
            raise ValueError("projection inputs must be finite")
        projected = np.empty(values.shape[:-1] + (2,), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for head in range(values.shape[1]):
                projected[:, head] = (values[:, head] - self.center[head]) @ self.projection[head]
        if not np.all(np.isfinite(projected)):
            raise ValueError("projected inputs must remain finite in float64")
        return projected

    def encode_keys(self, values: np.ndarray) -> np.ndarray:
        """Nearest-centroid uint8 IDs; duplicate/tied centers use canonical IDs."""
        projected = self.project_queries(values)
        codes = np.empty(projected.shape[:-1], dtype=np.uint8)
        for head in range(projected.shape[1]):
            distances = _squared_distances(projected[:, head],
                                          self.centroids[head, :int(self.centroid_counts[head])])
            codes[:, head] = np.argmin(distances, axis=-1)
        return codes

    def _head_codes(self, codes: np.ndarray, head: int) -> np.ndarray:
        codes = np.asarray(codes)
        if codes.dtype.kind not in "iu":
            raise TypeError("key codes must be integer arrays")
        if np.any(codes < 0) or np.any(codes >= int(self.centroid_counts[head])):
            raise ValueError("key codes must address active centroids")
        return codes

    def decode_keys(self, codes: np.ndarray) -> np.ndarray:
        """Decode [chunks,heads,seq] IDs to their real 2D assigned centroids."""
        codes = np.asarray(codes)
        if codes.ndim != 3 or codes.shape[1] != len(self.center):
            raise ValueError("key codes must be [chunks,heads,seq]")
        decoded = np.empty(codes.shape + (2,), dtype=np.float64)
        for head in range(codes.shape[1]):
            decoded[:, head] = self.centroids[head, self._head_codes(codes[:, head], head)]
        return decoded

    def select_causal(
        self, query: np.ndarray, key_codes: np.ndarray, position: int,
        budget: int = 32, recent_window: int = 8, *, head: int = 0,
        score_kind: str = "euclidean",
    ) -> np.ndarray:
        """Rank only the causal prefix using the declared centroid score.

        ``query`` is the real 2D result of ``project_queries``. The primary
        ``euclidean`` score is negative squared centroid distance. The
        secondary ``dot_product`` score restores the original projected
        origin with ``center @ projection`` for both query and centroid;
        it uses identical fitted state and codes, with no cached offset.
        Future code values are neither decoded nor validated. Mandatory recent keys occupy
        the same declared budget, and score ties prefer more recent keys.
        """
        head = _integer(head, "head", maximum=len(self.center) - 1)
        position = _integer(position, "position")
        query = np.asarray(query, dtype=np.float64)
        key_codes = np.asarray(key_codes)
        if query.shape != (2,) or not np.all(np.isfinite(query)):
            raise ValueError("query must be a finite real 2D vector")
        if key_codes.ndim != 1 or position >= len(key_codes):
            raise ValueError("position must address a key in a [keys] code array")
        prefix = self._head_codes(key_codes[:position + 1], head)
        centers = self.centroids[head, prefix]
        if score_kind == "euclidean":
            scores = -_squared_distances(query, centers)
        elif score_kind == "dot_product":
            with np.errstate(over="ignore", invalid="ignore"):
                offset = self.center[head] @ self.projection[head]
                scores = np.sum((query + offset) * (centers + offset), axis=-1)
            if not np.all(np.isfinite(scores)):
                raise ValueError("projected centroid dot products must remain finite in float64")
        else:
            raise ValueError("score_kind must be euclidean or dot_product")
        return select_from_scores(scores, budget, recent_window)
