"""Causal group-shared rank routing for a fixed numerical development study."""
from math import lcm
import numpy as np

from .routing import reverse_digits, digit_layout


def twice_midranks(scores):
    """Exact integer ranks; ties have their average rank, multiplied by two."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("nonempty [heads, optional keys] scores required")
    if not np.all(np.isfinite(scores)):
        raise ValueError("finite optional scores required")
    order = np.argsort(scores, axis=-1, kind="stable")
    values = np.take_along_axis(scores, order, axis=-1)
    n = scores.shape[1]
    positions = np.broadcast_to(np.arange(n, dtype=np.int32), scores.shape)
    first = np.ones(scores.shape, dtype=bool)
    first[:, 1:] = values[:, 1:] != values[:, :-1]
    last = np.ones(scores.shape, dtype=bool)
    last[:, :-1] = values[:, 1:] != values[:, :-1]
    starts = np.maximum.accumulate(np.where(first, positions, 0), axis=-1)
    ends = np.minimum.accumulate(np.where(last, positions, n - 1)[:, ::-1], axis=-1)[:, ::-1]
    ranks = np.empty(scores.shape, dtype=np.int32)
    np.put_along_axis(ranks, order, starts + ends, axis=-1)
    return ranks


def shared_rank_mask(length, heads, score_row, *, budget=128, recent=8, group_size=3):
    """Return [groups,queries,keys]; score_row never receives future keys."""
    for value in (length, heads, budget, group_size):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("positive integer dimensions/budget required")
    if not isinstance(recent, int) or isinstance(recent, bool) or not 0 <= recent <= budget:
        raise ValueError("mandatory recent count must fit the budget")
    if heads % group_size:
        raise ValueError("heads must form complete GQA groups")
    groups = heads // group_size
    selected = np.zeros((groups, length, length), dtype=bool)
    for position in range(length):
        count = min(position + 1, budget)
        if count == position + 1:
            selected[:, position, :position + 1] = True
            continue
        local = min(recent, count)
        stop = position + 1 - local
        if local:
            selected[:, position, stop:position + 1] = True
        if count == local:
            continue
        scores = np.asarray(score_row(position, stop))
        if scores.shape != (heads, stop):
            raise ValueError("score provider must return all heads and only the optional prefix")
        rank = twice_midranks(scores).reshape(groups, group_size, stop).sum(axis=1, dtype=np.int32)
        # Stable ascending order, take its end: exact ties prefer newer keys.
        indices = np.argsort(rank, axis=-1, kind="stable")[:, -(count - local):]
        np.put_along_axis(selected[:, position, :stop], indices, True, axis=-1)
    validate_group_mask(selected, budget=budget, recent=recent)
    return selected


def validate_group_mask(mask, *, budget=128, recent=8):
    if mask.ndim != 3 or mask.dtype != bool or mask.shape[-1] != mask.shape[-2]:
        raise ValueError("square boolean group masks required")
    length = mask.shape[-1]
    if np.any(np.triu(mask, 1)):
        raise ValueError("future key selected")
    if not np.all(mask.sum(-1) == np.minimum(np.arange(1, length + 1), budget)):
        raise ValueError("shared budget/cardinality differs")
    for pos in range(length):
        if not mask[:, pos, max(0, pos + 1 - min(recent, budget)):pos + 1].all():
            raise ValueError("mandatory recent key omitted")


def fixed_group_mask(length, groups, kind, *, budget=128, recent=8):
    if kind not in ("full", "recency", "sink_recency", "uniform"):
        raise ValueError("unknown fixed control")
    result = np.zeros((groups, length, length), dtype=bool)
    for pos in range(length):
        count = pos + 1 if kind == "full" else min(pos + 1, budget)
        if count == pos + 1:
            indices = np.arange(count)
        elif kind == "recency":
            indices = np.arange(pos + 1 - count, pos + 1)
        elif kind == "sink_recency":
            sinks = min(4, count - min(recent, count))
            indices = np.r_[np.arange(sinks), np.arange(pos + 1 - count + sinks, pos + 1)]
        else:
            local = min(recent, count)
            stop = pos + 1 - local
            optional = count - local
            spaced = ((2 * np.arange(optional) + 1) * stop) // (2 * optional) if optional else np.array([], dtype=int)
            indices = np.r_[spaced, np.arange(stop, pos + 1)]
        result[:, pos, indices] = True
    validate_group_mask(result, budget=length if kind == "full" else budget, recent=recent)
    return result


def depth_row(query, keys, digits, prime):
    """Direct finite congruences, equal finite coordinates contribute all levels."""
    layout = digit_layout(digits, query.shape[-1], prime)
    difference = keys.astype(np.int64) - query.astype(np.int64)[:, None, :]
    result = np.zeros(difference.shape[:-1], dtype=np.int16)
    for level in range(1, max(layout) + 1):
        result += np.all(difference % (prime ** level) == 0, axis=-1)
    return result


def grid_coordinates(codes, digits, prime):
    layout = digit_layout(digits, codes.shape[-1], prime)
    widths = [prime ** n - 1 for n in layout]
    unit = lcm(*widths)
    scale = np.array([unit // width for width in widths], dtype=np.int64)
    return reverse_digits(codes, layout, prime).astype(np.int64) * scale, scale


def grid_row(query, keys):
    delta = query[:, None, :] - keys
    return -(delta * delta).sum(axis=-1)


def mass_group_mask(probabilities, *, budget=128, recent=8, group_size=3):
    """Top summed native per-head mass; this upper control needs dense scores."""
    probabilities = np.asarray(probabilities)
    heads, length, width = probabilities.shape
    if width != length or heads % group_size or not np.all(np.isfinite(probabilities)):
        raise ValueError("finite square native probabilities in complete groups required")
    scores = probabilities.reshape(heads // group_size, group_size, length, length).sum(axis=1, dtype=np.float64)
    result = fixed_group_mask(length, heads // group_size, "recency", budget=budget, recent=recent)
    for pos in range(budget, length):
        stop = pos + 1 - min(recent, budget)
        result[:, pos] = False
        result[:, pos, stop:pos + 1] = True
        optional = budget - min(recent, budget)
        if optional:
            indices = np.argsort(scores[:, pos, :stop], axis=-1, kind="stable")[:, -optional:]
            np.put_along_axis(result[:, pos, :stop], indices, True, axis=-1)
    validate_group_mask(result, budget=budget, recent=recent)
    return result
