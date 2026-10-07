"""Exact finite-prefix rank selection with causal postings and threshold access.

CPU correctness prototype. No native speed, sublinear worst-case or new-math claim.
Every stored key belongs to each ancestor posting list. Query access orders each
head by prefix score, then newest key; a monotone aggregate threshold stops only
when unseen keys cannot outrank the retained heap, including timestamp ties.
"""
import heapq

import numpy as np


class CausalPrefixIndex:
    def __init__(self, prime, digits, heads=3, coarsen=1):
        if prime not in (2, 3, 5) or len(digits) != 2 or any(int(d) != d or d < 1 for d in digits):
            raise ValueError("prime2/3/5 and two positive digit precisions required")
        if heads < 1 or int(heads) != heads or coarsen < 1 or int(coarsen) != coarsen:
            raise ValueError("positive integer head count/coarsening required")
        self.prime, self.digits, self.heads, self.coarsen = int(prime), tuple(int(d) for d in digits), int(heads), int(coarsen)
        self.depth = max(self.digits)
        self.alphabet = self.prime ** sum(self.digits)
        if self.alphabet > 256:
            raise ValueError("this prototype requires the declared one-byte code alphabet")
        self.cut_depths = tuple(range(0, self.depth + 1, self.coarsen))
        # Packing is residue0 + p^digits0 * residue1, independently of ordinals.
        codes = np.arange(self.alphabet, dtype=np.int64)
        left, right = codes % self.prime ** self.digits[0], codes // self.prime ** self.digits[0]
        self.prefix = np.stack([(left % self.prime ** min(d, self.digits[0])) +
            (right % self.prime ** min(d, self.digits[1])) * self.prime ** min(d, self.digits[0])
            for d in self.cut_depths], axis=1)
        self.postings = [[[[] for _ in range(self.prime ** (min(d, self.digits[0]) + min(d, self.digits[1])))]
                          for d in self.cut_depths] for _ in range(self.heads)]
        self.codes = [[] for _ in range(self.heads)]
        self.keys = 0

    def validate_codes(self, values):
        raw = np.asarray(values)
        if raw.shape != (self.heads,) or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw >= self.alphabet):
            raise ValueError("one valid integer packed code per head required")
        return raw.astype(np.int64)

    def append(self, key_codes):
        """Append the next optional causal key; recent keys remain outside index."""
        values = self.validate_codes(key_codes)
        for head, code in enumerate(values):
            self.codes[head].append(int(code))
            for level, prefix in enumerate(self.prefix[code]):
                self.postings[head][level][prefix].append(self.keys)
        self.keys += 1

    def query(self, query_codes, quota=120):
        """Return ascending optional key IDs, matching summed exact twice-midranks."""
        values = self.validate_codes(query_codes)
        if quota < 0 or int(quota) != quota:
            raise ValueError("nonnegative integer optional quota required")
        quota = min(int(quota), self.keys)
        diagnostics = {"optional_keys": self.keys, "selected_optional_keys": quota,
                       "head_list_entries_yielded": 0, "posting_entries_examined": 0,
                       "distinct_candidates_scored": 0, "termination": "all_keys_fit"}
        if quota == 0 or quota == self.keys:
            return np.arange(self.keys if quota else 0, dtype=np.int64), diagnostics
        paths, rank_tables = [], []
        for head, code in enumerate(values):
            path = [self.postings[head][level][prefix] for level, prefix in enumerate(self.prefix[code])]
            counts = np.array([len(p) for p in path] + [0], dtype=np.int64)
            shells = counts[:-1] - counts[1:]
            ranks = 2 * (np.cumsum(shells) - shells) + shells - 1
            # cumsum above follows increasing prefix depth (increasing score).
            # All root keys occur in exactly one shell, including nonmatches.
            if shells.sum() != self.keys or np.any(shells < 0):
                raise AssertionError("nested causal postings differ")
            paths.append(path)
            rank_tables.append(ranks)

        def ordered(head):
            path = paths[head]
            for level in range(len(path) - 1, -1, -1):
                parent = path[level]
                child = path[level + 1] if level + 1 < len(path) else []
                if len(parent) == len(child):
                    continue
                child_position = len(child) - 1
                for key in reversed(parent):
                    diagnostics["posting_entries_examined"] += 1
                    if child_position >= 0 and key == child[child_position]:
                        child_position -= 1
                    else:
                        yield int(rank_tables[head][level]), key

        streams = [iter(ordered(head)) for head in range(self.heads)]
        frontier = [None] * self.heads
        seen, heap = set(), []
        while True:
            for head, stream in enumerate(streams):
                try:
                    rank, key = next(stream)
                except StopIteration:
                    if len(seen) != self.keys or len(heap) != quota:
                        raise AssertionError("exhausted list did not expose all optional keys")
                    diagnostics["termination"] = "one_complete_head_list"
                    return np.array(sorted(k for _, k in heap), dtype=np.int64), diagnostics
                frontier[head] = (rank, key)
                diagnostics["head_list_entries_yielded"] += 1
                if key not in seen:
                    seen.add(key)
                    diagnostics["distinct_candidates_scored"] += 1
                    total = 0
                    for other in range(self.heads):
                        prefix = self.prefix[self.codes[other][key]]
                        equal = prefix == self.prefix[values[other]]
                        level = int(np.count_nonzero(equal)) - 1
                        total += int(rank_tables[other][level])
                    item = (total, key)
                    if len(heap) < quota:
                        heapq.heappush(heap, item)
                    elif item > heap[0]:
                        heapq.heapreplace(heap, item)
                if len(heap) == quota and all(f is not None for f in frontier):
                    # An unseen key has score_h<=frontier_h. If the summed
                    # upper score is attained, all head scores are equal and
                    # its timestamp is <=every frontier timestamp.
                    upper = (sum(f[0] for f in frontier), min(f[1] for f in frontier))
                    if heap[0] >= upper:
                        diagnostics["termination"] = "aggregate_score_and_timestamp_threshold"
                        return np.array(sorted(k for _, k in heap), dtype=np.int64), diagnostics

    def stored_position_entries(self):
        """Logical posting positions only; excludes Python objects/code/table state."""
        return sum(len(p) for head in self.postings for level in head for p in level)
