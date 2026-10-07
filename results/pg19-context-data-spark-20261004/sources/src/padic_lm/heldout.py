"""Fixed document eligibility and paired quality estimates for untouched testing."""
import hashlib
from pathlib import Path

import numpy as np


PROTOCOL_HASH = "9a0095eb8e079773e63d259c292a4d7ad10b932ded90df58026bbf90be2048d0"
ACCEPTED_HASH = "033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454"
DEVELOPMENT_HASH = "ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8"
DEVELOPMENT_AUDIT_HASH = "28b682bdf864333c6161606cfc4e6c67260551cab60fe5eb6f162676a6439a36"
RESAMPLES = 20_000
RESAMPLE_SEED = 514203
QUALITY_MARGIN = np.log(1.01)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(docs, tokenize, forbidden_hashes, length=2048):
    """Preserve every article; select every eligible first prefix in source order."""
    records, selected, buffers, seen = [], [], [], set()
    for doc in docs:
        tokens = np.asarray(tokenize(doc.text), dtype="<i8")
        if tokens.ndim != 1 or np.any(tokens < 0):
            raise ValueError("one-dimensional nonnegative token IDs required")
        reasons = []
        if doc.text_sha256 in forbidden_hashes:
            reasons.append("train_or_validation_exact_text")
        if doc.text_sha256 in seen:
            reasons.append("repeated_test_exact_text")
        if len(tokens) < length:
            reasons.append("short_article")
        seen.add(doc.text_sha256)
        record = {"article_id": f"test-{doc.start_row:06d}-{doc.text_sha256[:16]}",
                  "title": doc.title, "start_row": doc.start_row, "stop_row": doc.stop_row,
                  "text_sha256": doc.text_sha256, "whole_article_tokens": len(tokens),
                  "whole_token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
                  "eligible": not reasons, "exclusion_reasons": reasons}
        records.append(record)
        if not reasons:
            prefix = tokens[:length].copy()
            selected.append({**record, "token_start": 0, "token_stop": length,
                             "prefix2048_sha256": hashlib.sha256(prefix.tobytes()).hexdigest()})
            buffers.append(prefix)
    ids = np.stack(buffers) if buffers else np.empty((0, length), dtype="<i8")
    return records, selected, ids


def paired_document_bootstrap(candidate_sums, reference_sums, targets, indices):
    """Recompute token-weighted paired means, preserving whole document units."""
    candidate, reference, counts = [np.asarray(a, dtype=np.float64) for a in
                                    (candidate_sums, reference_sums, targets)]
    draws = np.asarray(indices)
    if (candidate.ndim != 1 or len(candidate) == 0 or reference.shape != candidate.shape
            or counts.shape != candidate.shape or not np.all(np.isfinite(candidate))
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(counts))
            or np.any(counts <= 0) or np.any(counts != np.floor(counts))
            or draws.ndim != 2 or draws.shape[1] != len(candidate) or draws.shape[0] == 0
            or not np.issubdtype(draws.dtype, np.integer)
            or np.any(draws < 0) or np.any(draws >= len(candidate))):
        raise ValueError("finite document sums/counts and valid paired resample indices required")
    difference = candidate - reference
    samples = difference[draws].sum(1) / counts[draws].sum(1)
    endpoints = np.quantile(samples, [.025, .975], method="linear")
    return {"paired_delta_nll": float(difference.sum() / counts.sum()),
            "lower_025": float(endpoints[0]), "upper_975": float(endpoints[1])}, samples
