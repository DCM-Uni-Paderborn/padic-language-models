"""Independent complete official-test delimiter/token/eligibility reconstruction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

CONTEXT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path):
    assert sha(path / "completed.json") == (path / "completed.sha256").read_text().strip()
    record = json.loads((path / "completed.json").read_text())
    for name, digest in record["artifact_sha256"].items():
        assert sha(path / name) == digest
        assert (path / name).stat().st_size == record["artifact_bytes"][name]
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "accepted", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh independent held-out verification required")
    args.output.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(source)
    CONTEXT.update(output=args.output, started=started, prior=0.)
    completion = verify(args.input)
    CONTEXT["prior"] = completion["data_preparation_wall_seconds"]
    verify(args.accepted)
    assert sha(args.accepted / "completed.json") == "033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454"
    accepted = json.loads((args.accepted / "manifest.json").read_text())
    data = json.loads((args.input / "manifest.json").read_text())
    assert data["protocol_sha256"] == sha(args.input / "prospective_protocol.txt") == "9a0095eb8e079773e63d259c292a4d7ad10b932ded90df58026bbf90be2048d0"
    assert data["development_completion_sha256"] == "ca3453150c3727770871cefc4744b37984bc399f35803d8f7c5e67bce3e395e8"
    assert data["development_audit_sha256"] == "28b682bdf864333c6161606cfc4e6c67260551cab60fe5eb6f162676a6439a36"
    assert data["split"] == "test" and data["test_split_read"] and not data["model_output_read"]
    assert (data["dataset"], data["dataset_config"], data["dataset_revision"]) == (
        "Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3")
    assert data["model_revision"] == accepted["model_revision"]
    from datasets import load_dataset
    from transformers import AutoTokenizer
    rows = [record["text"] for record in load_dataset(data["dataset"], data["dataset_config"],
                 revision=data["dataset_revision"], split="test", streaming=True)]
    assert len(rows) == data["row_count"]
    boundaries = []
    for index, row in enumerate(rows):
        words = row.strip().split()
        if len(words) >= 3 and words[0] == words[-1] == "=" and words[1] != "=" and words[-2] != "=":
            boundaries.append(index)
    assert not any(row.strip() for row in rows[:boundaries[0]])
    assert boundaries == [record["start_row"] for record in data["inventory"]]
    forbidden = set()
    for split in ("train", "validation"):
        for record in accepted["inventory"][split]:
            forbidden.add(record["text_sha256"])
    with np.load(args.input / "tokens.npz", allow_pickle=False) as raw:
        tokens = raw["tokens2048"].copy()
    assert tokens.shape == (data["article_count"], 2048)
    tokenizer = AutoTokenizer.from_pretrained(data["model"], revision=data["tokenizer_revision"])
    selected, seen = [], set()
    for index, start in enumerate(boundaries):
        stop = boundaries[index + 1] if index + 1 < len(boundaries) else len(rows)
        text = "\n\n".join(rows[start:stop])
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        ids = np.array(tokenizer(text, add_special_tokens=False)["input_ids"], dtype="<i8")
        record = data["inventory"][index]
        assert record["title"] == rows[start].strip()[1:-1].strip()
        assert (record["start_row"], record["stop_row"], record["text_sha256"]) == (start, stop, text_hash)
        assert record["whole_article_tokens"] == len(ids)
        assert record["whole_token_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
        assert record["article_id"] == f"test-{start:06d}-{text_hash[:16]}"
        reasons = []
        if text_hash in forbidden:
            reasons.append("train_or_validation_exact_text")
        if text_hash in seen:
            reasons.append("repeated_test_exact_text")
        if len(ids) < 2048:
            reasons.append("short_article")
        seen.add(text_hash)
        assert record["eligible"] == (not reasons) and record["exclusion_reasons"] == reasons
        if not reasons:
            slot = len(selected)
            prefix = ids[0:2048]
            frozen = data["articles"][slot]
            for name in record:
                assert frozen[name] == record[name]
            assert frozen["token_start"] == 0 and frozen["token_stop"] == 2048
            assert frozen["prefix2048_sha256"] == hashlib.sha256(prefix.tobytes()).hexdigest()
            assert np.array_equal(tokens[slot], prefix)
            selected.append(record["article_id"])
    assert len(selected) == data["article_count"] == data["windows"]
    assert len(set(selected)) == len(selected) and data["lengths"] == [2048]
    assert (data["input_tokens"], data["scored_targets"], data["affected_targets"]) == (
            len(selected) * 2048, len(selected) * 2047, len(selected) * 1919)
    assert data["minimum_confirmatory_documents_met"] == (len(selected) >= 40)
    for name, digest in data["source_sha256"].items():
        assert sha(args.input / "sources" / name) == digest
    prior = completion["data_preparation_wall_seconds"]
    if prior + time.monotonic() - started >= 10800:
        raise TimeoutError("held-out data/audit cumulative10800-second ceiling reached")
    report = {"status": "heldout_context_data_verified", "created_utc": datetime.now(timezone.utc).isoformat(),
              "data_completion_sha256": sha(args.input / "completed.json"), "audit_source_sha256": sha(__file__),
              "inventory_articles_verified": len(boundaries), "eligible_articles_verified": len(selected),
              "input_tokens": len(selected) * 2048, "scored_targets": len(selected) * 2047,
              "minimum_confirmatory_documents_met": len(selected) >= 40,
              "audit_wall_seconds": time.monotonic() - started,
              "cumulative_gate_seconds": prior + time.monotonic() - started,
              "scope": "Same-operator independent delimiter/whole-token/prefix/eligibility reconstruction; no production imports or model output"}
    assert Path(__file__).read_bytes() == source
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if CONTEXT:
            (CONTEXT["output"] / "failed.json").write_text(json.dumps({
                "status": "heldout_data_audit_failed", "type": type(error).__name__, "message": str(error),
                "cumulative_gate_seconds": CONTEXT["prior"] + time.monotonic() - CONTEXT["started"]}, indent=2) + "\n")
        raise
