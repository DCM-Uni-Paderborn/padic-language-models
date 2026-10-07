"""Independent delimiter/token check of all nested development contexts."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "accepted", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh verification output required")
    for directory in (args.input, args.accepted):
        assert sha(directory / "completed.json") == (directory / "completed.sha256").read_text().strip()
        c = json.loads((directory / "completed.json").read_text())
        for name, digest in c["artifact_sha256"].items():
            assert sha(directory / name) == digest
            assert (directory / name).stat().st_size == c["artifact_bytes"][name]
    accepted = json.loads((args.accepted / "manifest.json").read_text())
    data = json.loads((args.input / "manifest.json").read_text())
    assert sha(args.accepted / "completed.json") == data["accepted_completion_sha256"] == "033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454"
    assert data["model_revision"] == accepted["model_revision"]
    assert sha(args.input / "prospective_protocol.txt") == data["protocol_sha256"]
    rows = [r["text"] for r in load_dataset(accepted["dataset"], accepted["dataset_config"],
            revision=accepted["dataset_revision"], split="validation", streaming=True)]
    boundaries = []
    for i, row in enumerate(rows):
        words = row.strip().split()
        if len(words) >= 3 and words[0] == words[-1] == "=" and words[1] != "=" and words[-2] != "=":
            boundaries.append(i)
    inventory = accepted["inventory"]["validation"]
    assert boundaries == [r["start_row"] for r in inventory]
    train_hashes = {r["text_sha256"] for r in accepted["inventory"]["train"]}
    tokenizer = AutoTokenizer.from_pretrained(accepted["model"], revision=accepted["tokenizer_revision"])
    selected = []
    with np.load(args.input / "tokens.npz", allow_pickle=False) as raw:
        short, long = raw["tokens512"].copy(), raw["tokens2048"].copy()
    assert short.shape == (43, 512) and long.shape == (43, 2048)
    assert np.array_equal(short, long[:, :512])
    for i, record in enumerate(inventory):
        stop = boundaries[i + 1] if i + 1 < len(boundaries) else len(rows)
        assert stop == record["stop_row"]
        text = "\n\n".join(rows[boundaries[i]:stop])
        assert hashlib.sha256(text.encode()).hexdigest() == record["text_sha256"]
        ids = np.asarray(tokenizer(text, add_special_tokens=False)["input_ids"], dtype=np.int64)
        assert len(ids) == record["tokens"]
        if boundaries[i] <= accepted["prior_last_consumed_row"] or record["text_sha256"] in train_hashes or len(ids) < 2048:
            continue
        index = len(selected)
        frozen = data["articles"][index]
        assert frozen["article_id"] == record["id"]
        assert (frozen["start_row"], frozen["stop_row"]) == (boundaries[i], stop)
        assert frozen["whole_token_sha256"] == hashlib.sha256(ids.tobytes()).hexdigest()
        assert np.array_equal(ids[:2048], long[index])
        assert frozen["prefix2048_sha256"] == hashlib.sha256(long[index].tobytes()).hexdigest()
        assert frozen["prefix512_sha256"] == hashlib.sha256(short[index].tobytes()).hexdigest()
        selected.append(record["id"])
    assert len(selected) == 43 and len(set(selected)) == 43 and data["test_split_read"] is False
    report = {"status": "shared_context_data_verified", "data_completion_sha256": sha(args.input / "completed.json"),
              "audit_source_sha256": sha(__file__), "articles_verified": 43, "nested_windows_verified": 86,
              "input_tokens": int(short.size + long.size), "scored_targets": 43 * (511 + 2047),
              "audit_wall_seconds": time.monotonic() - started,
              "scope": "Same-operator independent delimiter/token reconstruction; no production modules or test split"}
    args.output.mkdir(parents=True)
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
