"""Freeze all eligible nested 512/2048-token development articles; no test data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from padic_lm import corpus
from evaluate_qk_bridge import DATA_HASH, verify_manifest, digest, json_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh output required")
    verify_manifest(args.accepted, DATA_HASH)
    accepted = json.loads((args.accepted / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(accepted["model"], revision=accepted["model_revision"])
    rows = [r["text"] for r in load_dataset(accepted["dataset"], accepted["dataset_config"],
            revision=accepted["dataset_revision"], split="validation", streaming=True)]
    docs = corpus.articles(rows)
    if len(docs) != len(accepted["inventory"]["validation"]):
        raise ValueError("accepted article inventory differs")
    training_hashes = {r["text_sha256"] for r in accepted["inventory"]["train"]}
    with np.load(args.accepted / "tokens.npz", allow_pickle=False) as raw:
        accepted_ids = raw["development"].copy()
    old = {r["article_id"]: accepted_ids[i] for i, r in enumerate(accepted["windows"]["development"]) if r["token_start"] == 0}
    selected = []
    ids = []
    for doc, record in zip(docs, accepted["inventory"]["validation"]):
        if (doc.start_row, doc.stop_row, doc.text_sha256) != (record["start_row"], record["stop_row"], record["text_sha256"]):
            raise ValueError("article boundary/text hash mismatch")
        tokens = tokenizer(doc.text, add_special_tokens=False)["input_ids"]
        if len(tokens) != record["tokens"]:
            raise ValueError("tokenized article size differs")
        if record["prior_overlap_excluded"] or record["text_sha256"] in training_hashes or len(tokens) < 2048:
            continue
        full = np.asarray(tokens, dtype=np.int64)
        if record["id"] not in old or not np.array_equal(full[:512], old[record["id"]]):
            raise ValueError("accepted first512-token prefix differs")
        ids.append(full[:2048])
        selected.append({"article_id": record["id"], "title": record["title"], "start_row": doc.start_row,
                         "stop_row": doc.stop_row, "text_sha256": doc.text_sha256,
                         "whole_article_tokens": len(tokens), "whole_token_sha256": hashlib.sha256(full.tobytes()).hexdigest(),
                         "token_start": 0, "token_stop": 2048,
                         "prefix2048_sha256": hashlib.sha256(full[:2048].tobytes()).hexdigest(),
                         "prefix512_sha256": hashlib.sha256(full[:512].tobytes()).hexdigest()})
    if len(selected) != 43:
        raise ValueError("all43 eligible articles required")
    args.output.mkdir(parents=True)
    (args.output / "sources").mkdir()
    for p in (Path(__file__), Path(corpus.__file__)):
        (args.output / "sources" / p.name).write_bytes(p.read_bytes())
    (args.output / "prospective_protocol.txt").write_bytes(args.protocol.read_bytes())
    np.savez_compressed(args.output / "tokens.npz", tokens2048=np.stack(ids), tokens512=np.stack(ids)[:, :512])
    manifest = {"status": "shared_contexts_frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
                "accepted_completion_sha256": DATA_HASH, "model": accepted["model"], "model_revision": accepted["model_revision"],
                "dataset_revision": accepted["dataset_revision"], "articles": selected, "lengths": [512, 2048],
                "article_count": 43, "windows": 86, "input_tokens": 110080, "scored_targets": 109994,
                "test_split_read": False, "accepted_prefixes_exact": True,
                "data_preparation_wall_seconds": time.monotonic() - started,
                "tokens_sha256": digest(args.output / "tokens.npz"), "protocol_sha256": digest(args.protocol)}
    json_write(args.output / "manifest.json", manifest)
    artifacts = sorted(p for p in args.output.rglob("*") if p.is_file())
    json_write(args.output / "completed.json", {"status": manifest["status"],
        "artifact_sha256": {str(p.relative_to(args.output)): digest(p) for p in artifacts},
        "artifact_bytes": {str(p.relative_to(args.output)): p.stat().st_size for p in artifacts}})
    (args.output / "completed.sha256").write_text(digest(args.output / "completed.json") + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "article_count", "scored_targets", "data_preparation_wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
