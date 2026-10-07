"""Freeze all eligible official-test article prefixes after development verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from padic_lm import corpus, heldout


def verify(directory, expected):
    if heldout.digest(directory / "completed.json") != expected:
        raise ValueError("parent completion differs")
    record = json.loads((directory / "completed.json").read_text())
    for name, digest in record["artifact_sha256"].items():
        path = directory / name
        if heldout.digest(path) != digest or path.stat().st_size != record["artifact_bytes"][name]:
            raise ValueError(f"parent artifact differs: {name}")


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("accepted", "development", "development-audit", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError("fresh held-out data directory required")
    verify(args.accepted, heldout.ACCEPTED_HASH)
    if (heldout.digest(args.development / "completed.json") != heldout.DEVELOPMENT_HASH
            or heldout.digest(args.development_audit) != heldout.DEVELOPMENT_AUDIT_HASH
            or heldout.digest(args.protocol) != heldout.PROTOCOL_HASH):
        raise ValueError("verified development or frozen test protocol differs")
    audit = json.loads(args.development_audit.read_text())
    if audit["status"] != "shared_budget_audit_passed" or audit["windows_verified"] != 86:
        raise ValueError("requires completed development verification before test access")
    accepted = json.loads((args.accepted / "manifest.json").read_text())
    if (accepted["dataset"], accepted["dataset_config"], accepted["dataset_revision"]) != (
            "Salesforce/wikitext", "wikitext-2-raw-v1", "b08601e04326c79dfdd32d625aee71d232d685c3"):
        raise ValueError("pinned dataset differs")
    forbidden = {r["text_sha256"] for split in ("train", "validation") for r in accepted["inventory"][split]}
    args.output.mkdir(parents=True)
    (args.output / "sources").mkdir()
    paths = [Path(__file__), Path(corpus.__file__), Path(heldout.__file__)]
    sources = {path.name: path.read_bytes() for path in paths}
    for name, content in sources.items():
        (args.output / "sources" / name).write_bytes(content)
    protocol = args.protocol.read_bytes()
    (args.output / "prospective_protocol.txt").write_bytes(protocol)
    write(args.output / "test_access_record.json", {"first_test_access_utc": datetime.now(timezone.utc).isoformat(),
          "protocol_sha256": heldout.PROTOCOL_HASH, "passed_development_audit_sha256": heldout.DEVELOPMENT_AUDIT_HASH,
          "scope": "Text/token inventory access only; no model output"})
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(accepted["model"], revision=accepted["tokenizer_revision"])
        rows = [r["text"] for r in load_dataset(accepted["dataset"], accepted["dataset_config"],
                  revision=accepted["dataset_revision"], split="test", streaming=True)]
        records, selected, ids = heldout.inventory(corpus.articles(rows),
                      lambda text: tokenizer(text, add_special_tokens=False)["input_ids"], forbidden)
        if not selected:
            raise ValueError("no eligible test articles")
        np.savez_compressed(args.output / "tokens.npz", tokens2048=ids)
        manifest = {"status": "heldout_contexts_frozen", "created_utc": datetime.now(timezone.utc).isoformat(),
             "model": accepted["model"], "model_revision": accepted["model_revision"],
             "tokenizer_revision": accepted["tokenizer_revision"], "dataset": accepted["dataset"],
             "dataset_config": accepted["dataset_config"], "dataset_revision": accepted["dataset_revision"],
             "split": "test", "accepted_completion_sha256": heldout.ACCEPTED_HASH,
             "development_completion_sha256": heldout.DEVELOPMENT_HASH,
             "development_audit_sha256": heldout.DEVELOPMENT_AUDIT_HASH, "protocol_sha256": heldout.PROTOCOL_HASH,
             "inventory": records, "articles": selected, "article_count": len(selected), "windows": len(selected),
             "lengths": [2048], "input_tokens": int(ids.size), "scored_targets": len(selected) * 2047,
             "affected_targets": len(selected) * 1919, "minimum_confirmatory_documents_met": len(selected) >= 40,
             "test_split_read": True, "model_output_read": False,
             "tokens_sha256": heldout.digest(args.output / "tokens.npz"),
             "row_count": len(rows), "source_sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
             "scope": "All eligible source-order first2048-token test articles; exact text exclusions; no quality selection"}
        artifacts = sorted(path for path in args.output.rglob("*") if path.is_file())
        hashes = {str(path.relative_to(args.output)): heldout.digest(path) for path in artifacts}
        sizes = {str(path.relative_to(args.output)): path.stat().st_size for path in artifacts}
        manifest["data_preparation_wall_seconds"] = time.monotonic() - started
        write(args.output / "manifest.json", manifest)
        hashes["manifest.json"] = heldout.digest(args.output / "manifest.json")
        sizes["manifest.json"] = (args.output / "manifest.json").stat().st_size
        if args.protocol.read_bytes() != protocol or any(path.read_bytes() != sources[path.name] for path in paths):
            raise RuntimeError("data-preparation source or protocol changed")
        if time.monotonic() - started >= 10800:
            raise TimeoutError("held-out cumulative10800-second preparation ceiling reached")
        write(args.output / "completed.json", {"status": manifest["status"], "artifact_sha256": hashes, "artifact_bytes": sizes,
               "data_preparation_wall_seconds": time.monotonic() - started})
        (args.output / "completed.sha256").write_text(heldout.digest(args.output / "completed.json") + "\n")
        print(json.dumps({k: manifest[k] for k in ("status", "article_count", "scored_targets", "minimum_confirmatory_documents_met")}), flush=True)
    except Exception as error:
        (args.output / "completed.json").unlink(missing_ok=True)
        (args.output / "completed.sha256").unlink(missing_ok=True)
        write(args.output / "failed.json", {"status": "heldout_data_failed", "type": type(error).__name__,
                "message": str(error), "data_preparation_wall_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
