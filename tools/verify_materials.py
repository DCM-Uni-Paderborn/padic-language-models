"""Verify the publication package against its file manifest without modifying it."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    entries = json.loads((root / "inputs/file-manifest.json").read_text())
    for relative, expected in entries.items():
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("missing or unsafe file: " + relative)
        content = path.read_bytes()
        if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise ValueError("content differs: " + relative)
    return {"verified_files": len(entries), "verified_bytes": sum(v["bytes"] for v in entries.values())}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify(p.parse_args().root)))
