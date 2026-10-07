"""Inventory all official test books; freeze the predefined40 nested prefixes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from padic_lm.native_study import MODEL, REVISION, sha, write, snapshot, complete, guard, failure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('freeze', 'metadata', 'accepted', 'heldout', 'output'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError('fresh PG-19 data path required')
    a.output.mkdir(parents=True); snapshot(a.output, a.freeze)
    try:
        from transformers import AutoTokenizer
        (a.output / 'books').mkdir()
        meta = json.loads((a.metadata / 'manifest.json').read_text())
        freeze = json.loads((a.output / 'production_freeze.json').read_text())
        assert sha(a.metadata / 'manifest.json') == freeze['metadata_manifest_sha256']
        assert meta['status'] == 'metadata_only_no_book_text_read'
        for n, v in meta['artifacts'].items(): assert sha(a.metadata / n) == v['sha256']
        assert sha(a.metadata / 'test_files.txt') == 'c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c'
        files = sorted((a.metadata / 'test_files.txt').read_text().splitlines())
        assert len(files) == len(set(files)) == 100 and all(re.fullmatch(r'test/[0-9]+\.txt', n) for n in files)
        permutation = np.random.Generator(np.random.PCG64(271904)).permutation(100).tolist()
        write(a.output / 'test_access_record.json', {'first_book_text_access_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha(a.output / 'prospective_protocol.txt'), 'freeze_sha256': sha(a.freeze),
              'permutation_of_sorted_paths': permutation, 'model_output_read': False})
        accepted = json.loads((a.accepted / 'manifest.json').read_text())
        previous = json.loads((a.heldout / 'manifest.json').read_text())
        assert sha(a.accepted / 'completed.json') == '033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454'
        assert sha(a.heldout / 'completed.json') == '7ee2701ddb4800895ec20a98280b742dd5848f1d3b665efe9c3a20c44e996630'
        forbidden = {r['text_sha256'] for split in ('train', 'validation') for r in accepted['inventory'][split]}
        forbidden |= {r['text_sha256'] for r in previous['inventory']}
        with (a.metadata / 'metadata.csv').open(newline='') as f:
            metadata = {r[0]: {'title': r[1], 'publication_date': r[2], 'gutenberg_url': r[3]} for r in csv.reader(f)}
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        records, buffers, seen = [], {}, set()
        for index, path in enumerate(files):
            guard(started)
            url = 'https://storage.googleapis.com/deepmind-gutenberg/' + path
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
                headers = {k.lower(): v for k, v in response.headers.items()
                           if k.lower() in ('etag', 'x-goog-generation', 'x-goog-hash', 'content-length', 'last-modified')}
            assert 'x-goog-generation' in headers
            book_id = Path(path).stem
            (a.output / 'books' / Path(path).name).write_bytes(raw)
            text = raw.decode('utf-8')
            tokens = np.array(tokenizer(text, add_special_tokens=False)['input_ids'], dtype='<i8')
            h = hashlib.sha256(raw).hexdigest(); reasons = []
            if h in forbidden: reasons.append('previous_wikitext_exact_text')
            if h in seen: reasons.append('repeated_pg19_exact_text')
            if len(tokens) < 4096: reasons.append('shorter_than4096')
            seen.add(h)
            records.append({'book_id': book_id, 'path': path, 'url': url, **metadata[book_id],
                 'object_headers': headers, 'bytes': len(raw), 'text_sha256': h,
                 'whole_tokens': len(tokens), 'whole_token_sha256': hashlib.sha256(tokens.tobytes()).hexdigest(),
                 'eligible': not reasons, 'exclusion_reasons': reasons})
            if not reasons: buffers[index] = tokens[:4096].copy()
            if (index + 1) % 10 == 0: print(json.dumps({'books_inventoried': index + 1}), flush=True)
        chosen = [i for i in permutation if records[i]['eligible']][:40]
        if len(chosen) != 40:
            write(a.output / 'incomplete_inventory.json', {'inventory': records, 'selected_indices': chosen})
            raise ValueError('fewer than40 eligible books; no expansion/replacement')
        selected = [{**records[i], 'inventory_index': i, 'prefix4096_sha256': hashlib.sha256(buffers[i].tobytes()).hexdigest(),
                     'prefix2048_sha256': hashlib.sha256(buffers[i][:2048].tobytes()).hexdigest()} for i in chosen]
        ids = np.stack([buffers[i] for i in chosen])
        np.savez_compressed(a.output / 'tokens.npz', tokens4096=ids, tokens2048=ids[:, :2048])
        write(a.output / 'manifest.json', {'status': 'pg19_nested_contexts_frozen', 'model': MODEL, 'model_revision': REVISION,
             'tokenizer_revision': REVISION, 'split': 'test', 'hf_dataset_revision': meta['hf_revision'],
             'protocol_sha256': sha(a.output / 'prospective_protocol.txt'), 'freeze_sha256': sha(a.freeze),
             'metadata_manifest_sha256': sha(a.metadata / 'manifest.json'), 'inventory': records,
             'books': selected, 'book_count': 40, 'lengths': [2048, 4096], 'permutation': permutation,
             'input_tokens': 245760, 'scored_targets': 245680, 'model_output_read': False,
             'preparation_wall_seconds': time.monotonic() - started})
        guard(started); complete(a.output, 'pg19_nested_contexts_frozen', time.monotonic() - started)
        print(json.dumps({'status': 'pg19_nested_contexts_frozen', 'eligible_books': len(buffers),
                          'selected_books': 40, 'completion_sha256': sha(a.output / 'completed.json')}), flush=True)
    except Exception as error: failure(a.output, error, started); raise


if __name__ == '__main__': main()
