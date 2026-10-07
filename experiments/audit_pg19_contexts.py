"""Independent full PG-19 object/text/token/selection reconstruction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'metadata', 'accepted', 'heldout', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError('fresh independent data audit required')
    a.output.mkdir(parents=True)
    (a.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    completion = json.loads((a.data / 'completed.json').read_text())
    assert sha(a.data / 'completed.json') == (a.data / 'completed.sha256').read_text().strip()
    prior = completion['cumulative_gate_seconds']
    def guard():
        if prior + time.monotonic() - started >= 10800: raise TimeoutError('native PG-19 cumulative10800s cap reached')
    try:
        for name, h in completion['artifact_sha256'].items():
            assert sha(a.data / name) == h and (a.data / name).stat().st_size == completion['artifact_bytes'][name]
        freeze = json.loads((a.data / 'production_freeze.json').read_text())
        data = json.loads((a.data / 'manifest.json').read_text())
        assert sha(a.data / 'prospective_protocol.txt') == data['protocol_sha256'] == freeze['protocol_sha256']
        assert sha(a.data / 'production_freeze.json') == data['freeze_sha256']
        for name, h in freeze['source_sha256'].items(): assert sha(a.data / 'sources' / name) == h
        assert data['model_revision'] == data['tokenizer_revision'] == 'f8027fd0eaeea54caa13c31d31b9fdc459c38b49'
        assert data['split'] == 'test' and not data['model_output_read']
        assert data['hf_dataset_revision'] == '4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b'
        assert sha(a.metadata / 'test_files.txt') == 'c84c08139695f3312df83239a1a41e7b9cde1baf7c08bfcf230ae09eaaf18d8c'
        assert sha(a.metadata / 'metadata.csv') == 'fbb2fdb48522927b2e16aa52950f2afeb83c6fa8fed45f0c3dd834e9bc9b43b9'
        paths = sorted((a.metadata / 'test_files.txt').read_text().splitlines())
        assert len(paths) == 100 and [r['path'] for r in data['inventory']] == paths
        with (a.metadata / 'metadata.csv').open(newline='') as f: meta = {r[0]: r[1:] for r in csv.reader(f)}
        accepted = json.loads((a.accepted / 'manifest.json').read_text())
        previous = json.loads((a.heldout / 'manifest.json').read_text())
        assert sha(a.accepted / 'completed.json') == '033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454'
        assert sha(a.heldout / 'completed.json') == '7ee2701ddb4800895ec20a98280b742dd5848f1d3b665efe9c3a20c44e996630'
        forbidden = {r['text_sha256'] for s in ('train', 'validation') for r in accepted['inventory'][s]}
        forbidden.update(r['text_sha256'] for r in previous['inventory'])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(data['model'], revision=data['tokenizer_revision'])
        eligible, prefixes, seen = [], {}, set()
        for index, record in enumerate(data['inventory']):
            guard(); path = paths[index]
            assert re.fullmatch(r'test/[0-9]+\.txt', path) and Path(path).stem == record['book_id']
            assert [record['title'], record['publication_date'], record['gutenberg_url']] == meta[record['book_id']]
            expected_url = 'https://storage.googleapis.com/deepmind-gutenberg/' + path
            assert record['url'] == expected_url
            generation = record['object_headers']['x-goog-generation']; assert generation.isdigit()
            with urllib.request.urlopen(expected_url + '?generation=' + generation, timeout=60) as response:
                remote = response.read(); assert response.headers['x-goog-generation'] == generation
            raw = (a.data / 'books' / Path(path).name).read_bytes()
            assert raw == remote and len(raw) == record['bytes']
            h = hashlib.sha256(raw).hexdigest(); assert h == record['text_sha256']
            ids = np.asarray(tokenizer(raw.decode('utf-8'), add_special_tokens=False)['input_ids'], dtype='<i8')
            assert len(ids) == record['whole_tokens'] and hashlib.sha256(ids.tobytes()).hexdigest() == record['whole_token_sha256']
            reasons = []
            if h in forbidden: reasons.append('previous_wikitext_exact_text')
            if h in seen: reasons.append('repeated_pg19_exact_text')
            if len(ids) < 4096: reasons.append('shorter_than4096')
            seen.add(h)
            assert record['exclusion_reasons'] == reasons and record['eligible'] == (len(reasons) == 0)
            if not reasons: eligible.append(index); prefixes[index] = ids[:4096].copy()
            if (index + 1) % 10 == 0: print(json.dumps({'objects_tokens_verified': index + 1}), flush=True)
        rng = np.random.Generator(np.random.PCG64(271904))
        order = rng.permutation(np.arange(100)).tolist(); assert order == data['permutation']
        selected = [i for i in order if i in set(eligible)][:40]
        assert len(selected) == 40 and [r['inventory_index'] for r in data['books']] == selected
        with np.load(a.data / 'tokens.npz', allow_pickle=False) as r:
            assert r['tokens4096'].shape == (40, 4096) and r['tokens2048'].shape == (40, 2048)
            for slot, index in enumerate(selected):
                book = data['books'][slot]
                for name, value in data['inventory'][index].items(): assert book[name] == value
                assert np.array_equal(r['tokens4096'][slot], prefixes[index])
                assert np.array_equal(r['tokens2048'][slot], prefixes[index][:2048])
                for length in (2048, 4096): assert book[f'prefix{length}_sha256'] == hashlib.sha256(prefixes[index][:length].tobytes()).hexdigest()
        assert data['input_tokens'] == 40 * (2048 + 4096) == 245760
        assert data['scored_targets'] == 40 * (2047 + 4095) == 245680
        guard()
        report = {'status': 'pg19_data_independently_verified', 'data_completion_sha256': sha(a.data / 'completed.json'),
                  'audit_source_sha256': sha(__file__), 'inventory_objects_retrieved_at_recorded_generation': 100,
                  'whole_book_texts_and_token_hashes_verified': 100, 'eligible_books': len(eligible), 'selected_books': 40,
                  'nested_prefixes_verified': 80, 'input_tokens': 245760, 'scored_targets': 245680,
                  'audit_wall_seconds': time.monotonic() - started, 'cumulative_gate_seconds': prior + time.monotonic() - started,
                  'scope': 'No production imports; full pinned-generation/text/token/eligibility/permutation reconstruction; no model output'}
        (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)
    except Exception as error:
        (a.output / 'failed.json').write_text(json.dumps({'status': 'pg19_independent_data_audit_failed',
             'message': str(error), 'type': type(error).__name__, 'cumulative_gate_seconds': prior + time.monotonic() - started}, indent=2) + '\n')
        raise


if __name__ == '__main__': main()
