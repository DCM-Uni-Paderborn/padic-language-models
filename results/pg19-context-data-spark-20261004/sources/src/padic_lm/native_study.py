"""Integrity and clocks for the independent frozen native PG-19 study."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'HuggingFaceTB/SmolLM2-360M'
REVISION = 'f8027fd0eaeea54caa13c31d31b9fdc459c38b49'
CAP = 10800


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(directory, expected=None):
    directory = Path(directory)
    digest = sha(directory / 'completed.json')
    if digest != (directory / 'completed.sha256').read_text().strip() or (expected is not None and digest != expected):
        raise ValueError('completion fingerprint differs')
    record = json.loads((directory / 'completed.json').read_text())
    for name, h in record['artifact_sha256'].items():
        if sha(directory / name) != h or (directory / name).stat().st_size != record['artifact_bytes'][name]:
            raise ValueError('artifact differs: ' + name)
    return record


def check_freeze(path):
    value = json.loads(Path(path).read_text())
    if value['status'] != 'native_pg19_production_frozen_before_book_access':
        raise ValueError('fresh prospective production freeze required')
    for name, h in value['source_sha256'].items():
        if sha(ROOT / name) != h: raise ValueError('frozen source differs: ' + name)
    if sha(ROOT / 'research/native-quality-pg19-protocol.md') != value['protocol_sha256']:
        raise ValueError('prospective native PG-19 protocol differs')
    return value


def snapshot(output, freeze_path):
    value = check_freeze(freeze_path)
    for name in value['source_sha256']:
        target = output / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    (output / 'production_freeze.json').write_bytes(Path(freeze_path).read_bytes())
    (output / 'prospective_protocol.txt').write_bytes((ROOT / 'research/native-quality-pg19-protocol.md').read_bytes())


def complete(output, status, cumulative_seconds, extra=None):
    files = sorted(p for p in output.rglob('*') if p.is_file())
    write(output / 'completed.json', {'status': status, 'cumulative_gate_seconds': cumulative_seconds,
          'artifact_sha256': {str(p.relative_to(output)): sha(p) for p in files},
          'artifact_bytes': {str(p.relative_to(output)): p.stat().st_size for p in files}, **(extra or {})})
    (output / 'completed.sha256').write_text(sha(output / 'completed.json') + '\n')


def guard(started, prior=0):
    if prior + time.monotonic() - started >= CAP: raise TimeoutError('native PG-19 cumulative10800s cap reached')


def failure(output, error, started, prior=0):
    (output / 'completed.json').unlink(missing_ok=True)
    (output / 'completed.sha256').unlink(missing_ok=True)
    write(output / 'failed.json', {'status': 'native_pg19_stage_failed', 'type': type(error).__name__,
                                  'message': str(error), 'cumulative_gate_seconds': prior + time.monotonic() - started})
