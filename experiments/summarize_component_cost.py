"""Describe every audited Spark component case without selecting seeds or blocks."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(path):
    completed = read(path / 'completed.json')
    assert sha(path / 'completed.json') == (path / 'completed.sha256').read_text().strip()
    for name, expected in completed['artifact_sha256'].items():
        file = path / name
        assert sha(file) == expected, name
        assert file.stat().st_size == completed['artifact_bytes'][name], name
    return completed


def distribution(name, values):
    assert len(values) == 7
    return {name + '_blocks': values, name + '_median': median(values),
            name + '_min': min(values), name + '_max': max(values)}


def table(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'measurement', 'audit', 'dispatch', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('fresh descriptive output required')
    verify(args.preflight)
    verify(args.measurement)
    audit = read(args.audit / 'audit.json')
    terminal = read(args.dispatch / 'finished.json')
    assert audit['status'] == 'native_component_cost_independently_verified'
    assert audit['samples_verified'] == 378000
    assert audit['measurement_completion_sha256'] == sha(args.measurement / 'completed.json')
    assert audit['preflight_completion_sha256'] == sha(args.preflight / 'completed.json')
    assert terminal['independent_audit_sha256'] == sha(args.audit / 'audit.json')
    assert terminal['combined_outer_seconds'] < 3600
    config = read(args.measurement / 'config.json')
    cases = config['cases']
    summaries = read(args.measurement / 'summaries.json')
    phases = {(r['block'], r['case'], r['phase']): r for r in summaries}
    storage = {(r['block'], r['case']): r for r in read(args.measurement / 'storage.json')}
    ratios = {(r['block'], r['case']): r for r in read(args.measurement / 'ratios.json')}
    assert len(cases) == 78 and len(phases) == 1890
    methods = list(dict.fromkeys(method for _, _, method in cases))
    lookup = {(length, prefix, method): i for i, (length, prefix, method) in enumerate(cases)}
    complete_rows, phase_rows, storage_rows = [], [], []
    for case, (length, prefix, method) in enumerate(cases):
        identity = {'case': case, 'length': length, 'prefix': prefix, 'method': method}
        full = lookup[length, prefix, 'native_full']
        view = lookup[length, prefix, 'recency_view']
        for phase in dict.fromkeys(r['phase'] for r in summaries if r['case'] == case):
            for clock in ('wall', 'event'):
                values = [phases[block, case, phase][clock]['median_ms'] for block in range(7)]
                p95 = [phases[block, case, phase][clock]['p95_ms'] for block in range(7)]
                row = {**identity, 'phase': phase, 'clock': clock,
                       **distribution('milliseconds', values), **distribution('p95_milliseconds', p95)}
                phase_rows.append(row)
                if phase == 'complete':
                    full_ratios = [values[b] / phases[b, full, phase][clock]['median_ms'] for b in range(7)]
                    view_ratios = [values[b] / phases[b, view, phase][clock]['median_ms'] for b in range(7)]
                    assert all(abs(full_ratios[b] - ratios[b, case][clock + '_median_ratio_to_full']) < 1e-12
                               for b in range(7))
                    complete_rows.append({**row, **distribution('ratio_full', full_ratios),
                                          **distribution('ratio_recency_view', view_ratios)})
        stored = [storage[block, case] for block in range(7)]
        unique = {r['unique_storage_bytes'] for r in stored}
        logical = {r['logical_payload_bytes'] for r in stored}
        assert len(unique) == len(logical) == 1
        row = {**identity, 'unique_storage_bytes': unique.pop(), 'logical_payload_bytes': logical.pop()}
        for name in ('warm_allocated_bytes', 'warm_reserved_bytes', 'peak_allocated_bytes',
                     'peak_reserved_bytes', 'transient_allocated_increase_bytes', 'setup_process_seconds'):
            row.update(distribution(name, [r[name] for r in stored]))
        storage_rows.append(row)
    assert len(complete_rows) == 156 and len(phase_rows) == 540 and len(storage_rows) == 78
    family_ranges = {}
    for family in ('p2', 'p3', 'coarsened_p2'):
        selected = [r for r in complete_rows if r['method'].endswith('_' + family)]
        if family == 'p2':
            selected = [r for r in selected if not r['method'].endswith('_coarsened_p2')]
        assert len(selected) == 36
        family_ranges[family] = {}
        for clock in ('wall', 'event'):
            subset = [r for r in selected if r['clock'] == clock]
            family_ranges[family][clock] = {
                'matched_block_ratios': 126,
                'minimum_ratio_to_full': min(r['ratio_full_min'] for r in subset),
                'maximum_ratio_to_full': max(r['ratio_full_max'] for r in subset),
                'all_complete_calls_slower_than_full': all(r['ratio_full_min'] > 1 for r in subset)}
    args.output.mkdir(parents=True)
    table(args.output / 'complete-cases.csv', complete_rows)
    table(args.output / 'all-phases.csv', phase_rows)
    table(args.output / 'storage-cases.csv', storage_rows)
    write(args.output / 'summary.json', {
        'status': 'complete_descriptive_summary_of_independently_audited_component_costs',
        'complete_case_clock_rows': 156, 'phase_case_clock_rows': 540, 'storage_case_rows': 78,
        'methods': methods, 'hardware_blocks': 7, 'raw_samples': 378000,
        'family_ranges': family_ranges,
        'allocation_history': terminal,
        'quality_family_screens': config['quality_family_screens'],
        'scope': 'Warm single-component replay; medians/ranges describe seven hardware blocks, not confidence intervals. All real KV retained. No cached-model throughput or compression claim.'})
    (args.output / 'summary_source.py').write_bytes(Path(__file__).read_bytes())
    files = sorted(p for p in args.output.iterdir() if p.is_file())
    write(args.output / 'manifest.json', {
        'source_sha256': sha(__file__), 'measurement_completion_sha256': sha(args.measurement / 'completed.json'),
        'preflight_completion_sha256': sha(args.preflight / 'completed.json'),
        'independent_audit_sha256': sha(args.audit / 'audit.json'),
        'dispatch_terminal_sha256': sha(args.dispatch / 'finished.json'),
        'artifact_sha256': {p.name: sha(p) for p in files},
        'artifact_bytes': {p.name: p.stat().st_size for p in files}})
    print(json.dumps(family_ranges, indent=2))


if __name__ == '__main__':
    main()
