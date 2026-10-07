"""Independent fixed cost sample/order, summaries and tensor-storage audit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def verify(path, guard):
    assert sha(path / 'completed.json') == (path / 'completed.sha256').read_text().strip()
    value = json.loads((path / 'completed.json').read_text())
    for name, fingerprint in value['artifact_sha256'].items():
        guard(); assert sha(path / name) == fingerprint
        assert (path / name).stat().st_size == value['artifact_bytes'][name]
    return value


def percentile(values, probability):
    values = sorted(values); location = (len(values) - 1) * probability
    left = int(math.floor(location)); fraction = location - left
    return values[left] * (1 - fraction) + values[min(left + 1, len(values) - 1)] * fraction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('measurement', 'preflight', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); prior = 0
    if a.output.exists(): raise FileExistsError('fresh independent cost audit required')
    a.output.mkdir(parents=True); (a.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    def guard():
        if prior + time.monotonic() - started >= 3600: raise TimeoutError('native component cumulative3600s cost cap reached')
    try:
        parent = verify(a.measurement, guard); prior = parent['cumulative_cost_seconds']; guard()
        assert parent['status'] == 'native_component_measurement_complete_pending_independent_audit'
        assert parent['preflight_completion_sha256'] == sha(a.preflight / 'completed.json')
        previous = verify(a.preflight, guard)
        assert previous['status'] == 'native_component_preflight_passed_no_latency_measurement'
        config = json.loads((a.measurement / 'config.json').read_text())
        frozen = json.loads((a.preflight / 'config.json').read_text())
        assert config['source_sha256'] == frozen['source_sha256'] and config['input_sha256'] == frozen['input_sha256']
        for label, name in (('measurement', 'measurement_source.py'), ('native', 'native_source.py'), ('protocol', 'protocol.txt')):
            assert sha(a.measurement / name) == config['source_sha256'][label]
        assert config['source_sha256']['native'] == '2c07cbe6e7a331a0b8fd5b128eb4ef61d5fcaf4bfef16fa192b2c3cb4b344231'
        assert config['prior_cost_seconds'] == previous['cumulative_cost_seconds']
        assert config['blocks'] == 7 and config['warmups_per_phase'] == 20
        assert config['wall_samples_per_phase'] == config['event_samples_per_phase'] == 100
        assert config['order_pcg64_seed'] == 804211 and config['backend'] == 'forced_FLASH_ATTENTION' and config['tf32'] is False
        methods = ['native_full', 'recency', 'uniform'] + [f's{s}_{kind}' for s in (17, 29, 43) for kind in ('p2', 'p3', 'coarsened_p2')] + ['recency_view']
        cases = [[length, prefix, method] for length in (2048, 4096)
                 for prefix in (length // 2, 3 * length // 4, length) for method in methods]
        assert config['cases'] == cases and len(cases) == 78
        checks = json.loads((a.preflight / 'checks.json').read_text())
        assert [(r['length'], r['prefix'], r['method']) for r in checks] == [tuple(c) for c in cases]
        assert all(r['passed'] is True for r in checks)
        rng = np.random.Generator(np.random.PCG64(804211))
        orders = [rng.permutation(78).tolist() for _ in range(7)]
        assert json.loads((a.measurement / 'case_orders.json').read_text()) == orders
        groups, count = {}, 0
        with (a.measurement / 'samples.csv').open(newline='') as f:
            reader = csv.DictReader(f)
            for block, order in enumerate(orders):
                guard()
                for ordinal, case in enumerate(order):
                    length, prefix, method = cases[case]
                    phases = ['complete'] + (['address_update'] if method.startswith('s') else [])
                    if method == 'recency_view': phases += ['readout_only']
                    elif method != 'native_full': phases += ['selection', 'known_readout']
                    for phase in phases:
                        for clock in ('wall', 'event'):
                            key = block, case, phase, clock; values = []
                            for sample in range(100):
                                row = next(reader)
                                assert [int(row[n]) for n in ('block', 'order', 'case', 'length', 'prefix', 'sample')] == [block, ordinal, case, length, prefix, sample]
                                assert (row['method'], row['phase'], row['clock']) == (method, phase, clock)
                                value = float(row['milliseconds']); assert math.isfinite(value) and value > 0
                                values.append(value); count += 1
                            groups[key] = values
            assert next(reader, None) is None
        assert count == 378000 and len(groups) == 3780
        summaries = json.loads((a.measurement / 'summaries.json').read_text()); seen = set(); maximum = 0.
        complete = {}
        for row in summaries:
            case, block, phase = row['case'], row['block'], row['phase']
            assert (row['length'], row['prefix'], row['method']) == tuple(cases[case])
            assert (block, case, phase) not in seen; seen.add((block, case, phase))
            for clock in ('wall', 'event'):
                values = groups[block, case, phase, clock]
                for name, probability in (('median_ms', .5), ('p95_ms', .95)):
                    difference = abs(row[clock][name] - percentile(values, probability)); maximum = max(maximum, difference)
                    assert difference <= 1e-12
            if phase == 'complete': complete[block, case] = row
        assert len(seen) == 1890 and len(complete) == 546
        ratios = json.loads((a.measurement / 'ratios.json').read_text()); ratio_seen = set()
        for row in ratios:
            key = row['block'], row['case']; assert key not in ratio_seen; ratio_seen.add(key)
            actual = complete[key]; base_case = row['case'] - methods.index(row['method']); baseline = complete[row['block'], base_case]
            assert cases[base_case] == [row['length'], row['prefix'], 'native_full']
            for clock in ('wall', 'event'):
                expected = percentile(groups[*key, 'complete', clock], .5) / percentile(groups[row['block'], base_case, 'complete', clock], .5)
                assert abs(row[clock + '_median_ratio_to_full'] - expected) <= 1e-12
            assert (row['length'], row['prefix'], row['method']) == (actual['length'], actual['prefix'], actual['method'])
        assert len(ratio_seen) == 546
        storage = json.loads((a.measurement / 'storage.json').read_text()); storage_seen = set()
        widths = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8, 'torch.int64': 8, 'torch.uint8': 1, 'torch.bool': 1}
        for row in storage:
            guard(); key = row['block'], row['case']; assert key not in storage_seen; storage_seen.add(key)
            assert (row['length'], row['prefix'], row['method']) == tuple(cases[row['case']])
            unique = {}; logical = 0; tensors = {t['name']: t for t in row['tensors']}
            assert len(tensors) == len(row['tensors'])
            assert tensors['physical_keys']['shape'] == tensors['physical_values']['shape'] == [5, row['prefix'], 64]
            assert tensors['query']['shape'] == [15, 1, 64] and tensors['output_weight']['shape'] == [960, 960]
            assert tensors['current_key']['shape'] == tensors['current_value']['shape'] == [5, 1, 64]
            base_names = {'query', 'physical_keys', 'physical_values', 'current_key', 'current_value', 'output_weight'}
            assert all(tensors[name]['dtype'] == 'torch.bfloat16' for name in base_names)
            if row['method'].startswith('s'):
                ternary = row['method'].endswith('_p3'); cuts, bins = (26, 27) if ternary else (15, 16)
                bridge = [([15, 64, 2], 'torch.float32'), ([15, 2], 'torch.float32'),
                          ([15, 64, 2], 'torch.float32'), ([15, 2], 'torch.float32'),
                          ([15, 2], 'torch.float64'), ([15, 2, 2], 'torch.float64'),
                          ([15, 2, cuts], 'torch.float64'), ([2], 'torch.int64'),
                          ([2, cuts], 'torch.bool'), ([2, bins], 'torch.int64'), ([1, 1, 2], 'torch.int64')]
                for i, (shape, dtype) in enumerate(bridge):
                    assert tensors[f'bridge_{i}']['shape'] == shape and tensors[f'bridge_{i}']['dtype'] == dtype
                assert tensors['key_codes']['shape'] == [15, row['prefix']] and tensors['query_code']['shape'] == [15]
                assert tensors['lookup']['shape'] == [256, 256]
                assert all(tensors[name]['dtype'] == 'torch.uint8' for name in ('key_codes', 'query_code', 'lookup'))
                assert set(tensors) == base_names | {f'bridge_{i}' for i in range(11)} | {'key_codes', 'query_code', 'lookup'}
            else: assert set(tensors) == base_names
            for tensor in row['tensors']:
                payload = math.prod(tensor['shape']) * widths[tensor['dtype']]
                assert payload == tensor['logical_bytes'] and tensor['storage_bytes'] >= payload
                logical += payload
                if tensor['storage_id'] in unique: assert unique[tensor['storage_id']] == tensor['storage_bytes']
                unique[tensor['storage_id']] = tensor['storage_bytes']
            assert row['logical_payload_bytes'] == logical and row['unique_storage_bytes'] == sum(unique.values())
            assert row['warm_allocated_bytes'] >= row['unique_storage_bytes']
            assert row['warm_reserved_bytes'] >= row['warm_allocated_bytes']
            assert row['peak_allocated_bytes'] >= row['warm_allocated_bytes']
            assert row['peak_reserved_bytes'] >= max(row['warm_reserved_bytes'], row['peak_allocated_bytes'])
            assert row['transient_allocated_increase_bytes'] == row['peak_allocated_bytes'] - row['warm_allocated_bytes']
            assert math.isfinite(row['setup_process_seconds']) and row['setup_process_seconds'] > 0
        assert storage_seen == set(complete)
        auxiliary = json.loads((a.measurement / 'phase_auxiliary.json').read_text()); aux_seen = set()
        for row in auxiliary:
            key = row['block'], row['case']; assert key not in aux_seen; aux_seen.add(key)
            assert row['shape'] == [5, 128] and row['dtype'] == 'torch.int64' and row['logical_bytes'] == 5120
            assert row['storage_bytes'] == (5120 if row['method'].startswith('s') else 1024)
            assert (row['length'], row['prefix'], row['method']) == tuple(cases[row['case']])
        assert aux_seen == {key for key, row in complete.items() if row['method'] not in ('native_full', 'recency_view')}
        telemetry = json.loads((a.measurement / 'telemetry.json').read_text())
        assert [r['label'] for r in telemetry] == [f'{side}_block_{i}' for i in range(7) for side in ('before', 'after')]
        initial = json.loads((a.measurement / 'initial_telemetry.json').read_text())
        for row in [initial, *telemetry]:
            assert row['other_compute_process_observed'] is False
            assert all(int(line.split(',')[0].strip()) == row['measurement_pid'] for line in row['compute_processes'].splitlines())
        guard()
        report = {'status': 'native_component_cost_independently_verified', 'measurement_completion_sha256': sha(a.measurement / 'completed.json'),
                  'preflight_completion_sha256': sha(a.preflight / 'completed.json'), 'audit_source_sha256': sha(__file__),
                  'samples_verified': count, 'phase_summaries_verified': len(summaries), 'complete_ratios_verified': len(ratios),
                  'primary_storage_records_verified': len(storage), 'max_summary_difference_ms': maximum,
                  'audit_wall_seconds': time.monotonic() - started, 'cumulative_cost_seconds': prior + time.monotonic() - started,
                  'scope': 'No production imports; deterministic order/count checks, explicit order statistics and unique storage accounting; descriptive repeats only'}
        (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    except Exception as error:
        (a.output / 'failed.json').write_text(json.dumps({'status': 'native_component_cost_audit_failed', 'type': type(error).__name__,
              'message': str(error), 'cumulative_cost_seconds': prior + time.monotonic() - started}, indent=2) + '\n')
        raise


if __name__ == '__main__': main()
