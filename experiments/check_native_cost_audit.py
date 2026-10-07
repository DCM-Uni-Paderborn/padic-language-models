"""Synthetic structural tests of the cost auditor; no GPU measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n')


def seal(path, status, extra=None):
    files = sorted(p for p in path.iterdir() if p.is_file() and p.name not in ('completed.json', 'completed.sha256'))
    write(path / 'completed.json', {'status': status, 'cumulative_cost_seconds': 1.,
          'artifact_sha256': {p.name: sha(p) for p in files}, 'artifact_bytes': {p.name: p.stat().st_size for p in files}, **(extra or {})})
    (path / 'completed.sha256').write_text(sha(path / 'completed.json') + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    if a.output.exists(): raise FileExistsError('fresh structural fixture directory required')
    a.output.mkdir(parents=True); pre, run = a.output / 'preflight', a.output / 'measurement'; pre.mkdir(); run.mkdir()
    methods = ['native_full', 'recency', 'uniform'] + [f's{s}_{k}' for s in (17, 29, 43) for k in ('p2', 'p3', 'coarsened_p2')] + ['recency_view']
    cases = [[length, prefix, method] for length in (2048, 4096) for prefix in (length // 2, 3 * length // 4, length) for method in methods]
    sources = {}
    for label, name, source in (
        ('measurement', 'measurement_source.py', ROOT / 'experiments/benchmark_native_component.py'),
        ('native', 'native_source.py', ROOT / 'src/padic_lm/native_decode.py'),
        ('protocol', 'protocol.txt', ROOT / 'research/native-component-cost-protocol.md')):
        sources[label] = sha(source)
        for folder in (pre, run): (folder / name).write_bytes(source.read_bytes())
    config = {'source_sha256': sources, 'input_sha256': {'scope': 'synthetic structural fixture'}, 'prior_cost_seconds': 1.,
              'blocks': 7, 'warmups_per_phase': 20, 'wall_samples_per_phase': 100, 'event_samples_per_phase': 100,
              'order_pcg64_seed': 804211, 'backend': 'forced_FLASH_ATTENTION', 'tf32': False, 'cases': cases}
    for folder in (pre, run): write(folder / 'config.json', config)
    write(pre / 'checks.json', [{'length': l, 'prefix': pfx, 'method': m, 'passed': True} for l, pfx, m in cases])
    seal(pre, 'native_component_preflight_passed_no_latency_measurement')
    rng = np.random.Generator(np.random.PCG64(804211)); orders = [rng.permutation(78).tolist() for _ in range(7)]
    write(run / 'case_orders.json', orders); summaries, storage, auxiliary = [], [], []
    widths = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8, 'torch.int64': 8, 'torch.uint8': 1, 'torch.bool': 1}
    with (run / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('block', 'order', 'case', 'length', 'prefix', 'method', 'phase', 'clock', 'sample', 'milliseconds')); writer.writeheader()
        for block, order in enumerate(orders):
            for ordinal, case in enumerate(order):
                length, prefix, method = cases[case]
                base = {'block': block, 'case': case, 'length': length, 'prefix': prefix, 'method': method}
                phases = ['complete'] + (['address_update'] if method.startswith('s') else [])
                if method == 'recency_view': phases += ['readout_only']
                elif method != 'native_full': phases += ['selection', 'known_readout']
                for phase in phases:
                    values = {}
                    for clock in ('wall', 'event'):
                        samples = [1. + case / 100 + block / 1000 + phases.index(phase) / 100 + i / 10000 + (0.1 if clock == 'wall' else 0.) for i in range(100)]
                        for i, value in enumerate(samples): writer.writerow({**base, 'order': ordinal, 'phase': phase, 'clock': clock, 'sample': i, 'milliseconds': value})
                        values[clock] = {'median_ms': float(np.median(samples)), 'p95_ms': float(np.quantile(samples, .95, method='linear'))}
                    summaries.append({**base, 'phase': phase, **values})
                shapes = {'query': ([15, 1, 64], 'torch.bfloat16'), 'physical_keys': ([5, prefix, 64], 'torch.bfloat16'),
                          'physical_values': ([5, prefix, 64], 'torch.bfloat16'), 'current_key': ([5, 1, 64], 'torch.bfloat16'),
                          'current_value': ([5, 1, 64], 'torch.bfloat16'), 'output_weight': ([960, 960], 'torch.bfloat16')}
                if method.startswith('s'):
                    cuts, bins = (26, 27) if method.endswith('_p3') else (15, 16)
                    bridge = [([15, 64, 2], 'torch.float32'), ([15, 2], 'torch.float32'), ([15, 64, 2], 'torch.float32'),
                              ([15, 2], 'torch.float32'), ([15, 2], 'torch.float64'), ([15, 2, 2], 'torch.float64'),
                              ([15, 2, cuts], 'torch.float64'), ([2], 'torch.int64'), ([2, cuts], 'torch.bool'),
                              ([2, bins], 'torch.int64'), ([1, 1, 2], 'torch.int64')]
                    shapes.update({f'bridge_{i}': t for i, t in enumerate(bridge)})
                    shapes.update(key_codes=([15, prefix], 'torch.uint8'), query_code=([15], 'torch.uint8'), lookup=([256, 256], 'torch.uint8'))
                tensors = [{'name': n, 'shape': s, 'dtype': d, 'logical_bytes': math.prod(s) * widths[d],
                            'storage_id': n, 'storage_bytes': math.prod(s) * widths[d]} for n, (s, d) in shapes.items()]
                size = sum(t['logical_bytes'] for t in tensors); warm = size + 65536; peak = warm + 131072
                storage.append({**base, 'tensors': tensors, 'logical_payload_bytes': size, 'unique_storage_bytes': size,
                                'warm_allocated_bytes': warm, 'warm_reserved_bytes': warm + 1048576,
                                'peak_allocated_bytes': peak, 'peak_reserved_bytes': warm + 1048576,
                                'transient_allocated_increase_bytes': 131072, 'setup_process_seconds': .1})
                if method not in ('native_full', 'recency_view'): auxiliary.append({**base, 'shape': [5, 128], 'dtype': 'torch.int64', 'logical_bytes': 5120,
                       'storage_bytes': 5120 if method.startswith('s') else 1024, 'allocated_bytes_with_diagnostic_ids': warm + 5120})
    complete = {(r['block'], r['case']): r for r in summaries if r['phase'] == 'complete'}
    ratios = []
    for (block, case), row in complete.items():
        base = complete[block, case - methods.index(row['method'])]
        ratios.append({**{n: row[n] for n in ('block', 'case', 'length', 'prefix', 'method')},
                       'wall_median_ratio_to_full': row['wall']['median_ms'] / base['wall']['median_ms'],
                       'event_median_ratio_to_full': row['event']['median_ms'] / base['event']['median_ms']})
    for name, value in (('summaries', summaries), ('ratios', ratios), ('storage', storage), ('phase_auxiliary', auxiliary)): write(run / (name + '.json'), value)
    tel = {'measurement_pid': 4242, 'compute_processes': '4242, synthetic_fixture\n', 'other_compute_process_observed': False}
    write(run / 'initial_telemetry.json', {**tel, 'compute_processes': '', 'label': 'before_cuda_work'})
    write(run / 'telemetry.json', [{**tel, 'label': f'{side}_block_{i}'} for i in range(7) for side in ('before', 'after')])
    def seal_run(): seal(run, 'native_component_measurement_complete_pending_independent_audit', {'preflight_completion_sha256': sha(pre / 'completed.json')})
    def audit(name, success):
        output = a.output / name
        result = subprocess.run([sys.executable, str(ROOT / 'experiments/audit_native_component.py'), '--measurement', str(run),
                                 '--preflight', str(pre), '--output', str(output)], capture_output=True, text=True)
        (a.output / (name + '.log')).write_text(result.stdout + result.stderr)
        assert (result.returncode == 0) is success
        return {'case': name, 'expected_pass': success, 'returncode': result.returncode, 'passed': True}
    seal_run(); tests = [audit('consistent_fixture', True)]
    pristine = (run / 'summaries.json').read_bytes(); summaries[0]['wall']['median_ms'] += .25
    write(run / 'summaries.json', summaries); seal_run(); tests.append(audit('altered_summary', False))
    (run / 'summaries.json').write_bytes(pristine)
    with (run / 'samples.csv').open('rb+') as f:
        f.seek(-2, 2); position = f.tell()
        while position > 0:
            f.seek(position); b = f.read(1)
            if b == b'\n': f.truncate(position + 1); break
            position -= 1
    seal_run(); tests.append(audit('missing_sample', False))
    write(a.output / 'selfcheck.json', {'status': 'native_cost_audit_structural_selfcheck_passed', 'tests': tests,
          'auditor_sha256': sha(ROOT / 'experiments/audit_native_component.py'), 'selfcheck_source_sha256': sha(__file__),
          'scope': 'Synthetic metadata/clocks only;378000 prescribed rows checked; no measured GPU cost or numerical preflight'})


if __name__ == '__main__': main()
