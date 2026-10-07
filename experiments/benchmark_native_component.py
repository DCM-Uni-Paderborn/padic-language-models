"""Correctness-only preflight and fixed native component cost measurement."""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from padic_lm.native_decode import FiniteBridge, depth_lookup, selected_keys, fixed_keys, attention_component

METHODS = ('native_full', 'recency', 'uniform') + tuple(
    f's{s}_{k}' for s in (17, 29, 43) for k in ('p2', 'p3', 'coarsened_p2')) + ('recency_view',)
CASES = [(length, prefix, method) for length in (2048, 4096)
         for prefix in (length // 2, 3 * length // 4, length) for method in METHODS]
NATIVE_SHA = '2c07cbe6e7a331a0b8fd5b128eb4ef61d5fcaf4bfef16fa192b2c3cb4b344231'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path):
    with np.load(path, allow_pickle=False) as r: return {n: r[n] for n in r.files}


def verify(path, guard):
    value = json.loads((path / 'completed.json').read_text())
    assert sha(path / 'completed.json') == (path / 'completed.sha256').read_text().strip()
    for name, fingerprint in value['artifact_sha256'].items():
        guard(); assert sha(path / name) == fingerprint
        assert (path / name).stat().st_size == value['artifact_bytes'][name]
    return value


def finish(path, status, seconds, extra=None):
    files = sorted(p for p in path.rglob('*') if p.is_file())
    write(path / 'completed.json', {'status': status, 'cumulative_cost_seconds': seconds,
          'artifact_sha256': {str(p.relative_to(path)): sha(p) for p in files},
          'artifact_bytes': {str(p.relative_to(path)): p.stat().st_size for p in files}, **(extra or {})})
    (path / 'completed.sha256').write_text(sha(path / 'completed.json') + '\n')


def gpu(bits):
    return torch.tensor(np.ascontiguousarray(bits), device='cuda').view(torch.bfloat16)


def cpu_bits(tensor):
    return tensor.contiguous().view(torch.uint16).cpu().numpy()


def telemetry(label):
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    value = {'label': label, 'utc': datetime.now(timezone.utc).isoformat(), 'measurement_pid': os.getpid(), 'compute_processes': processes,
             'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}
    other = [line for line in processes.splitlines() if int(line.split(',')[0].strip()) != os.getpid()]
    value['other_compute_process_observed'] = bool(other)
    return value


class Component:
    def __init__(self, raw, weight, frozen, length, prefix, method):
        self.length, self.prefix, self.method = length, prefix, method
        self.q = gpu(raw['reference_queries_bits'][:, prefix - 1:prefix])
        self.k = gpu(raw['reference_keys_bits'][:, :prefix])
        self.v = gpu(raw['reference_values_bits'][:, :prefix])
        self.newk = self.k[:, -1:].clone(); self.newv = self.v[:, -1:].clone()
        self.weight = gpu(weight); self.bridge = None; self.known_ids = None
        if method.startswith('s'):
            seed = int(method.split('_')[0][1:]); kind = method.split('_', 1)[1]
            self.family = 'p3' if kind == 'p3' else 'p2'
            prime, digits = (3, (3, 2)) if self.family == 'p3' else (2, (4, 4))
            encoder, books = frozen[seed]
            self.bridge = FiniteBridge(encoder, {n: books[self.family + '_' + n]
                                       for n in ('center', 'projection', 'thresholds')}, prime, digits, self.q.device)
            self.lookup = depth_lookup(prime, digits, 2 if kind == 'coarsened_p2' else 1, self.q.device)
            self.kcodes = torch.tensor(raw[f's{seed}_{self.family}_key_codes'][:, :prefix].copy(), device='cuda')
            self.qcodes = torch.tensor(raw[f's{seed}_{self.family}_query_codes'][:, prefix - 1].copy(), device='cuda')
            self.seed = seed

    def address_update(self):
        self.k[:, -1:].copy_(self.newk); self.v[:, -1:].copy_(self.newv)
        qcode, qlatent = self.bridge.encode(self.q, 'query')
        kcode, klatent = self.bridge.encode(self.newk.repeat_interleave(3, 0), 'key')
        self.kcodes[:, -1:].copy_(kcode)
        self.qcodes.copy_(qcode[:, 0])
        return qcode, kcode, qlatent, klatent

    def selection(self):
        if self.method in ('native_full', 'recency_view'): return None
        if self.bridge is None: return fixed_keys(self.prefix, self.method, self.q.device)
        return selected_keys(self.qcodes, self.kcodes, self.lookup)

    def complete(self):
        if self.bridge is not None: self.address_update()
        else:
            self.k[:, -1:].copy_(self.newk); self.v[:, -1:].copy_(self.newv)
        if self.method == 'recency_view': return self.readout_only()
        return attention_component(self.q, self.k, self.v, self.weight, self.selection())

    def readout_only(self):
        if self.method != 'recency_view': raise ValueError('view-only recency control required')
        return attention_component(self.q, self.k[:, -128:], self.v[:, -128:], self.weight)

    def known_readout(self):
        return attention_component(self.q, self.k, self.v, self.weight, self.known_ids)

    def storage(self):
        tensors = {'query': self.q, 'physical_keys': self.k, 'physical_values': self.v,
                   'current_key': self.newk, 'current_value': self.newv, 'output_weight': self.weight}
        if self.bridge is not None:
            tensors.update({f'bridge_{i}': t for i, t in enumerate(self.bridge.tensors())})
            tensors.update(lookup=self.lookup, key_codes=self.kcodes, query_code=self.qcodes)
        rows = [{'name': name, 'shape': list(t.shape), 'dtype': str(t.dtype),
                 'logical_bytes': t.numel() * t.element_size(), 'storage_id': str(t.untyped_storage().data_ptr()),
                 'storage_bytes': t.untyped_storage().nbytes()} for name, t in tensors.items()]
        unique = {r['storage_id']: r['storage_bytes'] for r in rows}
        return {'tensors': rows, 'logical_payload_bytes': sum(r['logical_bytes'] for r in rows),
                'unique_storage_bytes': sum(unique.values())}


def check(component, raw):
    pos = component.prefix - 1
    method = 'recency' if component.method == 'recency_view' else component.method
    if component.bridge is not None:
        qcode, kcode, qlatent, klatent = component.address_update()
        name = f's{component.seed}_{component.family}_'
        for role, code, latent in (('query', qcode, qlatent), ('key', kcode, klatent)):
            assert np.array_equal(code.cpu().numpy(), raw[name + role + '_codes'][:, pos:pos + 1])
            assert np.array_equal(latent.cpu().numpy(), raw[name + role + '_latents'][:, pos:pos + 1])
    chosen = component.selection()
    if chosen is not None:
        assert np.array_equal(chosen.cpu().numpy(), raw[method + '_selected_ids'][:, pos, :128])
    projected, heads = component.complete()
    assert np.array_equal(cpu_bits(projected), raw[method + '_projected_bits'][:, pos:pos + 1])
    assert np.array_equal(cpu_bits(heads), raw[method + '_head_bits'][:, :, pos:pos + 1])


def summary(samples):
    values = np.array(samples, dtype=np.float64)
    return {'median_ms': float(np.median(values)), 'p95_ms': float(np.quantile(values, .95, method='linear'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('preflight', 'measure'), required=True)
    for name in ('evaluation', 'training', 'data-audit', 'numerical-audit', 'statistical-audit', 'protocol', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--preflight', type=Path)
    a = p.parse_args(); started = time.monotonic(); prior = 0
    if a.output.exists(): raise FileExistsError('fresh native component output required')
    a.output.mkdir(parents=True)
    native = ROOT / 'src/padic_lm/native_decode.py'
    for name, source in (('measurement_source.py', Path(__file__)), ('native_source.py', native), ('protocol.txt', a.protocol)):
        (a.output / name).write_bytes(source.read_bytes())
    def guard():
        if prior + time.monotonic() - started >= 3600: raise TimeoutError('native component cumulative3600s cost cap reached')
    try:
        assert sha(native) == NATIVE_SHA
        inputs = {name: sha(path) for name, path in (
            ('evaluation_completion', a.evaluation / 'completed.json'), ('training_completion', a.training / 'completed.json'),
            ('data_audit', a.data_audit), ('numerical_audit', a.numerical_audit), ('statistical_audit', a.statistical_audit))}
        numerical = json.loads(a.numerical_audit.read_text()); statistical = json.loads(a.statistical_audit.read_text())
        evaluation_config = json.loads((a.evaluation / 'config.json').read_text())
        assert inputs['training_completion'] == 'f01d2e872488fd2ac668f557341770c0bcc1d8d64e9a60b15f536c20d1fa64b1'
        assert evaluation_config['training_completion_sha256'] == inputs['training_completion']
        assert evaluation_config['data_audit_sha256'] == inputs['data_audit']
        assert json.loads(a.data_audit.read_text())['status'] == 'pg19_data_independently_verified'
        assert numerical['status'] == 'native_pg19_numerical_audit_passed'
        assert statistical['status'] == 'native_pg19_statistics_independently_verified'
        assert numerical['evaluation_completion_sha256'] == statistical['evaluation_completion_sha256'] == inputs['evaluation_completion']
        assert statistical['numerical_audit_sha256'] == inputs['numerical_audit']
        for path in (a.evaluation, a.training): verify(path, guard)
        sources = {'measurement': sha(__file__), 'native': sha(native), 'protocol': sha(a.protocol)}
        if a.mode == 'measure':
            if a.preflight is None: raise ValueError('successful correctness preflight required before timing')
            previous = verify(a.preflight, guard)
            assert previous['status'] == 'native_component_preflight_passed_no_latency_measurement'
            frozen_config = json.loads((a.preflight / 'config.json').read_text())
            assert frozen_config['source_sha256'] == sources and frozen_config['input_sha256'] == inputs
            prior = previous['cumulative_cost_seconds']; guard()
        initial = telemetry('before_cuda_work'); write(a.output / 'initial_telemetry.json', initial)
        if initial['other_compute_process_observed']: raise RuntimeError('GPU compute process list not empty at startup')
        torch.set_num_threads(1); torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
        raw = {length: load(a.evaluation / 'windows' / f'length-{length}-book-00.npz') for length in (2048, 4096)}
        weight = load(a.evaluation / 'output_projection.npz')['weight_bits']
        frozen = {s: (load(a.training / f'seed-{s}/final_encoder.npz'), load(a.training / f'seed-{s}/routing_state.npz')) for s in (17, 29, 43)}
        config = {'mode': a.mode, 'source_sha256': sources, 'input_sha256': inputs, 'cases': CASES,
                  'blocks': 7, 'order_pcg64_seed': 804211, 'warmups_per_phase': 20, 'wall_samples_per_phase': 100,
                  'event_samples_per_phase': 100, 'backend': 'forced_FLASH_ATTENTION', 'tf32': False,
                  'cache_policy': 'Warm repeated resident component; no hardware cache flush or other model layers resident',
                  'quality_family_screens': statistical['families'], 'packages': {n: importlib.metadata.version(n) for n in ('torch', 'numpy')},
                  'device': str(torch.cuda.get_device_properties(0)), 'prior_cost_seconds': prior,
                  'scope': 'Steady first-layer readout on sequence-derived QKV; no model/cached-generation latency or KV eviction'}
        write(a.output / 'config.json', config)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if a.mode == 'preflight':
                checks = []
                for length, prefix, method in CASES:
                    guard(); component = Component(raw[length], weight, frozen, length, prefix, method)
                    check(component, raw[length]); checks.append({'length': length, 'prefix': prefix, 'method': method, 'passed': True})
                    del component; gc.collect(); torch.cuda.empty_cache()
                final = telemetry('after_preflight'); write(a.output / 'final_telemetry.json', final)
                if final['other_compute_process_observed']: raise RuntimeError('other GPU process observed during preflight')
                assert {'measurement': sha(__file__), 'native': sha(native), 'protocol': sha(a.protocol)} == sources
                write(a.output / 'checks.json', checks); guard()
                finish(a.output, 'native_component_preflight_passed_no_latency_measurement', prior + time.monotonic() - started)
                return
            rng = np.random.Generator(np.random.PCG64(804211)); orders = [rng.permutation(len(CASES)).tolist() for _ in range(7)]
            write(a.output / 'case_orders.json', orders); summaries, storage, auxiliary, telemetry_rows = [], [], [], []
            with (a.output / 'samples.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=('block', 'order', 'case', 'length', 'prefix', 'method', 'phase', 'clock', 'sample', 'milliseconds'))
                writer.writeheader()
                for block, order in enumerate(orders):
                    t = telemetry(f'before_block_{block}'); telemetry_rows.append(t); write(a.output / 'telemetry.json', telemetry_rows)
                    if t['other_compute_process_observed']: raise RuntimeError('other GPU process observed before hardware block')
                    for ordinal, case in enumerate(order):
                        guard(); length, prefix, method = CASES[case]; setup = time.monotonic()
                        component = Component(raw[length], weight, frozen, length, prefix, method)
                        torch.cuda.synchronize(); setup_seconds = time.monotonic() - setup
                        phases = ['complete'] + (['address_update'] if component.bridge is not None else [])
                        if method == 'recency_view': phases += ['readout_only']
                        elif method != 'native_full': phases += ['selection', 'known_readout']
                        for phase in phases:
                            if phase == 'known_readout':
                                component.known_ids = component.selection(); torch.cuda.synchronize()
                                auxiliary.append({'block': block, 'case': case, 'length': length, 'prefix': prefix, 'method': method,
                                    'shape': list(component.known_ids.shape), 'dtype': str(component.known_ids.dtype),
                                    'logical_bytes': component.known_ids.numel() * component.known_ids.element_size(),
                                    'storage_bytes': component.known_ids.untyped_storage().nbytes(),
                                    'allocated_bytes_with_diagnostic_ids': torch.cuda.memory_allocated(),
                                    'scope': 'Extra already-selected IDs used only by the known-readout phase control'})
                                write(a.output / 'phase_auxiliary.json', auxiliary)
                            operation = getattr(component, phase)
                            for _ in range(20): operation()
                            torch.cuda.synchronize()
                            if phase == 'complete':
                                entry = {'block': block, 'case': case, 'length': length, 'prefix': prefix, 'method': method,
                                         'setup_process_seconds': setup_seconds, **component.storage(),
                                         'warm_allocated_bytes': torch.cuda.memory_allocated(), 'warm_reserved_bytes': torch.cuda.memory_reserved()}
                                torch.cuda.reset_peak_memory_stats(); output = operation(); torch.cuda.synchronize()
                                entry.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
                                entry['transient_allocated_increase_bytes'] = entry['peak_allocated_bytes'] - entry['warm_allocated_bytes']
                                storage.append(entry); del output; write(a.output / 'storage.json', storage)
                            measurements = {}
                            for clock in ('wall', 'event'):
                                values = []
                                if clock == 'event':
                                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                                    start.record(); end.record(); end.synchronize()
                                for sample in range(100):
                                    guard(); torch.cuda.synchronize()
                                    if clock == 'wall':
                                        origin = time.perf_counter_ns(); output = operation(); torch.cuda.synchronize()
                                        elapsed = (time.perf_counter_ns() - origin) / 1e6
                                    else:
                                        start.record(); output = operation(); end.record(); end.synchronize(); elapsed = start.elapsed_time(end)
                                    del output
                                    if not np.isfinite(elapsed) or elapsed <= 0: raise ArithmeticError('nonpositive/nonfinite timing')
                                    values.append(elapsed)
                                    writer.writerow({'block': block, 'order': ordinal, 'case': case, 'length': length, 'prefix': prefix,
                                          'method': method, 'phase': phase, 'clock': clock, 'sample': sample, 'milliseconds': elapsed})
                                measurements[clock] = summary(values)
                            summaries.append({'block': block, 'case': case, 'length': length, 'prefix': prefix, 'method': method,
                                              'phase': phase, **measurements})
                        f.flush(); del operation, component; gc.collect(); torch.cuda.empty_cache()
                    t = telemetry(f'after_block_{block}'); telemetry_rows.append(t); write(a.output / 'telemetry.json', telemetry_rows)
                    if t['other_compute_process_observed']: raise RuntimeError('other GPU process observed after hardware block')
                    write(a.output / 'progress.json', {'completed_blocks': block + 1, 'cumulative_cost_seconds': prior + time.monotonic() - started})
            ratios = []
            baseline = {(r['block'], r['length'], r['prefix']): r for r in summaries if r['phase'] == 'complete' and r['method'] == 'native_full'}
            for row in summaries:
                if row['phase'] == 'complete':
                    ref = baseline[row['block'], row['length'], row['prefix']]
                    ratios.append({**{n: row[n] for n in ('block', 'case', 'length', 'prefix', 'method')},
                          'wall_median_ratio_to_full': row['wall']['median_ms'] / ref['wall']['median_ms'],
                          'event_median_ratio_to_full': row['event']['median_ms'] / ref['event']['median_ms']})
            write(a.output / 'summaries.json', summaries); write(a.output / 'ratios.json', ratios)
            write(a.output / 'host_memory.json', {'peak_rss_kib_linux_entire_process': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
            assert {'measurement': sha(__file__), 'native': sha(native), 'protocol': sha(a.protocol)} == sources
            guard(); finish(a.output, 'native_component_measurement_complete_pending_independent_audit', prior + time.monotonic() - started,
                             {'preflight_completion_sha256': sha(a.preflight / 'completed.json')})
    except Exception as error:
        write(a.output / 'failed.json', {'status': 'native_component_cost_stage_failed', 'type': type(error).__name__,
              'message': str(error), 'cumulative_cost_seconds': prior + time.monotonic() - started})
        raise


if __name__ == '__main__': main()
