"""Fixed72-comparison whole-book native PG-19 quality screen."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from padic_lm.heldout import paired_document_bootstrap
from padic_lm.native_study import sha, write, snapshot, verify, complete, guard, failure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('freeze', 'evaluation', 'audit', 'output'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic(); prior = 0
    if a.output.exists(): raise FileExistsError('fresh native PG-19 statistics required')
    a.output.mkdir(parents=True); snapshot(a.output, a.freeze)
    try:
        verify(a.evaluation)
        audit = json.loads(a.audit.read_text())
        assert audit['status'] == 'native_pg19_numerical_audit_passed' and audit['evaluation_completion_sha256'] == sha(a.evaluation / 'completed.json')
        prior = audit['cumulative_gate_seconds']; guard(started, prior)
        rows = json.loads((a.evaluation / 'document_endpoints.json').read_text())
        config = json.loads((a.evaluation / 'config.json').read_text())
        assert config['freeze_sha256'] == sha(a.freeze)
        methods = config['methods']; table = {}
        for length in (2048, 4096):
            for method in methods:
                for endpoint in ('all', 'affected'):
                    selected = sorted((r for r in rows if r['length'] == length and r['method'] == method and r['endpoint'] == endpoint), key=lambda r: r['book_index'])
                    assert [r['book_index'] for r in selected] == list(range(40))
                    table[length, method, endpoint] = (np.array([r['loss_sum'] for r in selected], dtype=np.float64),
                                                      np.array([r['targets'] for r in selected], dtype=np.int64))
        draws = np.random.Generator(np.random.PCG64(514203)).integers(0, 40, size=(20000, 40), dtype=np.int64)
        np.savez_compressed(a.output / 'resample_indices.npz', draws=draws)
        absolute, primary, secondary = [], [], []
        def compare(length, method, reference, endpoint):
            sums, counts = table[length, method, endpoint]; refs, refcounts = table[length, reference, endpoint]
            assert np.array_equal(counts, refcounts)
            interval, samples = paired_document_bootstrap(sums, refs, counts, draws)
            return {'length': length, 'method': method, 'reference': reference, 'endpoint': endpoint, **interval,
                    'point_relative_ppl_percent': float(100 * np.expm1(interval['paired_delta_nll'])),
                    'upper_relative_ppl_percent': float(100 * np.expm1(interval['upper_975']))}
        for length in (2048, 4096):
            for method in methods:
                for endpoint in ('all', 'affected'):
                    sums, counts = table[length, method, endpoint]; nll = float(sums.sum() / counts.sum())
                    absolute.append({'length': length, 'method': method, 'endpoint': endpoint, 'nll': nll, 'subword_ppl': float(np.exp(nll))})
                    secondary.append(compare(length, method, 'native_full', endpoint))
                    if method != 'eager_original': secondary.append(compare(length, method, 'eager_original', endpoint))
            for family in ('p2', 'p3'):
                for seed in (17, 29, 43):
                    for reference in ('native_full', 'uniform', f's{seed}_coarsened_p2'):
                        for endpoint in ('all', 'affected'):
                            row = compare(length, f's{seed}_{family}', reference, endpoint)
                            row['family'] = family; row['seed'] = seed; row['passed'] = row['upper_975'] < float(np.log(1.01))
                            primary.append(row)
        assert len(primary) == 72 and len(absolute) == 52 and len(secondary) == 100
        decisions = {f: {'required_comparisons': 36, 'passed_comparisons': sum(r['passed'] for r in primary if r['family'] == f),
                         'screen_passed_pending_statistics_audit': all(r['passed'] for r in primary if r['family'] == f)} for f in ('p2', 'p3')}
        write(a.output / 'results.json', {'status': 'native_pg19_statistics_complete_pending_independent_audit',
              'evaluation_completion_sha256': sha(a.evaluation / 'completed.json'), 'numerical_audit_sha256': sha(a.audit),
              'allowance_nats': float(np.log(1.01)), 'documents': 40, 'draws': 20000, 'bootstrap_seed': 514203,
              'primary': primary, 'secondary': secondary, 'absolute': absolute, 'families': decisions,
              'scope': 'Approximate paired whole-book bootstrap; fixed3 seeds and2 contexts; native readout on original sequence QKV, no hardware or deployment claim'})
        guard(started, prior); complete(a.output, 'native_pg19_statistics_complete_pending_independent_audit', prior + time.monotonic() - started)
        print(json.dumps({'families': decisions, 'completion_sha256': sha(a.output / 'completed.json')}), flush=True)
    except Exception as error: failure(a.output, error, started, prior); raise


if __name__ == '__main__': main()
