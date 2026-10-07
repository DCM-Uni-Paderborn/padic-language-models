"""Independent loss sums, histogram resamples, order statistics and72 decisions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def verify(directory):
    assert sha(directory / 'completed.json') == (directory / 'completed.sha256').read_text().strip()
    value = json.loads((directory / 'completed.json').read_text())
    for name, h in value['artifact_sha256'].items(): assert sha(directory / name) == h and (directory / name).stat().st_size == value['artifact_bytes'][name]
    return value


def percentile(samples, probability):
    values = sorted(float(x) for x in samples)
    location = (len(values) - 1) * probability
    left, fraction = int(math.floor(location)), location - math.floor(location)
    right = min(left + 1, len(values) - 1)
    difference = values[right] - values[left]
    return values[left] + difference * fraction if fraction < .5 else values[right] - difference * (1 - fraction)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('evaluation', 'statistics', 'numerical-audit', 'output'): p.add_argument('--' + n, type=Path, required=True)
    a = p.parse_args(); started = time.monotonic()
    if a.output.exists(): raise FileExistsError('fresh independent statistics audit required')
    a.output.mkdir(parents=True); (a.output / 'audit_source.py').write_bytes(Path(__file__).read_bytes())
    stat_parent = verify(a.statistics); prior = stat_parent['cumulative_gate_seconds']
    try:
        eval_parent = verify(a.evaluation)
        result = json.loads((a.statistics / 'results.json').read_text())
        numerical = json.loads(a.numerical_audit.read_text())
        assert numerical['status'] == 'native_pg19_numerical_audit_passed'
        assert result['evaluation_completion_sha256'] == sha(a.evaluation / 'completed.json') == numerical['evaluation_completion_sha256']
        assert result['numerical_audit_sha256'] == sha(a.numerical_audit)
        assert result['allowance_nats'] == float(np.log(1.01)) and result['documents'] == 40 and result['draws'] == 20000
        methods = json.loads((a.evaluation / 'config.json').read_text())['methods']
        sums, counts, targets_verified = {}, {}, 0
        for length in (2048, 4096):
            for method in methods:
                for endpoint in ('all', 'affected'): sums[length, method, endpoint] = []; counts[length, method, endpoint] = []
            for book in range(40):
                with np.load(a.evaluation / 'windows' / f'length-{length}-book-{book:02d}.npz', allow_pickle=False) as r:
                    for method in methods:
                        values = r[method + '_target_nll']; assert len(values) == length - 1 and np.isfinite(values).all()
                        targets_verified += len(values)
                        for endpoint, start in (('all', 0), ('affected', 128)):
                            sums[length, method, endpoint].append(math.fsum(float(x) for x in values[start:]))
                            counts[length, method, endpoint].append(len(values) - start)
        rng = np.random.Generator(np.random.PCG64(514203))
        expected_draws = rng.integers(0, 40, (20000, 40), dtype=np.int64)
        with np.load(a.statistics / 'resample_indices.npz', allow_pickle=False) as r: assert np.array_equal(expected_draws, r['draws'])
        frequencies = np.array([np.bincount(draw, minlength=40) for draw in expected_draws], dtype=np.int64)
        checked, maximum = 0, 0.
        for row in result['primary'] + result['secondary']:
            key = row['length'], row['method'], row['endpoint']; refkey = row['length'], row['reference'], row['endpoint']
            delta = np.asarray(sums[key]) - np.asarray(sums[refkey]); n = np.asarray(counts[key])
            assert counts[key] == counts[refkey]
            samples = (frequencies @ delta) / (frequencies @ n)
            expected = {'paired_delta_nll': math.fsum(float(x) for x in delta) / sum(counts[key]),
                        'lower_025': percentile(samples, .025), 'upper_975': percentile(samples, .975)}
            for name, value in expected.items():
                difference = abs(row[name] - value); maximum = max(maximum, difference)
                assert difference <= 1e-14
            assert abs(row['point_relative_ppl_percent'] - float(100 * np.expm1(expected['paired_delta_nll']))) <= 1e-12
            assert abs(row['upper_relative_ppl_percent'] - float(100 * np.expm1(expected['upper_975']))) <= 1e-12
            if 'passed' in row: assert row['passed'] == (expected['upper_975'] < float(np.log(1.01)))
            checked += 1
        assert len(result['primary']) == 72 and len(result['secondary']) == 100 and checked == 172
        classifications = {}
        for family in ('p2', 'p3'):
            rows = [r for r in result['primary'] if r['family'] == family]
            assert all(r['method'] == f"s{r['seed']}_{family}" for r in rows)
            combinations = {(r['length'], r['seed'], r['endpoint'], r['reference']) for r in rows}
            expected_combinations = {(length, seed, endpoint, ref) for length in (2048, 4096) for seed in (17, 29, 43)
                                     for endpoint in ('all', 'affected') for ref in ('native_full', 'uniform', f's{seed}_coarsened_p2')}
            assert len(rows) == 36 and combinations == expected_combinations
            passed = sum(r['passed'] for r in rows)
            assert result['families'][family]['required_comparisons'] == 36 and result['families'][family]['passed_comparisons'] == passed
            assert result['families'][family]['screen_passed_pending_statistics_audit'] == (passed == 36)
            classifications[family] = {'passed_comparisons': passed, 'required_comparisons': 36,
                    'screen_passed': passed == 36, 'largest_upper_nats': max(r['upper_975'] for r in rows),
                    'label': 'quality non-inferior in the frozen three-seed two-context native-readout PG-19 screen' if passed == 36 else 'not established by the frozen native-readout PG-19 screen'}
        for row in result['absolute']:
            key = row['length'], row['method'], row['endpoint']; expected = math.fsum(sums[key]) / sum(counts[key])
            assert abs(row['nll'] - expected) <= 1e-14 and abs(row['subword_ppl'] - float(np.exp(expected))) <= 1e-12
        assert len(result['absolute']) == 52
        if prior + time.monotonic() - started >= 10800: raise TimeoutError('native PG-19 cumulative10800s cap reached')
        report = {'status': 'native_pg19_statistics_independently_verified', 'evaluation_completion_sha256': sha(a.evaluation / 'completed.json'),
                  'statistics_completion_sha256': sha(a.statistics / 'completed.json'), 'numerical_audit_sha256': sha(a.numerical_audit),
                  'audit_source_sha256': sha(__file__), 'raw_target_losses_verified': targets_verified,
                  'draw_indices_verified': expected_draws.size, 'paired_intervals_verified': checked,
                  'absolute_endpoints_verified': 52, 'max_interval_difference': maximum, 'families': classifications,
                  'audit_wall_seconds': time.monotonic() - started, 'cumulative_gate_seconds': prior + time.monotonic() - started,
                  'scope': 'No production imports; raw-loss fsum/histogram-dot resamples/explicit order-statistic interpolation; approximate fixed-family bootstrap screen'}
        (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n'); print(json.dumps(report), flush=True)
    except Exception as error:
        (a.output / 'failed.json').write_text(json.dumps({'status': 'native_pg19_statistics_audit_failed', 'message': str(error),
             'type': type(error).__name__, 'cumulative_gate_seconds': prior + time.monotonic() - started}, indent=2) + '\n'); raise


if __name__ == '__main__': main()
