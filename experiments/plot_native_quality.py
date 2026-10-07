"""Display every frozen PG-19 comparison only after both result audits pass."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

METHODS = ['eager_original', 'native_full', 'recency', 'uniform'] + [
    f's{s}_{kind}' for s in (17, 29, 43) for kind in ('p2', 'p3', 'coarsened_p2')]
LABELS = {'eager_original': 'Original eager', 'native_full': 'Native full',
          'recency': 'Recency', 'uniform': 'Uniform + recent', 'p2': 'Binary', 'p3': 'Ternary',
          'coarsened_p2': 'Coarsened binary'}


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def display(method):
    if method.startswith('s'):
        seed, kind = method.split('_', 1); return LABELS[kind] + ' / ' + seed[1:]
    return LABELS[method]


def color(method):
    if method.endswith('_p3'): return '#bb662c'
    if method.endswith('_p2') and 'coarsened' not in method: return '#286493'
    if 'coarsened' in method: return '#467f63'
    return '#666666'


def interval_axis(ax, rows, margin):
    for pos, row in enumerate(rows):
        low, point, high = (row[n] for n in ('lower_025', 'paired_delta_nll', 'upper_975'))
        assert np.isfinite([low, point, high]).all() and low <= high
        ax.plot([low, high], [pos, pos], color=color(row['method']), lw=1.5)
        ax.scatter(point, pos, color=color(row['method']), s=24, zorder=3)
    ax.axvline(0, lw=.8, color='#777777'); ax.axvline(margin, lw=1.1, ls='--', color='#a32635')
    ax.set_yticks(range(len(rows)), [display(r['method']) for r in rows])
    ax.set_ylim(len(rows) - .5, -.5); ax.grid(axis='x', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_xlabel('Paired delta NLL (nats/target)')


def save(fig, output, name):
    for suffix in ('png', 'svg'): fig.savefig(output / (name + '.' + suffix), dpi=175)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('statistics', 'statistical-audit', 'numerical-audit', 'output'): p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('fresh native-quality figure directory required')
    completion = a.statistics / 'completed.json'; parent = json.loads(completion.read_text())
    assert sha(completion) == (a.statistics / 'completed.sha256').read_text().strip()
    for name, fingerprint in parent['artifact_sha256'].items():
        assert sha(a.statistics / name) == fingerprint and (a.statistics / name).stat().st_size == parent['artifact_bytes'][name]
    result = json.loads((a.statistics / 'results.json').read_text())
    numerical, statistical = (json.loads(path.read_text()) for path in (a.numerical_audit, a.statistical_audit))
    assert numerical['status'] == 'native_pg19_numerical_audit_passed'
    assert statistical['status'] == 'native_pg19_statistics_independently_verified'
    assert statistical['statistics_completion_sha256'] == sha(completion)
    assert numerical['evaluation_completion_sha256'] == statistical['evaluation_completion_sha256'] == result['evaluation_completion_sha256']
    assert statistical['numerical_audit_sha256'] == result['numerical_audit_sha256'] == sha(a.numerical_audit)
    assert len(result['primary']) == 72 and len(result['secondary']) == 100 and len(result['absolute']) == 52
    assert statistical['paired_intervals_verified'] == 172 and statistical['absolute_endpoints_verified'] == 52
    margin = result['allowance_nats']; assert margin == float(np.log(1.01))
    primary = {(r['length'], r['method'], r['reference'], r['endpoint']): r for r in result['primary']}
    secondary = {(r['length'], r['method'], r['reference'], r['endpoint']): r for r in result['secondary']}
    assert len(primary) == 72 and len(secondary) == 100
    for family in ('p2', 'p3'):
        rows = [r for r in result['primary'] if r['family'] == family]
        assert len(rows) == 36 and statistical['families'][family]['passed_comparisons'] == sum(r['passed'] for r in rows)
    a.output.mkdir(parents=True)
    footer = ('20,000 paired whole-book resamples; approximate 2.5/97.5% percentile intervals. Red line: log(1.01).\n'
              'Fixed three seeds; layer 0 native single-query readout on original sequence QKV. Real KV retained.\n'
              'Subword likelihood; no PG-19 word-PPL leaderboard, cached deployment or hardware claim.')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12.3))
    for row, length in enumerate((2048, 4096)):
        for column, endpoint in enumerate(('all', 'affected')):
            selected = [secondary[length, method, 'native_full', endpoint] for method in METHODS]
            ax = axes[row, column]; interval_axis(ax, selected, margin)
            ax.set_title(f'{length:,} tokens / {endpoint} targets')
    fig.suptitle('PG-19: all 13 readouts versus native full attention', fontsize=14)
    fig.text(.025, .017, 'These 52 intervals are descriptive secondary comparisons.\n' + footer, fontsize=9)
    fig.tight_layout(rect=(0, .088, 1, .965)); save(fig, a.output, 'native-pg19-all-readouts')
    candidates = [f's{s}_{family}' for s in (17, 29, 43) for family in ('p2', 'p3')]
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 13.7))
    for row, (length, endpoint) in enumerate((l, e) for l in (2048, 4096) for e in ('all', 'affected')):
        for column, reference_kind in enumerate(('native_full', 'uniform', 'coarsened_p2')):
            selected = []
            for method in candidates:
                reference = method.split('_', 1)[0] + '_coarsened_p2' if reference_kind == 'coarsened_p2' else reference_kind
                selected.append(primary[length, method, reference, endpoint])
            ax = axes[row, column]; interval_axis(ax, selected, margin)
            ax.set_title(f'{length:,} / {endpoint} / {reference_kind.replace("_", " ")}')
    decisions = '; '.join(f"{LABELS[f]} {statistical['families'][f]['passed_comparisons']}/36" for f in ('p2', 'p3'))
    fig.suptitle('All 72 frozen primary comparisons: ' + decisions, fontsize=14)
    fig.text(.025, .014, 'Each family requires all 36 upper bounds strictly below the margin.\n' + footer, fontsize=9)
    fig.tight_layout(rect=(0, .087, 1, .965)); save(fig, a.output, 'native-pg19-primary-screen')
    fig, axes = plt.subplots(2, 2, figsize=(14, 11.4))
    native_methods = METHODS[1:]
    for row, length in enumerate((2048, 4096)):
        for column, endpoint in enumerate(('all', 'affected')):
            selected = [secondary[length, method, 'eager_original', endpoint] for method in native_methods]
            ax = axes[row, column]; interval_axis(ax, selected, margin)
            ax.set_title(f'{length:,} tokens / {endpoint} targets')
    fig.suptitle('All 48 native-versus-eager arithmetic comparisons', fontsize=14)
    fig.text(.025, .017, 'Descriptive secondary comparisons; native full is the primary quality comparator.\n' + footer, fontsize=9)
    fig.tight_layout(rect=(0, .095, 1, .965)); save(fig, a.output, 'native-pg19-versus-eager')
    (a.output / 'plotted_values.json').write_text(json.dumps({'primary': result['primary'], 'secondary': result['secondary'], 'absolute': result['absolute']}, indent=2) + '\n')
    (a.output / 'plot_source.py').write_bytes(Path(__file__).read_bytes())
    files = sorted(a.output.iterdir())
    (a.output / 'manifest.json').write_text(json.dumps({'status': 'all_audited_native_pg19_intervals_plotted_pending_visual_review',
          'statistics_completion_sha256': sha(completion), 'numerical_audit_sha256': sha(a.numerical_audit),
          'statistical_audit_sha256': sha(a.statistical_audit), 'plot_source_sha256': sha(__file__),
          'primary_intervals': 72, 'secondary_native_full_intervals': 52, 'secondary_eager_intervals': 48,
          'families': statistical['families'], 'files_sha256': {p.name: sha(p) for p in files}}, indent=2) + '\n')


if __name__ == '__main__': main()
