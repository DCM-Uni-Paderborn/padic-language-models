#!/usr/bin/env bash
# Frozen dependent stages. Any failure stops the study without a retry.
set -euo pipefail
cd /home/kuehne88/work/padic-language-models
study_python=.venv/bin/python
study_freeze=research/native-quality-execution.json
study_data=results/pg19-context-data-spark-20261004
study_eval=results/native-quality-pg19-spark-20261004
study_numeric=results/native-quality-pg19-audit-spark-20261004
study_stats=results/native-quality-pg19-statistics-spark-20261004
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$study_python" experiments/prepare_pg19_contexts.py --freeze "$study_freeze" --metadata results/pg19-source-metadata-20261004 --accepted results/learned-data-articles-360m-spark-20261004 --heldout results/heldout-context-data-spark-20261004 --output "$study_data" > results/environment/pg19-context-preparation.log 2>&1
"$study_python" experiments/audit_pg19_contexts.py --data "$study_data" --metadata results/pg19-source-metadata-20261004 --accepted results/learned-data-articles-360m-spark-20261004 --heldout results/heldout-context-data-spark-20261004 --output results/pg19-context-audit-spark-20261004 > results/environment/pg19-context-audit.log 2>&1
"$study_python" experiments/evaluate_native_quality.py --freeze "$study_freeze" --data "$study_data" --data-audit results/pg19-context-audit-spark-20261004/audit.json --training results/learned-qk-training-spark-20261004 --output "$study_eval" > results/environment/native-quality-pg19.log 2>&1
"$study_python" experiments/audit_native_quality.py --evaluation "$study_eval" --training results/learned-qk-training-spark-20261004 --output "$study_numeric" > results/environment/native-quality-pg19-audit.log 2>&1
"$study_python" experiments/analyze_native_quality.py --freeze "$study_freeze" --evaluation "$study_eval" --audit "$study_numeric/audit.json" --output "$study_stats" > results/environment/native-quality-pg19-statistics.log 2>&1
"$study_python" experiments/audit_native_statistics.py --evaluation "$study_eval" --statistics "$study_stats" --numerical-audit "$study_numeric/audit.json" --output results/native-quality-pg19-statistics-audit-spark-20261004 > results/environment/native-quality-pg19-statistics-audit.log 2>&1
