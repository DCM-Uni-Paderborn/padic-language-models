# p-adic numbers for LLM

Computational materials for the working manuscript by Thomas Kühne. The paper asks whether finite p-adic structure can be useful in language-model components, with conceptual feasibility as the primary question.

[Read the manuscript and supporting information](manuscript/main.tex). Both are contained in this single LaTeX source. Compile it with a standard LaTeX distribution or import it into Overleaf. The current source also compiles in the desktop document editor.

The experiments establish ordinary arithmetic and prefix-tree equivalences, a limited WikiText first-layer quality result, and a negative native PG-19 transfer result. A continuous disk learner recovers an exact modular rule on typed operands. Its fixed coupling to LLM logits does not beat the Fourier control. The semantic-prefix and finite-ring FFN studies provide further scoped results. The full-projection low-bit study adds 168 continuation fits. None of its 54 low-bit settings retains quality on both corpora in every seed, while matched BF16 continuation does. Some cyclic encoders beat same-alphabet clipping, and conventional FP4 is stronger in the principal comparisons. The paper makes no general quality, compression or hardware advantage claim.

## Contents

| Material | Location |
| --- | --- |
| Complete paper and SI | `manuscript/main.tex` |
| Scientific implementations | `src/padic_lm/` |
| Experiment and analysis drivers | `experiments/` |
| Mathematical and numerical tests | `tests/` |
| Frozen protocols and settings | `research/` |
| Exact token inputs, selected examples, learned states and compact results | `results/` |
| Input provenance and file checksums | `inputs/` |
| Fresh reproduction helpers | `tools/` |
| Dataset attribution and licenses | `THIRD_PARTY_NOTICES.md`, `licenses/` |

The package keeps inputs and reported summary tables. Large feature captures, per-query arrays, duplicate corpus downloads, model caches, historical manuscript drafts and technical failed attempts are omitted. Scientifically meaningful negative results remain in the paper and summaries. Version pins needed to identify external sources are in machine-readable manifests.

## Inputs and studies

| SI sections | Essential inputs | Main drivers and compact results |
| --- | --- | --- |
| S1–S2, S8 | Source-version manifests in `inputs/` | Theory and related mechanisms in the manuscript |
| S3 | Discovery token IDs in `results/llm-control-*` | `arithmetic_control.py`, `llm_control.py`, arithmetic and projection summaries |
| S4 | Discovery tokens, article-preserving training/development windows | Routing, encoder and ordinary-control drivers and summaries |
| S5–S7 | Shared-context and held-out WikiText tokens, PG-19 nested prefixes, three learned encoder/routing states | Quality, prefix-index, native-operator and component-cost summaries |
| S9 | Frozen WikiText training/development tokens | `pilot_ball_ffn.py`, `ball-ffn-summary-*` |
| S10 | CLINC150 texts, selected rows, tokens and domain taxonomy | `pilot_structural_intent.py`, `structural-intent-summary-*` |
| S11 | Three synthetic regression datasets, modular prompts and frozen LLM priors | `pilot_continuous_tree.py`, `pilot_modular_adapter.py`, continuous and modular summaries |
| S12 | Frozen WikiText training/development and both test corpora | Full-Transformer low-precision continuation and compact per-document statistics |

Each data directory preserves the original sample selection and content hashes. PG-19 supplies exact model token prefixes plus the whole-book source URLs and hashes. The upstream texts and checkpoint weights can be fetched at their recorded versions. The learned encoder/routing states are small outputs of the training study and essential inputs to downstream comparisons.

## Setup and verification

Use Python 3.12 or newer. The base package requires NumPy. Model experiments additionally require the language dependencies. Recorded original runs use Transformers 4.57.6 and datasets 4.4.1. The Spark runs use PyTorch 2.12.0+cu130 on NVIDIA GB10; the earlier Terok controls use PyTorch 2.6.0+cu124 on NVIDIA A40. These environments affect native BF16 contraction and quantile-boundary behavior.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python tools/verify_materials.py
```

For model experiments, install `.[language]` into an appropriate existing PyTorch/CUDA environment. Figure drivers use `.[figures]`. The complete test suite uses PyTorch as well. Run `PYTHONPATH=src python -m unittest discover -s tests` after installing the language dependencies. Runtime and dependency installation are separate from the scientific inputs.

## Reproduction

Always use a fresh output directory. Keep the supplied input and summary directories unchanged. Some historical audit drivers require the complete derived archive and its original provenance chain. Regenerate the derived files before using those auditors. The compact package is not a mirror of every intermediate archive.

The following commands use the included inputs and generate new outputs.

```bash
python experiments/arithmetic_control.py --samples 128 --output runs/arithmetic
python experiments/pilot_continuous_tree.py \
  --execution research/continuous-tree-execution.json --output runs/continuous
python experiments/audit_continuous_tree.py runs/continuous
```

The frozen WikiText data directly support a new training or FFN run.

```bash
python experiments/train_qk_bridge.py \
  --data results/learned-data-articles-360m-spark-20261004 \
  --protocol research/learned-encoder-protocol.md --output runs/qk-training
python experiments/pilot_ball_ffn.py \
  --data results/learned-data-articles-360m-spark-20261004 \
  --execution research/ball-ffn-pilot-execution.json --output runs/ffn
python experiments/pilot_structural_intent.py \
  --execution research/structural-intent-execution.json --output runs/semantic
python experiments/pilot_modular_adapter.py \
  --execution inputs/modular-execution.json --output runs/modular
```

These GPU runs use the original checkpoint and scientific settings. The modular driver rebuilds its frozen priors from the checkpoint. The published priors are also supplied for fitting or inspecting the branch without fetching the checkpoint.

The native helper evaluates the original 40 PG-19 prefixes at both context lengths using the supplied learned states. It writes compact per-book endpoints instead of all captured tensors. It retains the original BF16/Flash operator and model-state check.

```bash
python tools/reproduce_native_quality.py --output runs/native
python tools/rebuild_native_statistics.py \
  --endpoints runs/native/document-endpoints.json --output runs/native-statistics
```

The published native statistics can also be regenerated without a GPU from the compact endpoint sums.

```bash
python tools/rebuild_native_statistics.py \
  --endpoints results/native-quality-summary-20261004/document-endpoints.json \
  --output runs/published-native-statistics
```

Fresh execution clocks and output fingerprints will differ. Native numerical replay across hardware is a separate replication and must be assessed against the paper's stated contracts. The helper does not retune seeds, margins, context lengths or the method family.


## Low-precision and mixed-precision continuation

SI Section S12 compares 2-, 3- and 4-bit weight/activation alphabets throughout all Transformer attention and feed-forward projections. The fixed matrix contains 168 short continuation fits over three seeds. It includes integer and real FP4 controls, balanced residues, unit/exponent alphabets, cyclic low-digit projection alternating W2/W4 blocks and matched BF16 continuation controls. Embeddings, the tied output head, normalization and residual tensors retain higher precision.

Exact tokens and article selections are in `results/low-precision-inputs-20261019/`. The complete matrix and control protocols are indexed by `research/low-precision-execution.json`. The common training recipe, model configurations and source freeze are in `research/low-precision-*.md` and `inputs/low-precision-*.json`. The mixed schedule has an average prescribed weight cap of three bits. Actual index widths can underfill a cap, so the ternary-residue mixed and uniform-three-bit cases have different packed payloads. A three-state index is counted as two bits. Shared scales and remaining BF16 parameters are included in the parameter payload.

The complete GPU study can be regenerated from the supplied tokens. The optional download flag fetches the two checkpoints at the revisions in the input manifest. All output directories must be new.

```bash
python tools/reproduce_low_precision.py \
  --output runs/low-precision --download-models
```

The published document loss sums also permit CPU-only rebuilding of every reported endpoint, paired bootstrap interval and three-seed decision.

```bash
python experiments/analyze_low_precision.py \
  --published results/low-precision-summary-20261019/compact-fits.json \
  --output runs/low-precision-statistics
python experiments/summarize_low_precision.py \
  --statistics runs/low-precision-statistics --output runs/low-precision-tables
```

The compact records retain scalar document losses for both the initial quantized model and the final fixed update. They omit teacher logits, optimizer states, shadow-weight checkpoints and execution logs. The tests independently check alphabets, exact aliases, digit projection, surrogate gradients and index packing. A separate calculation verifies parameter payloads from the pinned model configurations.

Matrix operations use real arithmetic. These fits do not reproduce native NVFP4 scaling/training or BitNet pretraining, and they do not implement an LLM valued entirely in the p-adic field. The original BF16 checkpoint receives no continuation. The supplementary BF16 control uses the same updates with BF16-rounded forward weights and FP32 shadow-weight optimization. Test corpora are held out from these updates but were already used in earlier component studies.
