# P1 concept and learning pilot on Spark

2026-10-04. Prospective development experiment, frozen before any new P1 teacher activation, model fit or language-model output. Conceptual usefulness is primary; timing is merely an allocation ledger. This does not reopen or amend the completed P2 WikiText/PG-19 quality screens or component-cost allocations.

## Question and scope

Can the explicit small finite-ring affine/ball FFN learn a nontrivial teacher approximation and supply useful information to a real pretrained LLM? Does the chosen digit hierarchy help against identical-capacity ordinary and changed-encoding controls? This is a bounded exploratory test of Candidate B in `protocol-review.md`, not a claim that all neural computation uses Q_2, a matched BitNet/NVFP4 conversion, untouched confirmation, compression or hardware inferiority/superiority. An unsuccessful small bottleneck stops this particular recipe, not the overarching mathematical research question.

The source checkpoint is SmolLM2-360M at revision f8027fd0eaeea54caa13c31d31b9fdc459c38b49, BF16 eager execution, loaded-state SHA256 b9fa643b37b587b20e70991f30a7e24a9897aeaa159467fce79b59152f70d583. Replace the output of layer0's entire MLP with a hard candidate output, retaining the remainder of the model. This first intervention does not change the MLP input, which is checked exactly on every evaluated window. All output conversions into the original model are BF16 and included in the experiment. A teacher-output replay must reproduce every stock token loss for the first development window before any intervention can be interpreted.

## Data and supervision

Use the corrected frozen `results/learned-data-articles-360m-spark-20261004` completion SHA256 033d9984858ab0f417181980d64919a901fe813b6d6d97362410c02c959ed454. Capture MLP inputs/outputs for all16 training windows of512 tokens (8192 adaptation tokens) and all64 development windows (58 articles,32704 scored next-token targets). These official training/validation articles are disjoint under the existing manifest. Development text has previously been used in P2; it is explicitly reused development data, never an untouched test. No WikiText test or PG-19 quality text is read in this pilot. Pretraining contamination cannot be removed by this split.

Fit encoder and all candidates using only the8192 training states and teacher MLP outputs. No development state selects a hyperparameter or changes a ring weight. Every method sees the same training supervision, fixed regularization and final language-model assessment. Teacher capture uses80 original model windows, identity uses1, declared interventions use1664, and the independent numerical/LM auditor uses27 fresh development windows. Count finite alternatives separately; an algebraic objective update is not an additional teacher call.

## Fixed small architecture and controls

Training-only PCA maps the960-dimensional MLP input into16 real coordinates. Its center and projection are stored FP32; hard projection/calibration use FP64. Each coordinate is assigned16 training-quantile bins, with equality entering the upper bin. Thus the encoder is a learned real bridge, not ring arithmetic. The primary map reverses the four ordinary binary digits so ordered leading subdivisions become low p-adic digits. Use32 affine rows over Z/(16Z),16 inputs/row and32 biases; each row supplies four zero-ball indicators at depths1--4, giving128 features. The readout has128x960 FP64 coefficients and960 FP64 biases. Real decoder/encoder storage is counted; finite-ring weights are not the whole model.

Seeds17,29,43 initialize independent32x16 four-bit weight values and biases. All trained controls receive the same initial coefficient digits and coordinate proposal order within a seed. Report all seeds, both initial/final fits and every fixed control:

- `reverse_ball`: exact modulo16 affine map and four congruence-ball indicators.
- `ordered_ball`: identical bins/ring/balls, ordered labels without digit reversal.
- `shuffled_ball`: independent fixed permutation of16 bin labels for each input coordinate, identical ring/balls. This preserves bin occupancy/capacity while altering hierarchy and arithmetic; it is not claimed to isolate hierarchy alone.
- `reverse_signed`: primary encoded ring affine output decoded into[-8,7]/8; four powers supply128 real features. This keeps the ring map but changes its nonlinearity.
- `real_threshold`: same16 PCA coordinates standardized on training, same four-bit coefficient/bias alphabet interpreted as real signed values/8, ordinary real affine sums without wrapping. Divide each row by its fixed initial coefficient norm and use ordinary thresholds[-1,-.25,.25,1]. Same128 features and search/readout budget. This changes geometry/arithmetic jointly and is not a perfect causal single-factor baseline.
- `unmixed_ball`: ring row i copies coordinate i mod16, with the same fixed initialized biases, then applies four balls and ridge readout; no affine fitting. This is also the declared identity-like initialization/modeling control.
- `real_linear`:16 standardized PCA coordinates and112 explicit zero columns, ridge readout only. Report the16 active features, not128 effective linear degrees of freedom.

Also replace the FFN by zero and by its training-output mean, assessing whether a candidate contributes useful information beyond trivial replacement. An exactly equivalent low-digit prefix tree is the independent arithmetic/feature control, not a separate quality mechanism; its complete features are audited against every saved fitted candidate. Prime3 is reserved for a later independently specified design after binary correctness; this pilot makes no prime ranking.

## Hard learning and budgets

With encoder and initialized affine state fixed, fit the real readout by ridge regression minimizing mean training output SSE plus0.001 times coefficient squared norm, with unpenalized intercept. No regularization grid or development tuning. For each of five fitted methods, perform exactly two sweeps over all32 rows and all68 binary digit coordinates (16 weights plus bias, four digits each). Evaluate the current and flipped binary digit by exact hard features and the real FP64 training SSE, with the readout fixed for that sweep. Accept only strict improvement beyond max(1e-10, current full SSE *1e-12). Maintain analytic SSE differences, and check their agreement with the complete residual after every row. Refit the ridge readout after each sweep. No surrogate gradient, soft feature or claimed p-adic derivative is used. Each fitted method gets4352 flipped alternatives, giving65280 across all15 seed/method fits. Fixed unmixed/linear controls receive one ridge fit and no search; disclose the unequal fitting counts.

Archive original teacher inputs/outputs, all per-target stock/intervention losses, initial/final affine/readout arrays, real encoder, input permutations/normalizers, every row's proposal/acceptance/SSE ledger, source/protocol/execution hashes, complete record summaries and any failure. Primary diagnostics are training and development FFN NRMSE (real output before casting and actual BF16 output), development token NLL relative to stock and null/ordinary controls, hard parameter changes and feature occupancy/rank. FFN MSE alone is never evidence of language-model utility. Replacing a single early FFN does not establish multi-layer robustness or cached generation.

## Prospective development progression criteria

These point-estimate criteria decide whether this fixed small recipe warrants a new untouched follow-up; they are not a confirmatory non-inferiority test or publication competitiveness claim. All three primary seeds must:

1. Change hard affine weights and reduce full training SSE relative to their initialized/readout-fitted model.
2. Have development NLL at most stock + log(1.10). The10% exploration allowance is intentionally distinct from the closed1% P2 screens; it must never be presented as passing those screens.
3. Beat both zero and mean FFN replacements by at least0.005 nats/target.
4. Be at most the fitted ordinary real-threshold control + log(1.02).

Report each condition and seed, including failure. Digit-order/signed/unmixed results diagnose the mechanism descriptively, without post-hoc favorable selection or a universal uniquely p-adic interpretation. If the recipe fails, report the exact bottleneck, adaptation and control limits; a larger or different mechanism needs a separately frozen new plan and fresh quality evaluation.

## Execution and verification

Run only on Spark through its configured SSH alias via uts.hzdr.de; preserve host-key checking and disable credential forwarding. Terok is not needed for this concept pilot. Do not change packages/device clocks or stop foreign workloads. The producer has one1800-second process allocation and no automatic restart; a separate independent auditor has one1800-second allocation. Timing is a bounded work ledger, not a systems result. Observation timeouts attach to the same process. Preserve unsuccessful outputs if either stage fails; any repaired follow-up must explicitly retain its lineage.

Before execution, four local mathematical tests must pass: Python-integer ring oracle with safe-overflow rejection; exhaustive p=2,3,5 finite ball/tree and digit reversal identities; signed endpoint handling; hard-coordinate objective and ridge normal-equation checks. Freeze the protocol and all producer/core/auditor/test sources in an execution manifest and Git commit before the first new model output. The independent auditor verifies every artifact pin, training-only center/quantiles, BF16 capture representation, all44 seed/method/stage/null records, independent ring/tree/readout calculations, fit normal equations, raw loss aggregates and optimization ledgers. It reloads the checkpoint and recomputes every first-window stock/intervention token loss. A producer success without audit is not a verified scientific result.
