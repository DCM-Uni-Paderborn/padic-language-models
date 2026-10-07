# Research protocol: finite p-adic components for language models

Status: prospective working protocol, 2026-10-03. The protocol may be amended before confirmatory runs; amendments must be dated and justified without selecting favorable results.

## Objective and scope

Explain the theory of p-adic numbers and finite precision, investigate new applications in LLMs or their components, and produce an evidence-based scientific paper. A competitive component or localized advantage is a valid success. Mathematical clarity and sound negative findings remain valuable, but arithmetic controls alone do not satisfy the full research objective.

## Mathematical contracts

- Define Q_p, Z_p, valuation v_p, norm p^(-v_p), ultrametric inequality and finite quotients Z/p^K Z. Distinguish residue precision from real error and from exponent/mantissa representations.
- For residues, state modulus, encoding, decoder, valid range and the convention for zero. Division is available only for units in the finite quotient.
- A balanced residue decoder reconstructs a bounded integer dot product when its range lies inside the decoder interval. Test this exactly.
- Fixed real scales remain outside the residue ring. Keeping them is a hybrid construction.
- A modulo-2^K dot product with ordinary signed decoding is conventional wrapping integer computation. It is a null/control, not a new arithmetic advantage.
- A valuation or ball-indicator module must have an explicit finite definition and real readout. A distance/tree module must expose its ordinary prefix-tree equivalence.
- Conventional real-valued losses can supervise a hybrid component, but discrete optimization or a specified surrogate must be stated; ordinary differentiation through residue/valuation operations is not assumed.

## Candidate lanes

R0: frozen pretrained projection, ordinary quantized dot product versus residue dot product and balanced reconstruction. Purpose: implementation correctness, failure boundaries and conversion cost.

P1: finite p-adic affine/ball or valuation component with a real output decoder. Start with one block or narrow branch and an explicit training rule. Distillation from the original block is allowed, with a matched conventional distilled baseline.

P2: query-dependent hierarchical selection/routing of keys and values using p-adic codes and prefix shells. Compare to an identical trie implementation, conventional clustering/LSH approaches, and relevant existing KV selection methods. A new evaluated application is possible even if its finite representation is mathematically equivalent to a tree; novelty must be independently checked.

These are research candidates, not established recipes. Candidate choice will be justified by literature and early diagnostic experiments before the confirmation phase.

## Data separation and comparisons

- Pilot/development uses validation splits. Train/calibration, development and final test must be disjoint. Record source, revision, tokenizer, sequence lengths, boundaries, truncation and token counts.
- Primary initial model: SmolLM2-135M base. Larger model and independent architecture required before a general LLM claim.
- Match information supplied to models; hierarchy hints must also be supplied to conventional controls.
- Report nominal code bits and actual bytes, including codebooks, scales, padding, encoders, decoder/readout, activation/KV storage and accumulator widths.
- BF16 reference; calibrated INT8/INT4 baseline with the same scopes and data; native FP4/ternary baselines where supported. A simulated format cannot justify a hardware throughput comparison.
- For trained candidates, match data, teacher supervision, parameter budget and optimization effort; use at least three seeds initially, with a final replication count chosen before confirmation.

## Prospective competitiveness criteria

Amendment, 2026-10-03: the initial proposal floated an upper confidence bound on a perplexity ratio of at most 1.05 relative to a resource-matched conventional baseline. The routing-specific [next-experiments proposal](/Users/tkuehne/Documents/Science/padic-language-models/research/next-experiments.md) instead discusses 1.01, corresponding to `log(1.01)` nats/token, because selecting keys with unchanged weights motivates a tighter tolerance than an aggressive format conversion. These remain alternative, unfrozen design proposals, not established standards. The discovery runs are not retrospectively assessed against either criterion. Only the dated, frozen confirmatory manifest is authoritative for the selected baseline, domain, workload, resource endpoint, practical margin and statistical decision rule.

For a KV/routing candidate, distinguish physically accessed KV bytes from retained cache bytes, including GQA sharing and index metadata, and report a quality/latency/memory Pareto curve. The current routing candidates retain the full real cache and therefore do not establish KV compression. For a learned FFN component, include all component parameters and decoder overhead. A claim may concern a specific useful operating point rather than replacing the complete LLM.

Use paired document-level measurements and bootstrap intervals when defensible; token correlations and training-seed variation must not be hidden. Never treat a validation subset or repeated prompt completions as independent full-benchmark samples.

## Reproducibility and publication gates

Store commands, source hashes, model/data revisions, package versions, hardware, seeds, raw per-example measurements and failure cases. Lock confirmation configurations before running them. Report all predeclared operating points, including failures.

Paper completion requires: coherent theoretical exposition; a defensible contribution relative to nearest work; implemented candidate; verified controls; appropriate replicated experiments; uncertainty and limitations; reproducible artifacts; and an internally consistent, compiled manuscript. Writing a plan or completing a synthetic pilot is not completion.
