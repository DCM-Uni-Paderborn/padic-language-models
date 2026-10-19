# Low-precision representations and quantization-aware continuation

The scientific question is whether few-digit p-adic representations, decoded into a real pretrained LLM, yield useful quantization under matched training and precision budgets. Speed is secondary. No hardware kernel or exclusive p-adic advantage follows from code-alphabet performance.

## Fixed methods

Quantize all seven attention/MLP projections in every Transformer block of the pinned SmolLM2-135M checkpoint. Keep embeddings, the tied output head, normalization, nonlinear operations and residual tensors in BF16, counting those bytes. Groups of64 weights have learned FP32 scales. Activations are quantized per token/group with one learned clipping factor per projection. FP32 shadow weights, straight-through gradients, AdamW and FP32 matrix accumulation are shared by every method. There is no small modular accumulator or wraparound. This is numerical QAT emulation rather than packed low-bit hardware.

Compare symmetric INT2/3/4, full signed INT2/3/4 controls, finite residues in base3 and5, and unit/valuation alphabets in base2/3/5. A finite residue alphabet uses the largest p^k not exceeding2^b and balanced real decoding. The valuation alphabet contains zero and u*p^e, with complete signed unit representatives modulo4 for p=2 or modulo p for odd primes. Use the largest exponent count E for which1+E*number_of_units<=2^b. Enumerated b-bit indices jointly encode unit and exponent. This defines a finite p-adic state set; nearest-real encoding and real matrix arithmetic do not by themselves exploit an exclusively p-adic metric.

Include the E2M1 FP4 codebook at4 bits under identical group scales. It is a numerical codebook control, not native NVFP4 with its separate hardware/scaling recipe. A three-level W2/A8 control uses the same ternary alphabet as finite base3 and is not a reproduction of BitNet's architecture or large-scale pretraining.

The core regimes are W2/A2, W3/A3, W4/A4, W2/A8, W3/A8, W4/A8, W3/A4 and a mixed W2/W4 alternating-block schedule with ordinary A4. Alternating blocks have exactly the same average weight code budget as W3 because every block has the same projection shapes. All quantizers are applied in every block, not one diagnostic projection. All ordinary baselines receive the same higher-precision exceptions and metadata.

Mathematically identical codebooks/schedules are fitted once and retain all requested method labels as exact aliases. Independent scalar quantization, congruence and gradient fixtures must verify this equivalence. Report nominal code width, alphabet cardinality, alphabet entropy, scale bytes and unchanged BF16 bytes separately.

## Data and continuation

Use the existing pinned WikiText dataset and tokenizer. Select128 disjoint256-token training article prefixes and16 development article prefixes from the official training split, with disjoint article hashes and exclusion of earlier adaptation articles. Selection uses a fixed content-hash order. Keep the official validation/test splits out of parameter updates. Score the existing40 fixed WikiText test article prefixes and40 fixed PG-19 book prefixes, truncated to256 tokens for this full-model study. These corpora have been used in earlier component studies; they are held out from this continuation, not newly untouched evidence.

Use128 update steps, batch2 windows and sequence length256:65536 input tokens per fit. Use seeds17/29/43 and identical minibatch schedules across methods. Freeze the original teacher. Optimize the equally weighted sum of next-token cross entropy and teacher KL at temperature1. Weight learning rate1e-5, scale/clipping learning rate1e-3, AdamW betas(0.9,0.999), epsilon1e-8, zero weight decay, global gradient norm1. No test-selected rate, precision or checkpoint. The final fixed update is the endpoint; all PTQ and final QAT outcomes are reported. Retain compact optimization and per-document loss summaries. Bulk shadow weights and teacher logits can be regenerated.

## Endpoints and confirmation

Primary numerical evidence is full-model target NLL/perplexity before and after continuation, paired with the matched trained integer baseline. Reconstruct all whole-document aggregates and paired20000-draw bootstrap intervals, seed514203. A method is called quality-retaining only if every fixed seed has a97.5% upper paired NLL bound below log(1.10) relative to BF16 on both corpora. Superiority to the corresponding integer baseline requires upper bound<0 on both corpora in all three seeds. These engineering rules do not establish population-wide superiority.

The second-model confirmation uses SmolLM2-360M, the same inputs and seeds, W4/A8 for INT, FP4 and both base2/base3 valuation representations. Its role is model-size transfer. This remains one model family and a bounded short continuation, not a large-scale BitNet/ParetoQ replication. Report all negative outcomes and do not promote a failed small-budget run to a general impossibility result.

Each model fit has a fixed1800-second wall cap. A numerical or integrity failure closes that run and leaves its diagnostic record. Infrastructure corrections must be recorded separately from scientific outcomes. Substantial work uses established remote project directories through uts.hzdr.de, with host-key verification and no credential forwarding.
