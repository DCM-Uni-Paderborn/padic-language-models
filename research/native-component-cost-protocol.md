# Prospective native readout component cost study

Design fixed on 2026-10-04 while the PG-19 quality evaluation is running,
before reading its quality estimates or any new component timing. This is a
separate experiment, not a retry of the closed failed address-identity gate.
It is not yet an executable/source-frozen hardware result. Freeze the final
source and inputs after a correctness-only preflight and before timing.

## Workloads and prerequisites

Use the first selected PG-19 book, original layer0 QKV/output weight from the
new frozen native-quality study, at complete2048 and4096 contexts. At each
context use the last query of the half, three-quarter and full prefix: six
fixed workloads, including the two independently archived2048 prefixes.
Keep all12 native methods: full, recency, uniform, and p2/p3/coarsened-p2 for
seeds17/29/43. Add a thirteenth cost variant, recency_view, which reads the
last128 physical keys/values directly as a contiguous-position view, avoiding
redundant ID construction/gather. Its exact BF16 output must match the same
archived recency quality control. This stronger fixed-window implementation
was added in a design review before any cost preflight or hardware timing;
it changes no frozen quality method or decision. There is no fastest-seed,
position or implementation selection.
Original eager attention is a quality reference, not the optimized hardware
baseline. Force Flash SDPA for both full and gathered attention.

Before measuring, require the PG-19 data, numerical and statistical audits to
complete successfully. A family failing its quality screen remains in the
cost study; its speed cannot establish an acceptable-quality operating point.
Do not change the model, addresses, quantile thresholds, precision,128-key
budget, recent8 quota, three-head grouping or newest-tie rule.

Use batch1,15 query heads, five physical KV heads, dimension64 and one BF16
query. Each workload has a contiguous BF16 physical KV prefix and current
Q/K/V vectors copied from its archive. It is a steady snapshot replay of
one component on sequence-derived QKV, not full cached model generation.
QKV projection, RoPE, other layers, prefill, evolving generations and cache
eviction are outside this measurement. Require exact archived codes, latents,
selected IDs and BF16 head/projected outputs for all six workloads and all13
cost variants before timing. The direct recency view has no selected-ID
tensor, and must match the recency head/projected bits. A mismatch closes the preflight without performance
measurement. This checks the declared contiguous workload against its native
quality operator, not against the failed older batch-address identity target.

## Timed operations

The primary complete call writes the current real K and V into their fixed
last cache slot. Every finite method then repeats the current physical key
across its three query heads, computes actual t=1 query/key addresses with
the unchanged FP32 affine/FP64 book bridge, writes the new packed key codes,
scans the complete older code prefix for the fixed shell-rank selector,
gathers physical K/V, calls one-query Flash SDPA and applies the original
BF16 output projection. The full baseline performs the same real K/V write
and full-prefix readout; recency/uniform perform their actual ID construction.
The additional recency_view control performs the same writes and reads only
the trailing128-position views without ID construction or gather.
Repeated calls overwrite identical last-slot values rather than pretending
to advance a cache. All inputs and state remain on the GPU during timing.
The original model is not resident: only this component's archived inputs,
output weight and routing state are loaded. Warm repetitions allow these
small arrays to remain in hardware caches; no cache flush is performed.
Their behavior cannot represent cache pressure from a complete model.

Separate descriptive phases are address/update (finite methods), selection,
and readout with already selected IDs (all sparse methods). The last phase
omits routing and update costs and is explicitly an upper control for the
possible benefit of selection. Phase measurements are separate executions;
their times must not be summed as a replacement for measured complete calls.
For recency_view, the sole diagnostic phase is readout_only, which excludes
the K/V writes and otherwise uses the same trailing views.
Cache/state construction is recorded separately as a process clock, not
included in steady-state latency. No kernel fusion, compilation, CUDA graph,
precision change or result-dependent tuning is introduced in this study.

## Sampling, clocks and interruption

Use seven hardware blocks. For each block, permute the78 complete workload/
method cases with a single NumPy PCG64 generator seed804211, fixed before
measurement. Within each case evaluate complete then the applicable phase
controls in the fixed order above. Each phase receives20 warmup calls,
100 individual synchronized wall-clock samples and100 separate CUDA-event
samples. Wall timing synchronizes before starting and after the call; event
recording is excluded from wall-clock samples. Include Python dispatch and
synchronization in the stated wall latency. CUDA-event elapsed time is a
separate device-stream interval, which can include gaps from host dispatch.
The complete and applicable diagnostic phases total378000 raw samples and
1890 phase summaries. Preserve every sample, block/case/phase order and failure; never report the
minimum timing or treat hardware repeats as independent quality observations.

Report wall and event medians/p95, all seven complete-call block medians,
and ratios to the matched full method within each block/workload. These are
descriptive hardware repeats, not population confidence intervals. Any lower
latency claim must show every prescribed seed/workload and its spread. Pair
quality only with the matching native-quality screen; extrapolation to
multilayer/model tokens per second is unsupported.

Run Spark via its configured ProxyJump uts.hzdr.de alias, with strict host-key
verification and disabled credential forwarding. Start with an empty GPU
compute-process list; record the list and full device telemetry before and
after each hardware block. If another compute process is observed, preserve
partial samples and close the study as interrupted. These checks cannot
exclude transient unobserved interference. Change no other running job,
device clocks or power limit. Use a separate cumulative3600-process-second
cap for cost preflight, measurement and independent result audit. No cap
extension, subset confirmation, performance retry or post-result tuning.

## Storage and independent verification

For each workload/method record exact logical payload bytes and unique GPU
tensor storage bytes for current Q/K/V, physical KV prefix, output weight,
encoder, books, lookup tables, packed code prefix and all auxiliary state.
Original real KV stays resident. Record warm allocated/reserved PyTorch
bytes, complete-call peak allocated/reserved bytes and the allocated increase
above the warm baseline in a separate untimed probe after warmup. Account
separately for auxiliary already-selected IDs used only by phase controls.
Record global process/device memory telemetry and host process peak RSS with
their scopes; allocator counters are not complete driver memory or measured
DRAM traffic. Do not infer compression from128 selected positions.

Archive source/input hashes, immutable protocol/config, environment, the
correctness preflight, all timing samples, storage/probe records, telemetry
and terminal completion/failure. An independent auditor that imports no
production component code must verify the full artifact manifest, prescribed
sample counts/order, finite positive clocks, all summary medians/p95 and
matched baseline ratios, unique-storage accounting and complete-memory
inequalities. Numerical/source and timing-audit failures block a supported
cost claim, while retaining all raw evidence.

PyTorch's primary documentation supports synchronization/event timing and
the scope of its allocated/reserved counters: [CUDA semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html),
[CUDA Event](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.Event.html),
and [peak allocated tensor memory](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.max_memory_allocated.html).
The workload and decision rules above are our prospective choices; they are
not recommendations or performance guarantees from those sources.
