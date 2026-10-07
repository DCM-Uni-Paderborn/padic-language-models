# Continuous p-adic hull: independent mechanism reproduction

Fresh protocol, 2026-10-04, before scientific learning outputs. Spark through uts.hzdr.de is the compute host. This does not reopen any earlier failed gate.

Source: Salazar et al., Continuous Optimization for p-adic Models, arXiv:2609.25501v1 (21 September2026), https://arxiv.org/html/2609.25501v1. Upstream https://github.com/google-deepmind/padic-ml inspected at53ffd9e79890e76b82ef2dc3c7230984053f9801. Its README explicitly describes work in progress; this snapshot supplies arithmetic primitives, not a released optimizer. Our implementation is independent, with no copied upstream training code.

We reproduce the mechanism, not the paper's full experiment tables: affine disk states with exact integer centers over p², real positive radii, analytic one-sided derivatives and next-vertex clipping. We enumerate joint incident directions for the three parameters of one output, rather than the paper's approximate one-coordinate-per-group queue. A monotone training-loss backtracking rule accepts steps; no momentum, Adam or validation tuning. Centers cannot overflow int64 for these bounded inputs/depths; descending stops at the predeclared precision, while upward steps remain allowed. Equivalent disk centers must give equal scores. The ordinary quotient-tree interpretation is an exact comparator, not a novel advantage claim.

First, the paper's illustrated scalar path: p=3, target23, initialcenter14 andradius2/27, learningrate8/81,20updates, maximumdepth6. Gate requires finalcenter23 andradius at most3^-6 with independently verified losses.

Second, a new affine learning test, not the paper's selected adverse initialization: seeds17/29/43, three targetcoefficients drawn uniformly0..242;512fresh integer covariates in0..80 for each of training, validation and test, plus constant1. Targets x·theta+243eta with independent eta in0..26. All coefficients start at center0/radius1. Joint steepest descent uses learningrate2,100updates, minimumradius3^-5, seeded uniform ties, at most20 backtracks. No targetcoefficients enter fitting. No validation/test results influence updates. Raw datasets, all states, accepted/rejected updates and all endpoints are retained.

Progression requires allthree seeds recover every coefficient modulo243, accept at leastone nonzero update, and have maximum point test error at most3^-5. An independent Fraction-based auditor regenerates the input draws and checks every saved trajectory loss and endpoint. The producer and auditor each have a300-second cap and fresh directories; no automatic restart or extra seed/rate sweep.

Five mathematical fixtures (signed/fractional centers, disk representative invariance, joint max-radius slopes, ordinary quotient equivalence and classification slope finite differences) must pass before freeze. The dated execution manifest pins protocol, learner, producer, auditor and tests before any scientific outputs.

Only successful audited completion permits the separately frozen modular LLM adapter. This gate establishes a small implementable learning mechanism. It does not establish uniquely p-adic benefits, scalability, natural-language reasoning or generative language-model quality.
