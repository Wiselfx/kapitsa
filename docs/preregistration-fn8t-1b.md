# Pre-registered decision procedure: Flash-Next 8-row tensor verify tiles (fn8t-1b)

This file fixes, before any measurement exists, the acceptance gates for the
ROUND 239 step-1b kernel work: the 8/16-row `tensor_ops` verify tile for Q2_0
experts (`qmt8v_*`) and the dense verify projections on the tensor units
(`qmm6t` on the f6g64 slab), engaged only for the Flash-Next q20 verify path
at R = 8, behind `NI_FN8T_D` (dense) and `NI_FN8T_E` (experts), default off.
It follows the qmt6/#510 style recorded in AGENTS.md ("Changing speed or
numerics"): thresholds fixed in advance, evidence traced to dumps, a stated
FAIL condition, and no post-hoc threshold movement.

## Class claim, and why the floor's gate fails by construction

Both kernel families keep every product and reassociate each output's K
reduction onto the tensor units' `matmul2d` of codes — the `--reorder` class —
with one departure: activations are reconstructed from a hi/lo half pair
(`x = hi + lo`, both products exact half×half in fp32), so the computation is
a reassociation plus a ~2^-22 relative representation residual, not a pure
reassociation. The tensor path does not run the simdgroup kernels' fp32
chain, so `tools/logit_floors/flash-next-q20.m5.json`'s all-zero bitwise
floor fails by construction with the flags on, exactly as recorded for #570.
The gates below therefore bound the drift against the scalar path instead of
requiring bitwise equality, and they reuse the floor file's own
`reorder_gate` — the user-governed bounds for order/representation-class
changes on this model — which is the #581 procedure ("accepted when every
measured cell is inside the model's own gate where a floor exists").

## Arms

- One binary per PR head; arm A = `NI_FN8T_D=0 NI_FN8T_E=0`, arm B =
  `NI_FN8T_D=1 NI_FN8T_E=1`. The flag toggle is the only difference.
- Machine: this Mac (MacBook Pro 16-inch 2026, M5 Pro, 48 GB, macOS 26.5.2).
  Model: `models/flash-next-q20-full`. Engagement guard: Apple GPU family
  ≥ 10 and macOS ≥ 26, logged once per process.
- Quiet legs only (`peak_pressure = 1`, `lockf` on `/tmp/kapitsa-gpu.lock`,
  at most one retry). An envelope abort (`peak_pressure > 1`) makes the leg
  VOID and rerunnable; it is not a FAIL.
- Arm B runs the hi/lo two-term tables. Single-term is a diagnostic build
  only (`NI_FN8T_LO=0`) and never produces gate evidence (G5).

## G1 — 192-row drift dump (logit_dump + logit_compare.py, B against A)

`tools/logit_dump` on the step and verify paths, 192 rows, chunk 2048, same
prefixes in both arms (`records.txt`/`sequences.txt` hashes equal); compared
with `tools/logit_compare.py`'s arithmetic. PASS requires every cell:

| Cell | Bound | Provenance |
| --- | --- | --- |
| top-1 agreement | ≥ 0.989583 | floor file `reorder_gate.min_top1` |
| far flips (top-2 margin ≥ 0.551) | 0 | `reorder_gate.max_far_flips` |
| every flip's top-2 margin | < 0.551 | `reorder_gate.near_tie` |
| mean KL | ≤ 0.00184 | `reorder_gate.max_mean_kl` |
| median KL | ≤ 0.000101 | `reorder_gate.max_median_kl` |
| p99 KL | ≤ 0.0283 | `reorder_gate.max_p99_kl` |
| \|mean NLL change\| | ≤ 0.0349 | `reorder_gate.max_nll_change` |
| max \|Δlogit\| | ≤ 7.48 | 2× the floor file's worst pure-reassociation source `max_abs_diff` (3.7395, `NI_QMT=0 NI_MATH_SAFE=1` step), following the file's own twice-the-worst-source construction |

Expected band, recorded not gated: hi/lo drift should land ~1e-6 mean KL,
three orders inside these bounds. Measured values are booked in full either
way; a PASS at the 1e-3 scale would indicate the lo term is not engaged and
is caught by G5, not by widening G1.

## G2 — per-GEMM bench (bench_moe-style, both families)

For `qmm6t` against `qmvs_multi_f6g64` and for each `qmt8v` stage against
its `moe_q20v_*` scalar kernel, at the verify shapes (R = 8; dense
K = 2560, N ∈ {512, 640, 3584, 2560}; expert I = 640, D = 2560) plus the
M/N tail classes:

- every tensor output within one half step of the scalar path's output;
- both paths' max error against an fp64 reference within 2× of each other;
- the hi/lo build's max error against fp64 ≤ the single-term build's;
- small-shape dispatches layer-clean under `MTL_SHADER_VALIDATION=1
  MTL_DEBUG_LAYER=1` (static threadgroup memory ≤ 16,384 B per kernel, since
  the layer double-counts against the 32,768 B cap; a kernel above that says
  so in the PR with its size, per AGENTS.md).

## G3 — greedy continuations (ABBA, 256 tokens, w4k/w32k chat prompts)

`/tmp/df2-gate/ids/w4k.ids` and `w32k.ids`, temperature 0, no
`--ignore-eos`, 256 output tokens, arms interleaved A B B A, n = 2 legs per
arm per length, one binary.

- Kernel PRs (flags default off): PASS when the ids md5 is identical across
  all legs at each length. A mismatch is tolerated only when every divergent
  position's top-2 margin in the A arm is < 0.551 and at most 2 positions in
  256 diverge per length; otherwise FAIL.
- The default-on flip PR: the ids md5 must be identical across all legs at
  both lengths, strictly. Any mismatch blocks the flip; no conditional tier
  applies. (#585 reverted a tensor path because one continuation moved to
  `<|im_end|>` and looped; a flip that changes continuations is not this
  change's claim.)
- Every leg carries the ROUND 242 output check: 256 non-empty ids, zero
  248044/248046 among the first 256 output ids, ≥ 256 tokens before the
  first EOS special, decoded prefix in the ledger. A leg without it is VOID
  and is not booked.

## G4 — floor inviolability (flags off)

At every PR head, with both flags off, the flash-next-q20 floor leg passes
bitwise 192/192, and `flash-next-q20.m5.json` is unmodified in the diff. Any
bitwise failure with the flags off blocks the PR: it is a regression in a
shared path, not a gated departure. The floor file is never re-measured or
edited by this work; a re-measure with the path on is a PR of its own, per
AGENTS.md.

## G5 — hi/lo engagement evidence

Arm B's leg logs carry the once-per-process engagement line showing the hi/lo
two-term prep ran (lo tables written). Gate evidence from a build or leg
without that line is void. `NI_FN8T_LO=0` (single-term) exists for
diagnosis only and is never a gate arm.

## FAIL

Any cell of G1–G5 exceeded, missing, or untraceable to a dump is a FAIL. On
FAIL: `NI_FN8T_D`/`NI_FN8T_E` stay off, no flip PR is opened, the dumps and
the comparison output go to the orchestrator, and the thresholds in this
file do not move. Thresholds are immutable once the first gate leg is
witnessed (results posted to the ledger); changing any threshold requires a
new orchestrator ROUND that says so explicitly, before new legs run. There
is no deadline auto-ratification branch: unlike #510's standing
"default on, don't ask me again" ruling, the flip here waits for the
orchestrator's explicit PASS ruling on the booked evidence. A late explicit
user decline supersedes everything in this file and is executed as a
one-line default-off revert (`NI_FN8T_D=0 NI_FN8T_E=0`) without further
discussion.

## Flip PR requirements (after a PASS ruling)

- Opened via the verify-kapitsa procedure with vk checks output for the head
  sha in the body.
- Body states plainly: not bitwise against the floor; accepted under this
  pre-registered procedure, citing this file's commit sha; every G1–G5 cell
  with its measured value and dump trace; the ABBA md5s; the Runs: line
  (Mac, macOS, commit, model).
- Performance evidence (booked, not a numeric acceptance gate): ABAB tok/s
  at 4K and 32K on quiet legs, and the fixed-clock `NI_CYCLE_TRACE`
  decomposition (draft / store-stall / verify-compute) riding the first dfl
  leg. First milestone: verify ms/cycle at or below the MTP tie line
  (~92 ms/cycle 4K, ~87 ms/cycle 32K); end goal ≤ 45 ms/cycle at 8 rows.
- Store-stall exposure clause (orchestrator ROUND 270): if verify-compute
  falls below ~45 ms/cycle and the expert-store stall (~24 ms/cycle on quiet
  legs) becomes the largest share of ms/cycle, that split is reported to the
  orchestrator with the flip evidence, and the Store S2/S3 work is ordered
  before any step-1 follow-on. The clause does not gate this PR; it re-orders
  what comes next.

## Provenance of this procedure

Ordered by the orchestrator (ROUND 270) before any fn8t GPU leg runs.
Threshold sources: `tools/logit_floors/flash-next-q20.m5.json`
(`reorder_gate`, `reorder_sources`), AGENTS.md acceptance entries #510
(qmt6, nll-cell departure and ABBA-bitwise evidence), #570/#585 (tensor path
reverted on a continuation change), #581 (per-cell inside-the-model's-gate
procedure). Design scope: `/tmp/df2-gate/1b-design.md` (subagent 128e063b).
