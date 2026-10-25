# Pre-registered decision procedure: the folded HC pair at 5 to 8 verify rows (hc8)

This file sets the acceptance gates for port plan item A00 (ledger 0zs / I-108)
before any measurement exists. A00 is the hyper-connection (HC) residual on the
Flash-Next verify path at R = 5..8 rows. It follows the fn8t-2 bitwise-class
procedure in `docs/preregistration-fn8t-1b.md`: thresholds are fixed in
advance, every verdict traces to a dump or a JSON cell, and each FAIL
condition is stated. Thresholds are frozen once the first leg (including the
bench) has been witnessed. They do not auto-ratify on a deadline, and the
default is not flipped without an orchestrator PASS.

## What changes

`Qwen4Exp::enc_gr_rows` runs the folded HC pair `hc_down_sk3_n2` +
`hc_up_mix4_c4` only when R ≤ 4. The limit exists because `hc_up_mix4` stages
`threadgroup float hs[4][512]`, and 8 rows of that (17,536 B) exceed the
16,384 B that the validation layer can dispatch. At R = 5..8, which includes
DFlash2's 7-draft verify, the HC residual falls back to the one-row-per-threadgroup
pair `hc_down_sk2` + `hc_up_mix2`. That pair launches about 16× as many
threadgroups and re-reads the weights once per row. The ledger books it at
18.169 ms/cycle at R = 7 on 4K (hc_down_sk2_b6 9.259 + hc_up_mix2_b6 8.877).

This change adds `hc_up_mix8_c4_b{4,6,8}`, the same template as
`hc_up_mix4_impl` with `hs[8][384]`. It uses 13,184 B of static threadgroup
memory (`ni_kstats --limits`) and admits rank ≤ 384; Flash-Next's rank is 320.
`enc_gr_rows` routes 4 < R ≤ 8 to `hc_down_sk3_n2` + `hc_up_mix8_c4` behind
`NI_HC8` (default on; `NI_HC8=0` restores the sk2/mix2 pair; `NI_HC3=0` turns
off both). R ≤ 4 dispatch is unchanged, so this does not touch the model's
default MTP depth 2 (R = 3).

## Class claim: bitwise by construction

`hc_up_mix8` is `hc_up_mix4_impl` with larger staging arrays and a row loop
bound of 8. Every output keeps `hc_up_mix2_out`'s products and fma order (the
`m[r] = fma(gb, hsum, fma(gs, d, m[r]))` chain), and its per-row prep
(`inv`, `inj`, `hs`, `hsum`) is computed exactly as in `hc_up_mix4`.
`hc_down_sk3_n2` already loops over `p.R` rows with the per-row stat and the
same per-row reduction. The class therefore admits no drift: any departure from
bitwise identity is a bug, there is no HOLD or cause path, and
`tools/logit_floors/flash-next-q20.m5.json` is not modified.

## Arms

- One binary per PR head. Arm A = `NI_HC8=0`, arm B = `NI_HC8=1`. The toggle
  is the only difference between arms.
- Machine: this Mac (MacBook Pro 16-inch 2026, M5 Pro, 48 GB, macOS 26.5.2).
  Model: `models/flash-next-q20-full`.
- Quiet legs only: `peak_pressure = 1`, `lockf -k /tmp/kapitsa-gpu.lock`,
  one leg per lock hold, and at most one retry. An envelope abort
  (`peak_pressure > 1`) makes a leg VOID and rerunnable. It is not a FAIL.
- No GPU leg runs on a build whose `fn_check compile-only` is not green.

## H0: zero-GPU checks (recorded before any leg)

| Check | Bound | Evidence |
| --- | --- | --- |
| `fn_check compile-only` | library compiles, exit 0 | build log |
| `ni_kstats --limits hc_up_mix8_c4_b{4,6,8}` | ≤ 16,384 B static threadgroup memory | 13,184 B each at commit time |

## H1: kernel identity (fn_check, small sizes, validation layer on)

`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 FN_ROWS=1 fn_check kernels`.
`test_hc_rows` runs D 768, S 4, rank 128, bits 4/6/8, R = 1..8, and compares
`hc_down_sk3_n2` + `hc_up_mix8_c4` against the one-row `hc_down_sk2` +
`hc_up_mix2` per row. PASS: every cell bitwise equal, no validation-layer
error, process exit 0. Any miss is FAIL-bug, and the battery stops.

## H2: kernel bench (fn_check hc-time, Flash-Next shape)

`fn_check hc-time` at D 2560, S 4, rank 320, 6-bit, NL = 96 weight sets, best of
3, R = 1..8, run under the GPU lock. Its stdout is copied to
`/tmp/hc8-gate/h2.txt` and its cells to `/tmp/hc8-gate/h2.json`.

| Cell | Bound | Provenance |
| --- | --- | --- |
| bitwise output vs one-row pair, every R and variant | true | the class claim |
| (sk3_n2 + mix8_c4) / (sk2 + mix2) down+up time, R = 7 and R = 8 | ≤ 0.55 | ledger I-108 estimate 18.169 → ~9.1 ms/cycle (ratio 0.50) plus 10% slack |
| (sk3_n2 + mix8_c4) / (sk2 + mix2), R = 5 and R = 6 | < 1.00 | the route must not lose anywhere it is taken |

A miss on any cell is FAIL-perf, and the battery stops before any dump or
decode leg.

## H3: logit identity (logit_dump, both directions)

`tools/logit_dump` with `--paths step,verify`, 192 rows, chunk 2048. The run
is repeated at `--spec 7` (8 verify rows) and `--spec 6` (7 verify rows),
with the same workload in both arms (`records.txt` and `sequences.txt`
hashes equal). At `--spec 7` there are three dumps: a (arm A), b (arm B) and
a2 (arm A again). At `--spec 6` there are two: a and b. PASS needs
`tools/logit_compare.py` to show every row identical in both directions for
every pair: max |Δlogit| = 0, top-1 agreement 1.0, every KL and NLL cell 0,
and equal `logits.f32` sha256. The a/a2 pair proves arm stability. Any
non-zero cell is FAIL-bug.

## H4: decode legs (DFlash2, ABBA)

`ni_run <model> @<ids> 256 0 --dflash auto --ref <plain out>` on
`/tmp/df2-gate/ids/w4k.ids`, then `w32k.ids`, order a1 b1 b2 a2, temperature 0,
no `--ignore-eos`, and `NI_EXPERT_STORE=22` on every leg.

- Identity: the generated-ids md5 is equal across a1 b1 b2 a2 at each
  context. Any divergence is FAIL-bug, and the 32K block does not run.
- ROUND 242 validity: each leg has 256 non-empty ids, no 248044/248046 among
  the first 256 output ids, and ≥ 256 tokens before the first EOS special.
  `/tmp/df2-gate/eos.py` checks this, and the decoded prefix is recorded. A
  leg that fails this check is VOID.
- Speed: ni_run's GPU ms/cycle. PASS-perf at a context needs B < A in both
  pairs (b1 vs a1, b2 vs a2). When the within-pair spread of either arm
  (a1 vs a2, b1 vs b2) is over 1%, the context is INCONCLUSIVE and gets one
  rerun.
- Diagnostic, not gated: one `NI_PROFILE=2` leg per arm at w4k gives the HC
  family's ms/cycle. Arm A is the in-house baseline that replaces the
  ledger's 18.169, whose source log is not on this Mac. Arm B is the
  post-fix number. Port plan's booked bar is HC ≤ 4.5 ms/cycle, and ledger
  I-131's is ≤ 8.0. Both are reported against arm B and gate nothing.

## H5: engagement evidence

Arm B prints `[kapitsa] NI_HC8: hc_down_sk3_n2_b6 + hc_up_mix8_c4_b6 on <R>
verify rows (rank 320)` once per process. Arm A, and every process that does
not name `NI_HC8`, prints nothing new, so default stderr is unchanged. The B
legs' stderr must contain the line, and the A legs' must not.

## FAIL definitions and the decision

- FAIL-bug: any bitwise deviation in H1, H3 or H4. There is no cause path; the
  fix is code, re-run from a new PR head.
- FAIL-perf: any H2 cell miss, or H4 B ≥ A in a pair at either context.
- FAIL-untraceable: a missing cell, verdict, md5 or envelope log.
- VOID: envelope abort or ROUND 242 failure; at most one retry.
- PASS: H0 through H5 all pass. The PR then ships `NI_HC8` default on with
  the measured table, the ABBA md5s, the vk checks output and a Runs: line
  (Mac, macOS, commit, model). Merging waits for the orchestrator.

## Scope note recorded at commit time

The model's default MTP depth is 2 (R = 3), where the folded pair already
runs and HC is ≤ 0.66 ms/cycle (ledger I-146). This change pays off only
for paths that verify 5 to 8 rows: DFlash2 (`--dflash`, verify 7 of 7) and
MTP `--spec 4..7`. Whether those ship by default is a separate decision
(ledger A08) and is not taken here.
