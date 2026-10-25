# Pre-registered decision procedure: hc8 judged end to end (DFlash2 verify at 4K)

**Written after 82a1845a's H2 FAIL, citing orchestrator ROUND 297.** The
kernel-bench gate in `docs/preregistration-hc8.md` (82a1845a) failed at
R = 7 and R = 8 on head 3045a1fd (ratio 0.648 and 0.641 against ≤ 0.55, quiet
rerun accepted as valid in ROUND 297). That verdict is final and this file
does not reopen it: no kernel-ratio bar appears below, and 82a1845a's H2
numbers are cited only as context. This file asks a different question, set
in ROUND 297 (2): does the measured ~35% cut in the HC pair at 8 verify rows
pay end to end, in DFlash2 GPU ms/cycle at 4K? The thresholds below are fixed
before any leg of this procedure runs and are frozen once the first leg is
witnessed. No leg runs until the orchestrator accepts this file. Its GPU slot
comes after the fn8t-1b and fn8t-2 legs.

## What is measured

Head 3045a1fd routes 4 < R ≤ 8 verify rows to `hc_down_sk4_n2` +
`hc_up_mix8_c4` (default on; `NI_HC8=0` restores the one-row-per-threadgroup
pair `hc_down_sk2` + `hc_up_mix2`). R ≤ 4 dispatch is unchanged.

Where R = 5..8 occurs in decode (from the code at 3045a1fd):

- DFlash2 chain verify (`DFlash2::encode_cycle_verify`) encodes
  `encode_verify(..., Kg + 1)` = 8 rows at the default 7 proposals. The
  dynamic depth (tau 0.5, MoE target) only sets a GPU-side row limit, which the
  MoE routing reads; `enc_gr_rows` does not, so all HC calls run at R = 8
  whatever depth tau keeps.
- One verify issues 97 HC down+up calls: 48 layers × (`ga` + `gm`) in
  `enc_verify_layers`, plus the final `mix_`. The DFlash2 drafter has no HC.
- MTP at the model's default depth 2 verifies R = 3 and runs its head at
  R ≤ 3 and R = 1. hc8 never engages there and the dispatch is identical in
  both arms.

## Projection that motivated this file (estimate, not a result)

Per-call deltas from the valid ROUND 294 rerun (fn_check hc-time, D 2560, S 4,
rank 320, 6-bit, NL 96, best of 3), × 97 calls per cycle:

| Path at 4K | HC rows | Δ per call (µs) | Projected Δ ms/cycle |
| --- | --- | --- | --- |
| MTP, default depth 2 | 3 | 0 (not engaged) | 0 |
| DFlash2, verify 7 of 7 | 8 | 179.0 − 114.7 = 64.3 | −6.2 (−6.5 scaled to the ledger's in-model 18.169 ms/cycle) |

In-model HC at R = 8 books 18.169 ms/cycle / 97 = 187 µs per call against
fn_check's 179.0, so the kernel bench and the decode leg agree to ~5%. The
projection is 5–6% of the 105–118 ms GPU/cycle of the in-house DFlash2 4K
baseline (`/tmp/df2-gate/run2/dfl4k-{1,2}.err`). It gates nothing; E2 does.

## Arms and conditions

- One binary, head 3045a1fd (or a later head that changes nothing but docs;
  any code change restarts this procedure at E0). Arm A = `NI_HC8=0`, arm B =
  `NI_HC8=1`. The toggle is the only difference.
- Machine: this Mac (MacBook Pro 16-inch 2026, M5 Pro, 48 GB, macOS 26.5.2).
  Model: `models/flash-next-q20-full`, read from the internal disk.
  `NI_EXPERT_STORE=22` on every leg, tagged `store=22 (non-default)`.
- Every GPU leg: its own `lockf -k /tmp/kapitsa-gpu.lock` hold inside
  `detach.py`, preflight `/tmp/kap-preflight.sh once` inside the lock, the
  envelope monitor, `peak_pressure ≤ 1`.
- Quiet rule (from ROUND 294): a timed leg starts only when
  `sysctl -n vm.loadavg` shows 1-min load < 4 and no clang/cc1/make/ninja
  process runs. 1-min load is sampled at 1 Hz during the leg. A leg whose
  peak is ≥ 6 is VOID.
- VOID legs (envelope abort, quiet-rule breach, ROUND 242 failure) get at most
  one retry each. They are not FAILs.

## E0: zero-GPU checks (recorded before any leg)

| Check | Bound | Evidence |
| --- | --- | --- |
| `fn_check compile-only` | exit 0 | build log |
| `ni_kstats --limits hc_down_sk4_n2_b{4,6,8}`, `hc_up_mix8_c4_b{4,6,8}` | ≤ 16,384 B static threadgroup memory each | ni_kstats output |
| H1 of 82a1845a on 3045a1fd | PASS (48/48 bitwise under the validation layer) | `/tmp/hc8-gate/run-3045a1fd/h1.out`; stands per ROUND 294 |

## E1: logit identity (logit_dump, both directions)

`tools/logit_dump <model> <out> --paths step,verify --chunk 2048`, 192 rows
(8 corpus files × 24), at `--spec 7` (8 verify rows: dumps a, b, a2) and at
`--spec 6` (7 verify rows: dumps a, b). `records.txt` and `sequences.txt`
hashes are equal across the dumps. PASS: `tools/logit_compare.py` shows every
row identical in both directions for every pair (max |Δlogit| = 0, top-1
agreement 1.0, every KL and NLL cell 0) and the `logits.f32` sha256 values are
equal. a vs a2 proves arm stability. Any non-zero cell is FAIL-bug and the
battery stops before E2.

## E2: DFlash2 verify ms/cycle at 4K (ABBA), the gate

Order: `w4k-plain` (plain greedy reference for `--ref`; it also warms the page
cache, untimed), then a1 b1 b2 a2, each
`ni_run <model> @/tmp/df2-gate/ids/w4k.ids 256 0 --dflash auto --ref <w4k-plain out>`,
temperature 0, no `--ignore-eos`.

- Identity: the generated-ids md5 is equal across a1 b1 b2 a2 (and so is the
  cycle count). Any divergence is FAIL-bug.
- ROUND 242 validity: 256 non-empty ids, no 248044/248046 in the first 256
  output ids, ≥ 256 tokens before the first EOS special
  (`/tmp/df2-gate/eos.py`), and the decoded prefix recorded. Otherwise the leg
  is VOID.
- Metric: ni_run's `ms GPU/cycle` from the `dflash decode` line.
- PASS-perf: A − B ≥ 3.0 ms/cycle in both pairs (a1 − b1 and a2 − b2). The
  3.0 ms bar is the ROUND 297 "real share" line, about half the projection.
- FAIL-perf: either pair below 3.0 ms/cycle.
- INCONCLUSIVE: within-arm spread over 3% (|a1 − a2| / mean(a), or the same
  for b). The whole ABBA runs once more; the rerun's verdict is final, and a
  second INCONCLUSIVE is FAIL-perf.

Reported, not gated: tok/s, wall ms/cycle, the HC dispatch count from E3.

## E3: engagement evidence

Arm B's stderr contains `[kapitsa] NI_HC8: hc_down_sk4_n2_b6 + hc_up_mix8_c4_b6
on 8 verify rows (rank 320)` once per process; arm A's does not. A missing or
misplaced line is FAIL-untraceable.

## Not run under this procedure

No 32K legs, no `NI_PROFILE=2` legs and no kernel bench. Any of these needs
a separate orchestrator grant. Their results do not change this verdict.

## FAIL definitions and the decision

- FAIL-bug: any bitwise or md5 deviation in E1 or E2.
- FAIL-perf: E2 misses the 3.0 ms/cycle bar in either pair, or two
  INCONCLUSIVE ABBAs.
- FAIL-untraceable: a missing dump, cell, md5, load log or envelope log, or E3.
- PASS: E0 through E3 all pass. A PR may then be opened, shipping `NI_HC8`
  default on, with the E2 table, the ABBA md5s, the vk checks output and a
  Runs: line (Mac, macOS, commit, model). The PR says the saving applies only to
  5–8 verify rows (DFlash2, MTP `--spec 4..7`), not to the default MTP depth 2,
  and reports the decode saving as measured in E2. Merging waits for the
  orchestrator.
- Any FAIL parks the branch (pushed, no PR, `PARKED.md` with the verdict and
  evidence paths). No further reruns.
