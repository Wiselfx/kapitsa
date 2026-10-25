# Preregistration: Flash-Next automatic expert store leaves room for the KV cache (#601)

Written and committed before any GPU leg of this check (orchestrator ROUND 314). Thresholds below are fixed; a leg that
misses one parks the PR.

## What changes

At `NI_EXPERT_STORE` unset (STORE=auto), the guard that sizes the expert store from the system's available memory now
also subtracts what the process allocates after the store: the KV cache and slot state (and, in the tools, a 16K-token
KV cache plus 1.5 GiB). Before, that memory came out of the 4.3 GB headroom. At 32K on the 48 GB M5 Pro the store was
25.6 GB and the run fell to memorystatus level 9% (`kap_w32k_auto1`, booked NOT RUN). `fn_store_test` pins the same
inputs at 23.5 GB.

## Arms and binaries

- A (old): `origin/main` 64309d33, `ni_run` sha256 9bb9df920dea4dd0.
- B (new): this branch at 974027e2, `ni_run` sha256 a42a97ed372400cf.
- Both: `models/flash-next-q20-full`, STORE=auto, `--spec 2` (the MTP head), 512 tokens, the chat-templated w4k
  (4048 tokens) and w32k (31617 tokens) prompts of the fnx-dec battery.
- M5 Pro 48 GB, one leg at a time under `/tmp/kapitsa-gpu.lock`, envelope R280/R282 (abort: swap delta over 3072 MB,
  swap used over 12288 MB, pressure over 2, memorystatus level under 10% with no other process over 1 GB), load
  recorded per R292. One attempt per leg, no retries.

## Order (45 minutes at most)

1. Warm-up: A, w4k, 32 tokens. Discarded, so that neither arm of the first pair reads a cold page cache.
2. w4k A, B, B, A.
3. w32k B.
4. w32k A, only if 10 minutes remain. Diagnostic: it shows whether the old binary still falls under the floor; it
   decides nothing.

A leg is skipped and booked NOT RUN when fewer than 9 minutes of the 45 remain.

## Gates (all three must pass)

- **G1, 32K runs at the default.** The w32k B leg completes with no envelope abort (level_min at or above 10%,
  pressure at most 2), passes the ROUND 242 ids check (512 ids, none of 248044/248046 before the 512th), decodes at
  50.6 tok/s or more (within 5% of the STORE=22 reference 53.3), and prefills at 632.2 tok/s or more (within 5% of
  665.5).
- **G2, no cost at 4K.** In each pair (A1, B1) and (B2, A2), B's prefill tok/s and decode tok/s are each within 3% of
  A's. The ids of all four w4k legs are identical (the store's size decides what is resident, not what is computed).
- **G3, the store is the size the test pins.** Each B leg logs its store size, and the reason line names the KV
  reserve. The size equals the logged available memory less 4.3 GB headroom less the logged reserve, to 0.2 GB (the
  four figures are each logged to 0.1 GB). When the
  logged available memory is within 0.5 GB of the test's 29.9 GB, the size is between 23.0 and 24.0 GB.

## Outcome

The run writes `/tmp/fnx-full/STOREFIX-DONE` when G1, G2 and G3 pass, and `/tmp/fnx-full/STOREFIX-FAIL` with the
failed gate on every other exit (a gate miss, an aborted B leg, an A leg abort that leaves a G2 pair unscored, the time
cap, or an error). On PASS, w32k STORE=auto is rerun under the same rules, and that figure replaces each 32K figure
tagged 'store=22 (non-default)'.

## Amendment 2 (ROUND 368): G2 re-run with the old arm pinned

Written and committed before any leg of this re-run. Amendment 1 was the 0.2 GB G3 tolerance (822bbc86).

**What happened.** The first run (04:25 to 04:39 on 26 Oct) wrote `STOREFIX-FAIL`. All its rows are tagged 'ran
without GO (R354 not met)'. Every A leg (origin/main at STORE=auto, store 26.1 to 27.6 GB) aborted at pressure 4, so G2
was not scorable. The B legs ran clean: w32k B prefilled at 650.9 tok/s (-2.2% against 665.5) and decoded at 54.3 tok/s
(+1.9% against 53.3), and the B stores were 22.9 to 24.4 GB. G1 and G3 stay as booked. This amendment re-runs G2 only.

**Arms.** These are the same binaries and shas as above. A is origin/main 64309d33 with `NI_EXPERT_STORE=23` pinned, so
its store is the size B picks and it does not fall under the envelope. B is this branch at 974027e2 with STORE=auto.
Both use `--spec 2`, 512 tokens and the w4k chat-templated prompt (4048 tokens).

**Order (15 minutes at most).** First a warm-up: A, w4k, 32 tokens, discarded. Then w4k A1, B1, B2, A2. Each leg gets
one attempt. A leg is skipped and booked NOT RUN when fewer than 3 minutes of the 15 remain (2 minutes for the warm-up).
The envelope is R280/R282 with the level floor at 10% (`RV_MINLEVEL=10`), pressure at most 2, and load recorded per
R292. The run starts only on the orchestrator's explicit GO.

**Scoring** (`score.py`, in the order applied):

1. Every one of the four legs must be clean: rc 0, no abort, level_min at or above 10%, pressure at most 2. Otherwise
   the result is NOT SCORED.
2. If the two A prefills, or the two B prefills, differ by more than 10%, the result is NOT SCORED (noisy).
3. If a pair's A and B logged store sizes differ by more than 1.5 GB, the result is NOT SCORED (store mismatch).
4. **G2 PASS** requires all of the following:
   - B's median is within 3% of A's median, for prefill tok/s and for decode tok/s.
   - For each metric, the two pairs' B/A changes agree in sign, or both are within 3%.
   - The ids of all four legs are identical.
   - All four legs pass the ROUND 242 ids check.

   Anything else is FAIL.

**Outcome.** The run writes `/tmp/fnx-full/STOREFIX-G2-DONE` with `PASS`, `FAIL: <gate>` or `NOT SCORED: <reason>` on
every exit path. #601 leaves draft only when G1, G2 and G3 have all passed. If the result is PASS, w32k STORE=auto is
rerun under the same rules, as written under Outcome above.
