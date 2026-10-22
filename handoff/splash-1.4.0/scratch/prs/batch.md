## What changed

When 2–4 requests decode at once, `Engine::batch_rows` gave every member a fixed number of verify rows, measured once on
the M4 Pro: 4 each at 2 members, 2 each at 3 (or turns on the 27B), 2 each at 4, at most 8 in all. The batched path also
kept DFlash2's chain walk.

KFlash's budget now picks those rows every batched cycle (`src/model/kflash_budget.*`, `src/engine/engine.cpp`):

- **Rule:**
  - Every member verifies its anchor.
  - Drafts are then taken in decreasing calibrated survival across the batch: the chance that the draft and those
    before it are accepted. Each must pay for one more row at this Mac's measured cost of a batched cycle, by members
    and rows, up to 8 rows in all.
  - Whether a draft is verified depends only on drafts ranked above it, never on its own draw, so sampled output stays
    exact.
- **Calibration:**
  - Greedy members share KfBudget's per-depth calibration (the same quantity as the single-stream tree's) and feed it
    the batched labels.
  - Members sampled at T > 0 are accepted by exact speculative sampling, a different event, so they get a calibration
    of their own.
- **Cost:**
  - The cost table code moved out of KfBudget unchanged; KfBudget's outputs and saved files are bit-identical to before.
  - Costs are read as a robust line.
  - The stalest row count is re-measured every 64 cycles. An earlier version locked onto a table measured while the GPU
    was overloaded.
- **Where it runs:** the decision is made on the CPU between the drafting passes and the verify. The verify's first
  layer is sent as soon as it is encoded; the GPU then waits 0.18–0.31 ms per batched cycle.
- **Switches:** `NI_BATCH_KF=0` restores the fixed table. `NI_BATCH_KF_FLOOR=1` (one draft per member at least) cost
  2.5–3.9% at 4 requests, so it is off.
- **Tool:** `tools/batch_check` now takes a row count per sequence and runs the split verify pass. It is the second
  commit, with no engine change.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1. Qwen3.6-35B-A3B and Qwen3.8-27B 4-bit with
their DFlash2 drafters, `--slots 4`. Base: main 0553f55. `bench/batch/concurrency.py streams`, 512-token replies,
3 interleaved rounds per cell. Tok/s in all, mean (range):

| | greedy | T = 0.6 |
| --- | --- | --- |
| 35B, 2 at once | 190.0 (184.8–194.4) → 192.8 (191.9–193.9), +1.5% | 193.5 (190.6–195.7) → 194.0 (178.1–209.5), +0.3% |
| 35B, 3 at once | 170.8 (166.3–176.4) → 205.7 (204.5–206.8), **+20.4%** | 170.6 (167.6–174.8) → 197.7 (190.0–206.3), **+15.9%** |
| 35B, 4 at once | 191.9 (185.2–196.9) → 199.8 (198.5–201.3), +4.1% | 188.6 (184.2–191.6) → 195.0 (189.2–199.5), +3.4% |
| 27B, 2 at once | 63.5 (60.3–67.0) → 65.5 (63.3–68.6), +3.1% | 66.7 (63.7–68.5) → 65.9 (64.1–68.2), −1.3% |
| 27B, 3 at once | 57.7 (54.5–61.3) → 66.4 (64.5–69.7), **+15.1%** | 53.1 (50.5–55.4) → 65.2 (62.7–66.8), **+22.7%** |
| 27B, 4 at once | 61.9 (58.9–67.2) → 64.4 (61.7–67.0), +4.1% | 64.6 (61.3–67.6) → 65.2 (61.9–67.0), +1.0% |

The 3-at-once gain comes from using all 8 rows. At 2 and 4 requests the change is within the runs' spread.

- **A request alone:**
  - Same tokens on both models: 7 of 7 requests identical, greedy and seeded.
  - 35B per-prompt medians 185.7 → 184.0 tok/s; the same KFlash state on both builds gave 170.0 → 171.3.
  - 27B medians 55.6 → 56.0 tok/s.
- **Prefill:** unchanged. 35B time to first token 3.56 vs 3.56 s at 4K and 16.67 vs 16.68 s at 16K.
- **Replay estimate made before building**, from lattice recordings of 6 prompts, tokens per cycle against the fixed
  table: 35B +3% / +31% / +4.8%, 27B +4.1% / +33% / +2.5% at 2 / 3 / 4 requests.
- **Requests together vs alone** (greedy, 13 requests):
  - 35B: 6 of 13 identical on this PR vs 8 of 13 on main. Every first difference falls at a single-stream logit margin
    below 0.39 (median 0.018), so all are near-ties, as docs/architecture.md allows for requests decoded together.
  - 27B: 12 of 13 identical on both builds.

## Checks

- **CPU:** `api_test`, `test_server.py --cli`, `test_sidecar.py`, `test_bench.py`, `test_update.py`,
  `test_community.py`, `test_reasoning.py`, `test_pr_size.py`.
- **New `dflash_accept_test --cpu` cases:** the rule against brute force, a draft's place never depending on later
  drafts, the cost line, and the exploration, cache and calibrations.
- **Under `MTL_SHADER_VALIDATION=1`:**
  - `batch_check` with uneven row layouts on 2–4 sequences: no errors, argmax equal in 288 of 288 rows.
  - The server's mixed batch plan.

## Not verified

- Bolt / LiLiCorr, which this PR doesn't touch.
- Other chips.
- A statistical test of sampled output from the batched path; it rests on the ordering test and the existing acceptance
  tests.
- `MTL_DEBUG_LAYER=1` aborts in prefill before reaching this code: `attn_prefill11g` asks for 40 KB of threadgroup
  memory against a 32 KB limit. This also happens on main.
- One exactness run saw a branch server crawl at about 5 s per batched cycle while the system's GPU monitor also
  stalled. The system log shows no GPU restart or fault, and it did not reproduce in two re-runs, one under shader
  validation. The cause is unknown.

The idea of choosing rows across a batch follows Splash 1.4.0's verify budget (Apache-2.0); no code was taken.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
