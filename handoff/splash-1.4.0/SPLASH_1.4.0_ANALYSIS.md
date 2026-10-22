# Splash 1.4.0: what changed, and what Kapitsa can take from it

Sources:
- The source at `engines-src/splash`, tag `1.4.0` (`1482f6bb`, 2026-10-20). `git diff 1.3.2 1.4.0` covers 382 files
  (+48,315 / −6,603 lines) and 260 commits, 55 of them PR merges.
- The release tarball at `splash-release/splash-1.4.0-arm64-macos26/`.
- Splash's DEVELOPMENT.md and commit messages.

Read on 2026-10-21. Nothing was built or run, and no GPU was used. Splash's numbers are their own, as stated in their
commits and docs; most come from an M5 Pro, an M5 Max or an "M6". Kapitsa's numbers come from this repo's docs and
`kapitsa/internal/PERF.md`. "Estimate" marks our own inference. For how Splash works in general, see
`SPLASH_ANALYSIS.md` (written against 1.3.2). Commit ids below are Splash's.

## TL;DR

- **The release has three themes:**
  - Qwen3.8-Flash-Next support.
  - Running models and contexts larger than memory: experts, the n-gram table and KV streamed from the SSD; up to 1M
    tokens with YaRN.
  - A verify-row budget for MoE targets.

  The rest is kernel tuning for M5 and Flash-Next shapes, plus server and cache work.
- **Their verify-row budget is our KFlash budget**, reached independently: drafter confidence calibrated by online
  logistic regression, plus a measured cost per row. They get +2..9% on the 35B. To get it they **gave up run-to-run
  greedy repeatability on MoE targets**, which is a documented behavior change. Kapitsa's version is exact and also does
  trees, so we keep our design.
- **The real lesson is batched decode.** Splash picks every lane's verify rows jointly, from survival probabilities
  across the batch. Kapitsa's batched path still uses chains with a fixed row table (`Engine::batch_rows`: 4/2/2 rows
  per member, 8 in all). Extending KfBudget to the batch is the top item here.
- **Flash-Next is where Splash moved ahead of us:**
  - It runs on 64 GB Macs in 2-bit (Q2_0), 95 tok/s on a 20-core M5 Pro.
  - It runs on 36 GB Macs with streamed experts.
  - It has a trained DFlash2 draft (`incoai/Qwen3.8-Flash-Next-DFlash2`).

  Kapitsa runs Flash-Next only on 128 GB Macs, with MTP at depth 2 (+27% over plain decoding). Its prefill there is
  0.65x oMLX's.
- **Cheap items:**
  - Take the residency set off the command queue. Splash measured 100-170 ms stalls on 1-2% of command buffers when
    memory is nearly full; Kapitsa holds the same queue-wide set.
  - Gate numeric changes on logits against a measured noise floor.
- **Already fine in Kapitsa, no work needed:**
  - Host encode overlaps GPU work.
  - Shared events have one writer each. Splash hit a hang in 1.4.0 from an event that both sides wrote.
  - A stray `</think>` is already dropped.

## 1. What shipped

| Theme | What | Their numbers |
| --- | --- | --- |
| New model | Qwen3.8-Flash-Next ("Qwen4Exp": 48 layers, 36 GDN with a sigmoid gate + 12 QSA sparse attention, 4 hyper-connection streams, a PLE n-gram table, 512 experts with top-10 + shared) on GGUF (Q2_0, UD-IQ3_XXS, ...); DFlash2 draft released (`972dfd70`) | M5 Pro 20c 64 GB, Q2_0: decode 95 tok/s, 32K prefill 1,312 tok/s, cached TTFT at 32K 171 ms, 4 concurrent 171 tok/s aggregate (README; ratios against "Strata" on the same GGUF). Teacher-forced against the reference GGUF implementation, Q2_0: top-1 96.9/96.5/97.3%, mean KL 0.0034/0.0041/0.0051 at 1K/8K/32K (`2192ff77`) |
| Bigger than memory | `--expert-memory`: routed experts stay in their files, with an ARC-recycled slot store (`99dcf73c`, `6d5b97d9`) | Cycle ms at 1/2/4 lanes: Flash-Next Q2_0 with 12G of experts in RAM on an M5 Max, 59/101/177. The 35B UD-Q2_K_XL with 4G on a 24 GB M6, 43-44/62-63/109-117 |
| | `--ple-memory`: Flash-Next's 28.8 GB n-gram table stays in its GGUF, behind a row cache of 512 MiB or less | A 128K prompt reads about 1.1M distinct rows (90 B each in IQ4_NL) |
| | KV on the SSD for block-sparse targets (Flash-Next): a spilled page keeps only its index (24 KiB of 414 KiB), with a hot block cache | A request of 1,010,000 tokens takes 13.5 GB of slots. Outputs are identical whether its KV spills or not |
| | One memory plan sizes weights, stores, KV and states from one budget | — |
| Context | `--max-context` up to 1,010,000 tokens with static YaRN: factor = context / 256K, opt-in, because it changes answers at every length (`1fec2078`) | On an M5 Max the 35B prefills 1M tokens in about 45 min, then decodes at about ¼ speed |
| Speculation | `--verify-rows auto` (default): on MoE targets, each cycle picks each lane's rows by expected tokens per cost (`a8c10adb`, `507f6f3f`) | Resident 35B on an M5 Max, 1/2/4 agentic requests: +8.7/+8.5/+2.1% (GGUF UD-Q4_K_M), +2.1/+0.9/+6.0% (MLX 4-bit) against verifying every row. With streamed experts, +13..+55% |
| Batching | Constrained (grammar) and free lanes decode in one command (`f52d1ad5`) | 2 constrained + 2 free lanes: 35B MLX 4-bit 349 → 457 tok/s; 27B 138 → 210 |
| Runtime | Host steps: CPU work inside one command between GPU work, with an optional GPU spin that keeps clocks up (`51172500`); one-sided events (`e2c0af27`) | Nothing in decode uses them unless something streams |
| | Only GPU-addressed buffers in the command queue's residency set (`c608e23f`) | Before the fix, with memory nearly full, 1-2% of command buffers waited 100-170 ms (M5 Max 36 GB) |
| Kernels | Float projection (router, alpha/beta, indexer) loads 16 B per lane (`d6019714`) | M5 Pro: −31..−45% per projection; 35B router −10..−19% |
| | Lone expert tile of ≤ 2 rows on a column (GEMV) tile (`ed63ffb5`) | 35B MoE layer −5..−9% (M5 Pro); 35B step 29.7 → 27.7 ms (M6, UD-Q2_K_XL) |
| | Narrow Apple10 decode grids on the register tile, not MPP (`97c55fa0`) | 2048×4096 at one lane: 18.5 vs 22.5 µs (M5 Max) |
| | One lane under 4.5 bpw on the 16-row staged tile (`b701328b`) | 27B UD-IQ3_XXS cycle −3.3% (M6) |
| | K split of 16 on the narrowest register-tile grids (`6198321e`) | Flash-Next HC down: −11..−20% for the projection, 0.4-1.4% of a cycle |
| | QSA indexer q and k in one dispatch (`4944799d`); sparse split on a 1D grid (`ee917162`); block-list mask words in registers (`694243cf`) | −13..−24%; −18.5% overall (M5 Max); −42..−55% |
| Tooling | `teacher_logits` + `logit_compare.py`: full-vocabulary logits after fixed prefixes, teacher-forced through the production verify; top-1, mean, p99 and max KL, and the NLL change (`bb5d2c30`) | Noise floor, 35B UD-Q4_K_M at 1 lane vs 4: top-1 1.0000, mean KL 7.6e-4, p99 1.15e-2. The gate is set at twice that |
| Server | Stray `</think>` dropped; looser tool-schema handling; local model preparation; SSD and persistent-cache quota follows free space; requests wait for memory with keepalives | — |

Apple9 (M3/M4) gets few of the kernel changes. The float-projection rewrite applies (same simdgroup tile). The K-split
16 tier is "unmeasured on Apple9". The column tile affects Apple9 only for IQ-format experts.

## 2. Where Kapitsa already is (no work)

1. **Verify-row budget.**
   - Splash's VerifyBudget:
     - Model: `logit(p) = slope·logit(conf) + intercept[pos]`.
     - Refit every 64 cycles on the last 4096 labels.
     - Rows priced by a least-squares line over the last ~30 cycles.
     - Rows picked on the GPU by `draft_verify_rows`.
   - Kapitsa's `KfBudget` (`src/model/kflash_budget.h`, in main via #187) is the same design:
     - Model: `sigmoid(α_d + β_d·logit(q))` per depth.
     - Cost per row count measured on the Mac.
     - Rows picked by `E(n) − rate·cost(n)`, a marginal-rate rule that beats Splash's ratio rule in principle.
     - Adds best-first trees, and greedy output stays plain decoding's.
   - Splash had to declare that greedy and seeded MoE outputs no longer repeat run to run (`507f6f3f`). Two reasons: the
     rows follow wall-time costs, and their MoE tiles make a row's result depend on which rows share its expert tile
     (`ed63ffb5`).
   - That makes a positioning line: **Kapitsa's greedy output is reproducible; Splash 1.4.0's is not on MoE models.**
2. **Host encode overlaps the GPU.** Kapitsa commits the next drafting half while the verify runs
   (`src/engine/engine.cpp` around line 2214). Splash still keeps one command in flight, and its host steps exist for
   streamed reads, not for this.
3. **Event handoffs.** Kapitsa's Neural Engine handoff uses a ready event (written by the GPU) and a done event
   (written by the ANE) (`src/ane/ffn_split.cpp:560-565`, `src/model/nemotron_ane.cpp:649-655`). That is the
   one-writer-per-event rule Splash adopted after a once-in-a-million GPU timeout (`e2c0af27`). The CPU writes into them
   only when retiring after a failure, and only upward.
4. **Stray `</think>`.** `OutputParser::text_push` already treats a late `</think>` as a tag and drops it.
5. **Lone expert rows.** Kapitsa's expert verify kernels already run per-row matvecs. PERF.md measured dedupe across
   rows as slower. Splash's column tile catches up to that on its MPP path.
6. **Speed on the M4 Pro (35B, 27B).** The last same-Mac runs were Splash 1.3.2 on 2026-10-17:
   - 35B: 130.7 / 132.7 tok/s at T = 0.6 / greedy.
   - 27B: 57.7 greedy / 55.2 at T = 0.6.

   Kapitsa's README now has 154 and 63. 1.4.0's Apple9-relevant decode changes for these two models are the MoE budget
   (their M5 Max figure: +2.1% at one lane in MLX 4-bit) and the router/alpha-beta rewrite (a fraction of a ms per
   cycle). Estimate: low single digits on the M4 Pro, so the lead should hold, but it has to be re-measured before we
   say so.

## 3. What to take, ranked

### 3.1 KFlash's budget for batched decode (high value, measurable on the M4 Pro)

- **Splash:** a GPU step after the selector gives every lane its anchor, then the rows with the largest survival
  probabilities across the whole batch. It keeps adding rows while expected tokens over the cycle's cost at that total
  still pays. Costs are fit per lane count, and constrained cycles are fit separately.
- **Kapitsa today:** the batched path keeps DFlash2's chain walk (`src/model/dflash2.h:196`: "encode_draft (the
  batched path) keeps the walk"). It uses a fixed table measured once on the M4 Pro (`Engine::batch_rows`,
  `src/engine/engine.cpp:805-844`):
  - 2 members: 4 rows each.
  - 3 members: 2 rows each, or taking turns on the 27B.
  - 4 members: 2 rows each.
  - 8 rows in all at most.

  The table ignores how confident each member's drafter is in that cycle, and on other Macs, M5 included, it is a
  guess.
- **What to do:** run KfBudget over the batch. Take the members' calibrated path probabilities (chains first, trees
  later), choose the total row count n by the same `E(n) − rate·cost(n)` rule, and hand rows out by marginal survival.
  The cost table becomes cost per total rows of a batched pass, which is measured anyway. Where the cost table says rows
  are cheap (bf16 or M5, where 16 rows cost 1.05x), let the batch go past 8 rows.
- **How to measure:** `bench/batch/concurrency.py streams` at 2/3/4 requests against the fixed table, with the 35B and
  the 27B on the M4 Pro. Interleave A B A B, and check the near-tie rule for batched tokens.
- **Expected (estimate):** a few percent at 2-4 members on the 35B. Splash's 35B gain over every row was +2..+9%, and
  our fixed table is already closer to optimal than every row. It also removes the per-Mac table and makes the 27B's 3
  members stop taking turns whenever a cycle's drafts are confident.

### 3.2 Flash-Next with the DFlash2 draft (high value, needs the M5 Max)

- **Splash:** `incoai/Qwen3.8-Flash-Next-DFlash2` (rev `a22a1744`, the same org as the 27B and 35B drafts we ship).
  - The draft reads the target's capture layers as the **mean of the 4 hyper-connection streams**, summed in fp32 and
    stored bf16 (`dflash_config.target_hidden_reduction = stream_mean`). Everything else is a standard DFlash2 draft.
  - Their results: essay 2.4 tokens/cycle (63.2 tok/s), code 3.1 (68.3 tok/s), on an M5 Pro with their defaults.
    Acceptance per proposal is about 0.31 over 26 greedy answers (`6198321e`).
- **Kapitsa today:** MTP at depth 2 (`Qwen4Exp::default_spec_depth`, `block_drafter_ok() == false`): 107.6 greedy /
  101.8 at T = 0.6, against 84.8 without speculation (+27%; M5 Max, `docs/benchmarks-m5-max.md`). MTP-2 can never give
  more than 3 tokens per cycle.
- **What to do:**
  - Load the draft through the existing DFlash2 path, with a stream-mean capture hook in `Qwen4Exp`'s layer loop.
  - Run it under KFlash with dynamic rows. Rows are expensive here: one lane's cycle routes about 43 experts a layer at 8
    rows and 17 at 2.
  - A/B against MTP-2 on the M5 Max (128 GB).
- **Caveat:** acceptance is low, so the gain over MTP-2 is not certain. Trees over DFlash2's lattice may help more than
  the chain does. Check the draft's license, as for the other two.

### 3.3 Flash-Next below 128 GB (product reach)

- **Splash:** Q2_0 (ISTA-DASLab GSQ-RCO GGUF; 2-bit codes with one half scale per 64) is resident on 64 GB Macs. With
  12 GB of experts in RAM and the rest streamed, it runs on a 36 GB M5 Max at 59 ms per one-lane cycle. Estimate: 40-50
  tok/s at 2.4-3 tokens per cycle.
- **Kapitsa today:** a 4-bit pack only, a 105.9 GB download and 70.5 GB peak, so 128 GB Macs only. The GGUF import
  takes `qwen35` and `qwen35moe` only (`src/gguf/import.cpp:277-279`) and has no type 42 (Q2_0). The user's own M4 Pro
  (48 GB) cannot run Flash-Next at all, and Splash 1.4.0 can, streaming.
- **Step 1 (smaller): a 2-bit variant resident on 64 GB Macs.** Either import `qwen4exp` GGUFs with Q2_0/IQ4_NL, or
  quantize a Nace 2-bit pack chosen by KL, as the 4-bit was. Kapitsa already has IQ2 kernels (Bolt, the 35B's `16gb`).
  This is partly underway, but not merged. Worktree `kapitsa-strata-flash-2fa090` holds uncommitted Q2_0 work from
  2026-10-19/20: `src/model/moe_q20.*`, `src/kernels/zq2moe.metal`, `tools/qwen4_exp_q20.py`, `tools/fn_q2_bench.cpp`.
  It also made a 38.3 GB Q2_0 conversion at `~/.cache/kapitsa/models/qwen3.8-flash-next-q20` on the M4 Pro.
- **Step 2 (bigger): streamed experts (§3.4) for 36-48 GB Macs.**

### 3.4 Streamed experts: Splash's design and measurements (if we do 3.3 step 2)

This is a capacity feature, not a speed feature. For the 35B on 16 GB, our 2-bit `16gb` GGUF (53 tok/s on the M1 Pro)
probably beats 4-bit streaming. Estimate: 100-200 MB of misses a token at about 5 GB/s is 20-40 ms. For reference,
Splash's 35B with 4 GB of experts on a 24 GB M6 runs a 43 ms one-lane cycle.

What they settled on, and why:
- **Slot table indirection:** experts are read through `slots[e]`, and a resident model uses an identity table, so there
  is one kernel path. The cost is one more dependent load per expert tile.
- **ARC replacement:** a long prompt's sweep passes through and leaves decode's working set alone.
- **Reads:** `F_NOCACHE` reads widened to whole 16 KiB pages. 460 KiB pieces from 4 threads ran at 9.3 vs 8.1 GB/s
  with less than half the CPU (M5 Pro). Our M4 Pro: 4.5 GB/s at 8 threads, 6.0 GB/s at 16, for 1.3 MB random reads.
- **One pool of 32 SSD readers:** random 4 KiB reads reach 230K/s only at 32 in flight (168K at 16). Big expert pieces
  gain little past 4 threads.
- **Read-ahead:** run the *next* layer's router on *this* layer's input and prefetch the experts it picks. A wrong
  prefetch may fail without failing anything.
- **Prefill groups:** up to 4 chunks of 2048 rows run layer by layer, so each layer's experts are read once. Passes of
  about 500 ms let other requests' decode steps interleave.
- **Host steps:** the loads run inside the command, between routing and the expert passes. A GPU spin keeps clocks up
  through them. It relies on a memory scope missing from Metal's public headers, built as a separate metallib; see §4.
- **The verify budget prices rows by wall time, misses included:** on reasoning text a row costs 8 ms, on prose 3 ms.

### 3.5 Take the residency set off the command queue (cheap; tail latency under memory pressure)

- **Splash:** a command buffer waits for its queue's residency sets before it starts. With memory nearly full, 1-2% of
  command buffers waited 100-170 ms. Now every buffer stays in one backend set that a heartbeat keeps wired, and the
  queue holds only the buffers kernels reach through a GPU address (`c608e23f`).
- **Kapitsa:** one set holding every buffer is added to the queue (`src/metal/gpu.mm:403-416`). Kapitsa binds every
  buffer per dispatch (no `gpuAddress` anywhere in `src/`), so it could drop `addResidencySet` on the queue and keep the
  set wired with `requestResidency`.
- **How to measure:** p99 decode-cycle time and cached TTFT near the memory limit, for example Bolt on a 24-32 GB
  budget or the 35B with `--mem-budget` tight. Also watch for page-in on the first use after idle.

### 3.6 SSD prefix cache, optionally persistent (agent workloads)

- **Kapitsa:** prefix reuse lives in RAM slots and dies with the process. `kapitsa update` restarts servers.
- **Splash:** writes evicted KV pages and states to the SSD (on by default). With `--persistent-cache` they survive
  restarts. A coding agent with many sub-conversations then reloads a long prefix in about 0.1-0.3 s instead of
  prefilling it again.
- **Defaults worth copying:**
  - Quota: ¼ of free space beyond a 10 GiB floor, at most 32 GiB, shrinking as the disk fills. The user's 512 GB disk
    is nearly full.
  - An optional daily write budget, to spare SSD wear.
  - `F_NOCACHE` with a CRC per slot.
  - After two failed starts in a row, the next start begins with an empty cache.

### 3.7 Constrained decoding (feature gap)

- **Kapitsa:** no `json_schema`, strict tools or `tool_choice: required` (nothing in `src/` parses them).
- **Splash:** llguidance masks per cycle. Its 1.4.0 lesson for when we add constrained decoding: per-lane masks in the
  sampler, so constrained and free lanes share one batch. Splitting them by mode cost them 24-34% of aggregate time per
  token.

### 3.8 A logits gate with a measured noise floor (cheap tooling)

- **Splash:** `teacher_logits` dumps full-vocabulary logits after fixed prefixes, teacher-forced through the production
  verify. `logit_compare.py` gates on top-1, mean KL and p99 KL, at 2x the measured noise floor of the same prefixes
  scored 1 vs 4 at a time.
- **Kapitsa:** AGENTS.md asks for "differences only at near-ties". A dump format shared across engines would make that
  a number we can publish, and would also compare us against llama.cpp/MLX references.

### 3.9 Kernel checks (only if our profiles show the same waste)

- **Small float projections** (35B router, alpha/beta; Flash-Next router and indexer). Splash's ran at 148 GB/s on 2.6
  MB. 16-byte loads per lane and an 8-way K split cut 31-45%. Check our µs per dispatch with `NI_PROFILE=1`.
- **Two narrow projections of one input in one dispatch:** Splash's QSA indexer q|k saves 12 dispatches a step.
- **A K split of 16 when 8 leaves fewer than 3 threadgroups per core:** narrow-N, long-K shapes such as Flash-Next's HC
  down projection (320×10240).
- **Flash-Next sparse attention** (`Qwen4Exp::enc_attn_sparse`): a 1D grid over (split, head) instead of 2D, and
  block-list mask words in registers.
- **Flash-Next prefill:** ours is 860/850 tok/s on the M5 Max, against oMLX's 1,247/1,316 on the same Mac. Splash
  claims 1,312 at 32K on a 20-core M5 Pro (Q2_0). This release's diff doesn't explain the gap, but it makes it more
  visible. Profile where our Flash-Next prefill goes: the CPU n-gram gather, the QSA masks, the HC mixes.

### 3.10 YaRN past the native window (low priority)

If we add `--context` beyond 256K:
- Make it opt-in. Static YaRN changes outputs at every length.
- Scale only the target's attention RoPE. The draft keeps its own, which keeps acceptance.
- Shrink prefill rows past the native window so a command stays under the watchdog. Splash uses 512 rows at 1M.

A 1M prefill of the 35B takes about 45 min even on an M5 Max.

## 4. What not to take

- **Giving up greedy repeatability on MoE.** Our exactness rule, plus KFlash's budget, already gets the same kind of
  gain exactly.
- **The GPU keep-awake spin.** It relies on a memory scope missing from Metal's public headers, and Splash builds it into
  a separate metallib because "a Metal compiler that refuses it only warns". We compile from source at run time. Our
  only mid-command CPU wait is Flash-Next's n-gram gather (completion handler + `encodeWaitForEvent`,
  `src/model/qwen4_exp_spec.cpp:245-272`), which layer 0 already overlaps. Revisit only if a trace shows a GPU wake-up
  gap there.
- **MPP / Apple10 tile choices** (the 16-row staged tile for one lane, staged vs register tile rules). They matter only
  if we adopt MPP for decode on the M5.
- **1M context as a default.**

## 5. Benchmarks and positioning

- Splash's README now claims Flash-Next numbers against "Strata", and our M5 Max Flash-Next table has no Splash. Splash
  lists no MLX source for Flash-Next, so a same-weights comparison needs a GGUF we can import (3.3, step 1).
- Re-measure Splash 1.4.0 on the M4 Pro for the 35B and 27B (§2.6). Its MoE budget is on by default, so for exact
  comparisons with `--verify-rows all` and for our "fastest setup" rule with `auto`, record which one was used.
- Docs line worth adding when we publish batched or MoE numbers: greedy output on Kapitsa repeats run to run; on Splash
  1.4.0 MoE targets it does not (their DEVELOPMENT.md, Sampling and Drafts).

## 6. Not verified

- None of Splash's numbers were reproduced here. They come from their commits, mostly on M5-class Macs.
- The M4 Pro effect of 1.4.0 (§2.6), the batched-budget gain (3.1) and the residency stall (3.5) are estimates or
  hypotheses until measured under the GPU lock.
- The license of `incoai/Qwen3.8-Flash-Next-DFlash2` was not checked.
