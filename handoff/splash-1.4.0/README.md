# Splash 1.4.0 follow-ups: handoff (2026-10-21/22)

This is a handoff branch, not meant for main. It holds the work of one session that analyzed Splash 1.4.0 (incoai,
released 2026-10-20) and turned what it found into Kapitsa changes. Everything that was finished is already merged.
This folder adds:
- the analysis;
- the one open task, a Flash-Next three-way benchmark;
- the scripts and logs behind every number;
- the open issues found along the way.

Every measurement below is from Yerbol's MacBook Pro 16-inch 2024: M4 Pro (20-core GPU), 48 GB, macOS 27.0.1.

## Merged into main

| PR | Change | Measured effect |
| --- | --- | --- |
| [#206](https://github.com/nace-ai/kapitsa/pull/206) | Residency set off the command queue, kept wired by a 500 ms heartbeat; let go under memory pressure | 35B, first token after a 10 s pause: 432–535 → 46–118 ms. Decode unchanged |
| [#207](https://github.com/nace-ai/kapitsa/pull/207) | KFlash budget picks each request's verify rows in batched decode | 3 requests at once: 35B +16–20%, 27B +15–23%. 2 and 4 at once: within noise. A request alone: unchanged |
| [#208](https://github.com/nace-ai/kapitsa/pull/208), [#210](https://github.com/nace-ai/kapitsa/pull/210) | Output check: `tools/logit_compare.py`, `tools/logit_dump`, `tools/logit_floor.py`; noise floors for the 35B, 27B, 4B and Dory | Gate at 2× the floor; it catches `NI_V8_CM=0` on the 4B |
| [#211](https://github.com/nace-ai/kapitsa/pull/211) | Structured output: JSON Schema, `tool_choice` required/named, strict tools; grammar masks in verify and batched passes | 4B: unconstrained decode and prefill unchanged; constrained replies as fast as unconstrained; 54 of 54 schema-valid |
| [#212](https://github.com/nace-ai/kapitsa/pull/212) | SSD prefix cache (`kapitsa serve --disk-cache`) | After a restart: 35B 16K first token 17–18 s → 0.16–0.18 s. 27B 47 s → 0.65–0.70 s |
| [#213](https://github.com/nace-ai/kapitsa/pull/213) | Flash-Next Q2_0 with routed experts streamed from the SSD (expert store, ARC, `F_NOCACHE` reads, read-ahead); DFlash2 plumbing | Flash-Next runs on a 48 GB Mac: about 20 tok/s decode (below) |

#209 was folded into #210.

## Open: Flash-Next on the M4 Pro, Kapitsa vs Splash 1.4.0 vs Strata

**Kapitsa, measured** (main `37eeb94`; the 2-bit pack; automatic expert store 13.6 GB; medians of 3; two values are two
rounds; on AC power). Details are in `scratch/fn3/RESULTS-kapitsa.txt`.

| | Kapitsa |
| --- | --- |
| Decode greedy, MTP depth 2 / plain | 20.5, 20.3 / 18.9, 17.7 tok/s |
| Decode T = 0.6, MTP depth 2 / plain | 20.9, 20.6 / 19.0, 18.8 tok/s |
| Prefill 4K / 16K, 2048-row chunks | 257 / 231 tok/s |
| Prefill 4K / 16K, `--prefill-chunk 4096` | 328–365 / 406–420 tok/s |
| Cached first token, 4K / 16K | 0.92 / 0.57 s |
| Peak memory | about 19 GB |

Caveat: the pack's n-gram table was sparse. It held only the benchmark prompts' rows, and missing rows read as zero.

**Splash and Strata: not run yet.** The disk on Yerbol's Mac ran out. What they need:
1. **The shared model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, `Q2_0/…-00001-of-00002.gguf` (37.62 GB) and
   `…-00002-of-00002.gguf` (28.80 GB, the n-gram table), revision `ed59f920`. `scratch/fn3/fetch.py` downloads them
   resumably. 20.4 of 61.9 GiB is already in `~/.cache/flash-next-gguf` on Yerbol's Mac. Plan on about 72 GiB free in
   all, and about 90 minutes at the ~13 MB/s Hugging Face gave here.
2. **Splash 1.4.0:** the release tarball, plus its draft `incoai/Qwen3.8-Flash-Next-DFlash2` at `a22a1744` (1.21 GB,
   Apache-2.0).
   - Serve it with `--max-cache-disk 0`.
   - `scratch/fn3/run_splash.sh` and `bench/harness/configs/splash.json` (for the 35B and 27B) show how.
3. **Strata** (github.com/Niko1221/Strata, commit `011c9e60`):
   - Build the Metal binary by hand: `setup.sh` would download a model.
   - Its MTP layer needs about 5 GB of BF16 tensors from `Qwen/Qwen3.8-Flash-Next`, so on 48 GB run setup's own config
     (`--spec 2 --suffix-draft 0`), plus suffix-draft and 4096-row prefill variants.
   - Configs: `scratch/fn3/strata-q2_0*.json`. The build recipe is in Octarin memory: search "Strata".
4. **Run order:**
   1. `scratch/fn3/guard.sh` (stops everything below 3 GiB free).
   2. `files.sh draft`.
   3. `run_mix.sh s-dec t-dec s-pf-a t-pf-a s-pf-b t-pf-b t-dec-sfx t-pf4096-a t-pf4096-b`.

   Use the same prompts, nonces and seeds as Kapitsa's runs: `hbench.py` replays them.
5. **To re-run Kapitsa:** its 2-bit pack was deleted for room. Rebuild it with `tools/qwen4_exp_q20.py`, which uses
   ranged reads of the same GGUF (about 41 GB over the network, no source shard on disk).

## Open issues found along the way

1. **Flash-Next, short prompts:** any prompt of 64 tokens or more reads every expert of every layer (24–31 GB, 6–8 s to
   the first token). Load only the routed experts for short chunks, and run prefill in groups of chunks layer by layer,
   as Splash does.
2. **Flash-Next, store size:** expert stores above about 40% of RAM made every command buffer 0.5–3.5 s slower on the
   48 GB Mac (a driver stall, inferred). The automatic size is capped at 40%; the cause is unknown.
3. **Flash-Next, misses:** each expert miss cost 1–1.5 ms of wall time, but its read took about 0.1 ms. The suspect is
   the GPU waking up after waiting. Splash's host steps keep the GPU spinning through a wait (see the analysis).
4. **Metal debug layer:** `MTL_DEBUG_LAYER=1` aborts in prefill because `attn_prefill11g` asks for 40 KB of
   threadgroup memory against a 32 KB limit. This also happens on main.
5. **Batched budget:** one exactness run crawled at about 5 s per batched cycle. The system log showed no GPU fault, and
   it did not reproduce in two re-runs.
6. **SSD cache:** writing only the deepest restore point would about halve the bytes per turn (240 MB on the 35B,
   430 MB on the 27B). A restore's first GPU work starts 40–95 ms later than from RAM.
7. **Structured output:** masked speculation for MTP, LiLiCorr, DSpark and prompt lookup is not built. Schema
   `pattern`, `format` and number bounds are not enforced.
8. **Residency:** real memory-pressure warnings were never reached here; that path is tested by simulation only
   (`NI_RESIDENCY_PRESSURE_TEST`).
9. **From Splash 1.4.0, not taken yet:** YaRN past 256K (opt-in), KV on the SSD for sparse attention (Flash-Next at 1M
   tokens), a DFlash2 drafting loop for Flash-Next, and host steps that keep the GPU spinning. All are in
   `SPLASH_1.4.0_ANALYSIS.md`.

## What's here

- **`SPLASH_1.4.0_ANALYSIS.md`:** what Splash 1.4.0 shipped, what Kapitsa took, what not to take, with numbers.
- **`scratch/`:** the session's working files, as they were. Builds, model files, third-party docs and files over 2 MB
  are left out. Paths inside are absolute to Yerbol's Mac and the session's scratch folder, so adjust them.

| Folder | Contents |
| --- | --- |
| `fn3/` | The Flash-Next three-way kit and Kapitsa's results (`RESULTS-kapitsa.txt`) |
| `fnq/` | The first Flash-Next 48 GB runs: store sizes, the prefill profile |
| `runs/`, `replay/` | Batched-budget A/B logs and the replay estimate (`replay/batch_replay.py`) |
| `so/` | Structured-output A/B scripts and logs |
| `prs/` | PR descriptions and `verify.sh` (CI's build and CPU tests, run locally) |
| `kfeq/`, `tools/`, `py/` | Smaller checks |

- **Elsewhere:** design notes and gotchas the agents recorded are in Octarin memory. Search "Splash 1.4.0", "residency",
  "expert store", "SSD prefix cache", "structured output", "Strata".

## Rules learned on this Mac

- **GPU lock:** one lock for the whole Mac, `/tmp/kapitsa-gpu.lock` since #292. Hold it at most 10 minutes, with a hard
  deadline that also kills servers. `sleep`-based watchdogs don't count system sleep.
- **Wait loops:** never start background processes in a wait loop. One spawned about 7,500 processes and hit the
  8,000-process limit.
- **Freeing disk:** Kapitsa model folders can be APFS clones of the Nace app's `~/.nace/models`, so deleting them frees
  nothing.
