# M5 Pro, 48 GB: the 27B after the Neural Engine split moved to the matrix units, 2026-10-22

[#318](https://github.com/nace-ai/kapitsa/pull/318) runs the GPU's share of the 27B's Neural Engine FFN split on the
M5's matrix units and gives the GPU 10 of the 34 units by default (24 stay on the Neural Engine; before, all 34 did).
This directory re-measures the poster's M5 Pro Qwen3.8-27B cells, Kapitsa's and mlx-serve's, after that merge, with
the poster's protocol: the harness, one config at a time under the GPU lock, prefill and decode from the same runs
(`compare_engines.py configs/kapitsa.json configs/mlx-serve.json --only <label> --skip tools,agent`).

- Mac: MacBook Pro M5 Pro 48 GB (20-core GPU), macOS 26.5.2 (25F84), AC power
- Kapitsa: Release `kapitsa serve` with the DFlash2 drafter (KFlash trees, 16-row block), `--ctx 32768 --slots 1`, on
  qwen3.8-27b-4bit-mtp. The first two runs are main `3f7f4c8` (#318's merge). The pair's two runs report `a4a32b7`:
  this branch on top of `cff1b1e` ([#328](https://github.com/nace-ai/kapitsa/pull/328)) and `daf6e95` (0.2.8). Neither
  changes the 27B's engine: #328 turns the verify-pass prefill off only below a weight scale of 3e9 and the 27B is 5.7e9,
  and 0.2.8 is a version bump.
- mlx-serve 26.10.2 with DFlash2 (draft tree, block 8, `--no-mtp`), its pack rebuilt with
  `bench/harness/tools/mlx_serve_pack.py qwen3.8-27b` (the same trunk and drafter files as Kapitsa's, cloned).
- Every Kapitsa server log shows `24/34 units ... from the default for this chip` and a cached Neural Engine program
  (ready in 0.1-0.2 s), so no run paid the first-start compile.

## The cells: an ABBA pair

mlx-serve, Kapitsa, Kapitsa, mlx-serve, each engine's two runs in opposite order (the second engine in a pass reads
2-5% faster, so the cell is the mean over both orders), as the [earlier pair on `16f2b6a`](../2026-10-22-m5pro-48gb/README.md)
was measured.

| Pass | Engine | Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Peak footprint (MB) | Flags |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | mlx-serve | `q38-27b-mlxserve-dflash2-20261022-202754` | 91.37 (83.25-93.95) | 91.04 | 463 / 455 | 21,402 | none, 0% foreign GPU |
| 1 | Kapitsa | `q38-27b-kapitsa-20261022-203123` | 73.03 (66.22-91.05) | 85.30 | 651 / 541 | 20,562 | contention in the 16K prefill only (WebKit, 2.2% mean, 6.1% max) |
| 2 | Kapitsa | `q38-27b-kapitsa-20261022-203441` | 78.81 (66.28-81.94) | 84.73 | 533 / 524 | 20,544 | memory: 0.67M / 0.64M / 0.12M pages swapped out in the decode, greedy and 4K prefill phases; 0% foreign GPU |
| 2 | mlx-serve | `q38-27b-mlxserve-dflash2-20261022-210433` | 79.68 (69.44-90.16) | 90.81 | 472 / 458 | 21,210 | none, 0% foreign GPU |
| | **Kapitsa mean** | | **75.92** | **85.02** | **592 / 533** | | |
| | **mlx-serve mean** | | **85.53** | **90.93** | **468 / 457** | | |

The second mlx-serve run waited 28 minutes for the GPU lock while other sessions' jobs held it. The swap on Kapitsa's
second run is other processes' memory use on this Mac during that run (the harness flags it; Kapitsa's footprint is
the same 20.5 GB as in its other runs).

**mlx-serve decodes faster: 13% at T = 0.6 (85.5 against 75.9) and 7% greedy (90.9 against 85.0).** Kapitsa reads the
prompt faster: 1.27x at 4K (592 against 468) and 1.17x at 16K. Neither engine's decode lead depends on which runs are
Whichever of Kapitsa's runs on this build are counted (the pair 75.9, the two flag-free runs below 79.2, all four 77.6),
mlx-serve's mean is 8-13% ahead on decode.

## Two earlier Kapitsa runs on `3f7f4c8`, flag-free

Before the mlx-serve pack was rebuilt, Kapitsa alone ran twice (`compare_engines.py configs/kapitsa.json --only
q38-27b-kapitsa --skip tools,agent`). Both runs are flag-free; the first waited 244 s for the GPU to go quiet.

| Run | Decode T=0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (MB) | Foreign GPU load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q38-27b-kapitsa-20261022-195053` | 76.62 (72.83-85.62) | 86.18 | 583 / 536 | 0.21 / 0.24 | 20,562 | 0.97% mean, 3.4% max |
| `q38-27b-kapitsa-20261022-200236` | 81.80 (77.13-91.36) | 84.31 | 653 / 546 | 0.23 / 0.26 | 20,564 | 0.24% mean, 2.8% max |
| **Mean** | **79.21** | **85.25** | **618 / 541** | | | |

[#332](https://github.com/nace-ai/kapitsa/pull/332) put these on the poster (618 / 79) against mlx-serve's stored
`16f2b6a` pair (464 / 78) and called the sampled decode a tie until mlx-serve was re-run in the same session. The pair
above is that re-run, and these runs read the same way as it: mlx-serve's pair mean decodes 8% faster than their 79.2.
The poster now takes the pair, so that both engines' cells come from one interleaved session.

## What changed on the poster

- **Kapitsa: 618 / 79 to 592 / 76** (prefill 4K / decode; the cell #332 set from the two flag-free runs above). Before
  #318 the cell was 459 / 77: the pair's prefill is 1.29x that at 4K and 1.19x at 16K (446 to 533), against the four-run
  means on `16f2b6a`; against that pair's only flag-free Kapitsa run (484 / 453), 1.22x / 1.18x, since
  three of the old runs carried the browser's GPU load. #318 reported 494 to 619 and 441 to 516 with its own serve A/B.
  Decode is unchanged within the spread (76.89 to 75.92 sampled, 83.08 to 85.02 greedy); #318 does not touch decode.
- **mlx-serve: 464 / 78 to 468 / 86.** Its decode reads 9% higher than the `16f2b6a` pair (78.16 to 85.53 sampled,
  86.23 to 90.93 greedy), whose runs were mostly contention-flagged; its prefill is unchanged (464 / 472 to 468 / 457).
- The M5 Pro's 27B row is mlx-serve's on decode, no longer a tie; the README's headline stays at 24 of 26 pairs.
- Against Splash 1.4.0's stored 27B cell (674 / 72, two contention-flagged runs on `4b89f4e`), Splash still reads the
  prompt faster: 1.14x at 4K and 1.16x at 16K against the pair (1.50x and 1.38x before #318, 1.09x and 1.14x against
  the two flag-free runs); Kapitsa decodes 5% faster at T 0.6 and 20% greedy.

The runs are in `results/`, the server logs in `logs/`.

## Not verified

- Splash and the other engines were not re-run; their cells are the stored runs.
- Not an interleaved A/B against the pre-#318 build in this session: the before numbers are the stored `16f2b6a` runs.
- Both of Kapitsa's runs in the pair carry a flag (above); the flag-free runs were not interleaved with mlx-serve.
- Tools and the agent session were skipped (`--skip tools,agent`), as for every poster cell.
