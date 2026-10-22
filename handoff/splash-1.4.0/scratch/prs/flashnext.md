Draft: it runs on the real model, but only with a sparse n-gram table, and the DFlash2 part is plumbing only (see "Not
verified"). It will be split before it's ready for review.

Size override: four commits of one feature, Flash-Next on Macs below 128 GB. The first ports the 2-bit kernels and the
converter as one unit. Override approved by the maintainer for this PR.

## What changed

**1. Q2_0 routed experts for Qwen3.8-Flash-Next** (`src/kernels/zq2moe.metal`, `src/model/moe_q20.*`, the converter
`tools/qwen4_exp_q20.py`, `tools/fn_q2_bench.cpp`)
- Ported from the uncommitted work in worktree `kapitsa-strata-flash-2fa090`, with its kernels, dispatch and converter
  kept as they were.

**2. Routed experts streamed from the SSD**
- An arena of expert slots read through the existing slot-table kernels, so resident and streamed experts run the same
  kernels.
- A per-layer handshake between routing and the expert passes.
- ARC replacement.
- A pool of 16 SSD readers doing `F_NOCACHE` reads widened to whole 16 KiB pages.
- Next-layer read-ahead: the next layer's router run on this layer's input.
- `NI_EXPERT_STORE=<GB>|all|off` sets the store size.

**3. DFlash2 plumbing for Flash-Next**
- `target_hidden_reduction` checks, any default-rope theta, and a kernel that captures a layer's 4-stream mean.
- No drafting loop yet.

**4. The automatic store kept within 40% of RAM**
- Stores of 15–24 GB made every command buffer 0.5–3.5 s slower on the M4 Pro; at 12 GB it runs at full speed.
- Each request's store hits and GB read are logged with `NI_LOG=1`.
- `qwen4_exp_q20.py` gains `rows --stream`.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1. The Q2_0 pack converted from
`nace-ai/Qwen3.8-Flash-Next-MLX-4bit` with a sparse n-gram table. `kapitsa serve`, 32K context, 1 slot,
`--dflash none`, 12 GB store. The harness's prompts with fixed nonces and seed; medians of 3.

| | M4 Pro 48 GB, Q2_0, streamed |
| --- | ---: |
| Decode, MTP depth 2, greedy / T = 0.6 | 16.8 / 17.8 tok/s |
| Decode, plain, greedy / T = 0.6 | 16.1 / 16.4 tok/s |
| Prefill 4K / 16K, 2048-row chunks | 265–284 / 237 tok/s |
| Prefill 4K / 16K, `--prefill-chunk 4096` | 357 / 407 tok/s |
| First token of a follow-up, 4K / 16K | 0.97 / 0.74 s |
| Peak memory | 17.4–19.6 GB |

- **Store:** decode finds 88–92% of routed experts in it and reads 105–200 MB per token from the SSD.
- **Store size:**

  | Store | MTP cycle |
  | --- | ---: |
  | 4–12 GB | 187–247 ms |
  | 15 GB | 743 ms |
  | 18 GB | 2,968 ms |
  | 24 GB | 3,696 ms |

  A thread sample shows command-buffer submission blocked inside the driver; that cause is inferred.
- **Synthetic Flash-Next-shaped pack:** the logits are bit for bit the same whether the experts are resident, all in the
  arena, or 52% streamed, through prompt, decode, verify of 3 and 8 rows, and MTP.
- **Store CPU test:** passes.
- **Q2_0 kernels:** checked under `MTL_SHADER_VALIDATION=1` at small sizes; within 1e-4 of a double-precision
  reference.

## Not verified

- **The full n-gram table:** it would not fit on the disk. With the sparse table, missing rows read zero, so answers
  are the model's own only on covered tokens, and n-gram reads are slightly cheaper than with the full table.
- **The DFlash2 draft** (`incoai/Qwen3.8-Flash-Next-DFlash2`, Apache-2.0): never run, since the drafting loop isn't
  built.
- **Logits:** only greedy tokens were compared.
- **Short prompts:** prompts of 64 tokens or more read every expert of every layer (24–31 GB, 6–8 s to the first token).
  Not fixed.
- **The 40% cap:** not checked on other Macs or under other memory loads.
- **The M5 Max:** its resident 4-bit path was not run with this change.

The streaming design follows Splash 1.4.0's streamed experts (Apache-2.0); no code was taken.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
