Part 3 of 3 of the output check. It is stacked on part 2; merge parts 1 and 2 first, then retarget this PR to main.

## What changed

- **Measured floors** for the 35B, 27B, 4B and Dory on the M4 Pro: four files in `tools/logit_floors/`.
- **Docs:** an "Output check" section in `CONTRIBUTING.md`. The AGENTS.md "Changing speed or numerics" paragraph points
  to it, and the reviewer rules are mirrored in `.cursor/BUGBOT.md`.
- **Two gates per model.** The Neural Engine split moves the dense models' prefill logits about 1000× more than their
  decode passes move. So when two dumps' prefill rows are bitwise equal, the decode paths are held to a decode-only
  floor; otherwise to the whole floor.
- **Near-tie bound:** the largest top-2 margin that the floor itself moved by half or more.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1, engine 0553f55, final runs 10:24–13:22.
KL over the 192 verify rows, mean / p99:

| Model | Decode passes (alone vs 4 together) | Chunks 512 vs 2048 | Neural Engine vs GPU | Gate | Decode gate |
| --- | --- | --- | --- | --- | --- |
| 35B-A3B | 2.6e-4 / 3.0e-3 | 2.0e-4 / 6.8e-3 | no split | 5.1e-4 / 1.4e-2, near-tie 0.060 | 5.1e-4 / 6.0e-3, 0.060 |
| 27B | 2.0e-7 / 3.3e-6 | 6.6e-5 / 1.5e-3 | 1.9e-4 / 7.0e-3 | 3.8e-4 / 1.4e-2, 0.40 | 4.1e-7 / 6.6e-6, 0 |
| 4B | 5.1e-7 / 5.3e-6 | 2.1e-4 / 3.7e-3 | 9.3e-4 / 6.6e-3, 1 flip | 1.9e-3 / 1.3e-2, 1.52 | 1.0e-6 / 1.1e-5, 0 |
| Dory | no batching | 1.4e-3 / 1.6e-2, 2 flips | no split | 2.8e-3 / 3.1e-2, 0.99 | bitwise only |

A near-tie bound of 0 means no row came close enough to flip, so any flip fails.

**Control runs on the 4B,** each using an existing runtime switch:

| Switch | What it changes | Result |
| --- | --- | --- |
| `NI_V8_GUSPLIT=0` | the verify's MLP gate/up as one fused kernel | passes the decode gate |
| `NI_V8_CM=0` | GEMM codes encoded as 1024 + q | fails the decode gate (KL 3.3e-6; one row 5.4e-4) |
| `NI_REQUANT=8` | the same weights re-encoded at 8 bits | passes the whole gate |

## Not verified

- Floors on any other Mac: these are this M4 Pro's.
- Nemotron-H floors: no full checkpoint on this Mac.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
