Size override: parts 2 and 3 of the output check in one PR (1,064 counted lines, 64 over the cap): part 1 (#208) is merged, and #209 (part 2) is folded in here so both land in one CI round. Override approved by the maintainer.

Parts 2 and 3 of the output check; part 1 is #208 (merged). Two commits: `logit_dump` / `logit_floor.py`, then the measured floors and docs.

---

### Part 2: tools/logit_dump and tools/logit_floor.py

## What changed

- **`tools/logit_dump.cpp`** writes full-vocabulary logits for 8 windows cut from `bench/corpus.txt`: prompts of 512 to
  3,200 tokens, 24 rows each, 192 in all.
  - The rows are fed the text's own next tokens through three production paths: plain decoding, the 8-row verify (on
    Qwen, the KFlash tree verify as DFlash2 runs it), and 4 requests batched together.
  - It also writes each window's last prompt row from the prefill.
  - The file layout is Splash's (`sequences.txt`, `records.txt`, `logits.f32`), so another engine's dump of the same
    prefixes compares directly.
- **`tools/logit_floor.py`** measures a model's noise floor from such dumps: the same build twice, alone vs batched,
  prefill chunk sizes, and the Neural Engine split vs the GPU.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1; Qwen3.6-35B-A3B, Qwen3.8-27B and Qwen3.5-4B
4-bit, and Dory; engine 0553f55; each dump under the GPU lock.

- **Repeatability:** the same build run twice is bitwise identical on every path of all four models.
- **Order independence:** a path dumped alone matches the same path dumped after another one, bitwise (4B, 35B).
- **Over time:** `--from` re-dumped the 4B bitwise an hour later.

## Not verified

- The Nemotron-H paths of `logit_dump`, which were never run: there's no full Nemotron-H checkpoint on this Mac.
- Qwen3.8-Flash-Next, which the dump refuses because it decodes with MTP.
- No Metal kernels were changed.

---

### Part 3: floors, docs and the review rule

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
