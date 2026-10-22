Stacked on #208 (part 1). Merge #208 first, then retarget this PR to main.


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

🤖 Generated with [Claude Code](https://claude.com/claude-code)
