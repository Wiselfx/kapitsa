Size override: one feature in two reviewable commits, the store (CPU-tested on its own) and its use by the server. The
store alone has no caller. Override approved by the maintainer for this PR.

## What changed

**The store** (`prefix_cache_test`, 67 checks, in CI)
- KV rows in 512-row blocks shared across turns, and restore points: the recurrent state plus the drafter's ring or the
  MTP head's rows.
- Written as CRC-checked files with atomic renames and `F_NOCACHE` reads and writes, within a free-space limit.

**The server** (`kapitsa serve --disk-cache auto|0|SIZE`, default `auto`)
- **Where:** `~/.cache/kapitsa/prefix-cache`.
- **Size:** `auto` takes the smaller of 32 GiB and a quarter of the free space beyond 10 GiB, and gives up the least
  recently used conversations as the disk fills.
- **Writes:** in the background after 10 s without requests, when a conversation the SSD lacks is about to be replaced,
  and at stop (up to 8 s). Never between decode cycles.
- **Reads:** only when no RAM slot holds a deeper prefix, on up to 4 reader threads.
- **Models:** Qwen 4B, 27B and 35B (DFlash2 or MTP) and Dory. Image conversations are cached too.
- **Without it:** Nemotron-H and Qwen3.8-Flash-Next run without the cache and say so in one log line.
- **Benchmarks:** `kapitsa bench` runs with `--disk-cache 0`. The harness's Kapitsa configs keep the cache in a folder
  emptied before and after each run, as they already do for oMLX's SSD cache.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1, Qwen3.6-35B-A3B and Qwen3.8-27B 4-bit with
DFlash2. Base: main 0553f55.

**A follow-up after a server restart**, two runs each:

| Model, prompt | Cold (TTFT, tok/s) | From the SSD after restart (TTFT, effective tok/s) | From a RAM slot | SSD read |
| --- | --- | --- | --- | --- |
| 35B, 4,359 tokens | 3.834 / 3.812 s (1,137 / 1,143) | 0.119 / 0.116 s (36.6K / 37.6K), ~32× | 0.062 / 0.056 s | 205 MB in 54 ms |
| 35B, 16,322 tokens | 17.24 / 17.99 s (947 / 907) | 0.164 / 0.178 s (99.5K / 91.7K), ~100× | 0.086 / 0.075 s | 450 MB in 81–82 ms |
| 27B, 4,365 tokens | 12.17 / 12.18 s (359 / 358) | 0.408 / 0.396 s (10.7K / 11.0K), ~30× | 0.269 / 0.268 s | 483 MB in 107–128 ms |
| 27B, 16,325 tokens | 46.74 / 47.99 s (349 / 340) | 0.646 / 0.700 s (25.3K / 23.3K), ~70× | 0.320 / 0.329 s | 1,267 MB in 286–291 ms |

**A request alone, main vs this PR with the cache on** (A B A B, the same text in all four runs):

| Model | Prefill, 4,324 tokens | Decode, 512 tokens | Follow-up decode |
| --- | --- | --- | --- |
| 35B | 3771, 3773 / 3768, 3767 ms | 133.2, 133.3 / 135.2, 134.7 tok/s | 125.1, 127.0 / 122.3, 126.9 tok/s |
| 27B | 11952, 11926 / 11939, 11927 ms | 49.6, 49.5 / 50.2, 49.2 tok/s | 48.0, 47.9 / 48.5, 47.7 tok/s |

**Greedy replies** after a read-back match the RAM path token for token.

**Bytes written per turn** after an idle pause: about 240 MB (35B) and 430 MB (27B). The first save of a 16K
conversation is 566 MB (35B) or 1,466 MB (27B).

## Checks

- **CPU:** `api_test`, `prefix_cache_test`, `test_server.py --cli`, `test_sidecar.py`, `test_update.py`,
  `test_reasoning.py`, `test_pr_size.py`, prek.
- **GPU:** `tests/test_disk_cache.py` on Dory, on the 4B with its DFlash drafter and with its MTP head. It covers a
  restart, `kill -9` after a background write, three turns sharing blocks, a replaced conversation, and an image
  conversation. The full `tests/test_server.py` suite also ran against a 4B server with the cache on.

## Not verified

- Other Macs and network volumes.
- A disk that actually fills.
- The stop-save under `kapitsa claude` or launchd.
- The MTP head on the 27B and 35B.
- Dory's per-loop KV.
- A second server of the same model, which runs without the cache.
- The time to copy out a replaced conversation (up to 1 GiB).
- `test_bench.py` / `test_community.py`, which need jsonschema, not installed on this Mac.

**Separate finding.** After a few idle seconds, a request's first GPU work starts about 0.35 s late in every path,
cache on or off. The residency PR (`residency-off-queue`) addresses it; the timings above send a short request first so
the stall doesn't distort them.

The design follows Splash 1.4.0's SSD and persistent cache (Apache-2.0); no code was taken.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
