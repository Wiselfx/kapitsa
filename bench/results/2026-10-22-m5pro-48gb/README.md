# Results, 2026-10-22: MacBook Pro M5 Pro, 48 GB (Qwen3.8-27B, 16-row drafting blocks)

Kapitsa's 27B decode on this Mac with the 16-row drafting block that [#266](https://github.com/nace-ai/kapitsa/pull/266)
made the default on a one-slot engine, measured side by side with mlx-serve 26.10.2's DFlash2 draft tree, written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/kapitsa.json configs/mlx-serve.json --only
q38-27b-kapitsa,q38-27b-mlxserve-dflash2 --skip agent`, the full protocol minus the agent session.

MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power in High Power mode. One config at a
time under the GPU lock, 07:36-08:09 local (UTC+5), two passes with the engines in opposite order: 07:41-08:01 Kapitsa
then mlx-serve, 08:02-08:07 mlx-serve then Kapitsa. Every run recorded 0.0% foreign GPU load; the one background GPU
user on this machine, Drata Agent (7-12% of the GPU in the 07:36 run and in the 2026-10-21 evening runs), was paused
before the first clean run and resumed after the last, so the numbers below are quiet-GPU numbers.

Kapitsa is the merge of #266, `d473e2f`; the binary reports `git_head` 6180e99, the branch head whose tree differs from
the merge only in README, docs, poster and stored results (the same tree-identity the 2026-10-21 set used). Main's later
[#270](https://github.com/nace-ai/kapitsa/pull/270) (routed experts on `tensor_ops`) is off by default and only touches
the 35B's MoE prefill, so the 27B path measured here is main's.

The runs, one row each:

| Result file | Engine | Build | Decode T = 0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | TTFT 4K (s) | Peak memory (GB) | Flags |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q38-27b-kapitsa-20261022-073603.json` | Kapitsa (DFlash2 drafter) | 6180e99 | 69.2 | 86.4 | 454 / 449 | 9.10 | 20.8 | contention: Drata Agent used 7.3% of the GPU; not used |
| `q38-27b-kapitsa-20261022-074612.json` | Kapitsa (DFlash2 drafter) | 6180e99 | 79.2 | 88.0 | 463 / 452 | 8.94 | 20.8 | none |
| `q38-27b-kapitsa-20261022-075459.json` | Kapitsa (DFlash2 drafter) | 6180e99 | 80.4 | 87.9 | 469 / 452 | 8.81 | 20.8 | none |
| `q38-27b-mlxserve-dflash2-20261022-075759.json` | mlx-serve 26.10.2 (DFlash2 draft tree, block 8) | 26.10.2 | 81.1 | 90.9 | 475 / 457 | 8.71 | 21.8 | none |
| `q38-27b-mlxserve-dflash2-20261022-080111.json` | mlx-serve 26.10.2 (DFlash2 draft tree, block 8) | 26.10.2 | 79.1 | 91.1 | 474 / 459 | 8.71 | 21.8 | none |
| `q38-27b-kapitsa-20261022-080422.json` | Kapitsa (DFlash2 drafter) | 6180e99 | 84.3 | 89.8 | 496 / 456 | 8.34 | 20.8 | none |

The pass order matters on this Mac: the engine that runs second in a pass reads 2-5% faster (GPU clocks are warm by
then), so neither single run is a fair comparison on its own. Pass by pass, same conditions:

| Pass | First | Second |
| --- | --- | --- |
| 07:55-08:01 | Kapitsa 80.4 | mlx-serve 81.1 |
| 08:02-08:07 | mlx-serve 79.1 | Kapitsa 84.3 |

Averaged over the two passes (each engine once in each position): Kapitsa 82.4 tok/s at T = 0.6, mlx-serve 80.1;
greedy 88.8 against 91.0; prefill 482 / 454 against 475 / 458. Greedy favours mlx-serve by about 2%, the gap the
2026-10-21 runs showed (77.4 against 89.4) has narrowed on both sides, and the sampled decode that the poster's decode
column reads now sits with Kapitsa by about 3%.

Before #266 the same build drafted 8-row blocks by default: the 2026-10-21 run of the 27B on this Mac
([`q38-27b-kapitsa-20261021-221847.json`](../2026-10-21-m5pro-48gb/results/q38-27b-kapitsa-20261021-221847.json),
74.1 at T = 0.6, 77.4 greedy, 463 / 451 prefill) is the before number, and its greedy decode was measured before the
narrowed tape growth of #266's last commit. The three clean runs here are the after number: 79.2, 80.4 and 84.3 at
T = 0.6 against 74.1, and 88.0-89.8 greedy against 77.4.

The README poster's M5 Pro 27B cells use these runs: Kapitsa 82 / 482 and mlx-serve 80 / 475, each the mean of its two
passes, as [docs/benchmarks.md](../../../docs/benchmarks.md#macbook-pro-m5-pro-48-gb-2026-10-19) explains.
