# Splash expert-memory headroom: A−14 GiB or A−10 GiB? (issue #753, protocol only)

This folder is the protocol and the tools. **It holds no measurements yet.** Results go in `legs/` and `summary.json` / `table.md`
in a later commit of the PR that carries them.

## Question

`src/splash/plan.cpp` gives Splash `--expert-memory = A − 14 GiB` (kHeadroom 14, floor 8 GiB, only when A is 22–38 GiB), where A is
free + inactive + speculative + purgeable memory at launch. Issue #753 proposes `clamp(floor(A − 10), 8, 28)`. Does the 4 GiB
bigger expert cache make a turn-2 or cold-prefill request faster, without eating the margin that keeps the Mac out of swap?
The issue's gain estimates (0.05–0.1 s on turn 2, about 0.3 s floor) come from one probe at 16 GiB and are hypotheses here.

## Arms

| arm | `--expert-memory` | at A = 32 GiB |
|---|---|---|
| A14 (today's default) | `floor(A) − 14` GiB | 18 GiB |
| A10 (the proposal) | `clamp(floor(A) − 10, 8, 28)` GiB | 22 GiB |

Both arms are plain Splash 1.4.0 (`python -m server.server`, as the engines bench's `S` arm) with `--expert-memory N --idle-release off`
and nothing else changed. Kapitsa's plan only computes this one flag, so plain Splash with the same flag is the same engine
without a second variable (Kapitsa's own wrapper was within noise of plain Splash in the engines bench, which the PR should
re-check with one `kapitsa serve` pair if it adopts A10). A is read once, at the first leg of each pair, and both legs of the pair use
that pair's numbers, so a pair differs by the headroom and nothing else. The second leg of a pair does not start if A fell more
than 1.5 GiB below the pair's A.

## Procedure

- Four pairs, eight legs, order `A14 A10 | A10 A14 | A14 A10 | A10 A14`; one server start per leg, three repetitions each
  (n = 4 legs and 12 requests per cell and arm). About 2 min a leg once running.
- A leg starts only when A ≥ 31 GiB and no other model server runs (the harness refuses if one does), under
  `lockf -k -t 120 /tmp/kapitsa-gpu.lock`. A at launch is recorded in every leg's JSON.
- Requests are the engines bench's, imported unchanged from `../2026-10-28-m5-pro-flash-next-engines/scripts/flash_next_bench.py`:
  decode at T=0 and T=0.6 (256 tokens), cold prefill at 1K / 4K / 16K (16 tokens out, no cache), and the two-turn chat on the 4K
  context (a `read_file` tool call, then turn 2). One chat on both arms; this is *not* `tools/splash_ab.py --turn2`'s chat, and the
  bench's turn-2 TTFT (0.56 s) is not comparable with #743's 0.20–0.26 s.
- Per request: TTFT, decode tok/s, and the numeric deltas of Splash's `GET /status` `streamed_experts` block before and after
  (`reads` in each leg's JSON). The block's field names are not documented, so every numeric leaf is diffed and the first and last
  whole `/status` are stored.
- Per leg: min free + reclaimable (sampled every 0.5 s here, every 2 s by the harness), swap growth, peak pressure level, peak
  footprint, any Splash refusal. The guard stops a leg at pressure level 4, free < 3 % or swap > 16 GB.

## Commands

```bash
cd bench/results/2026-10-28-m5-pro-splash-headroom
python3 scripts/headroom_ab.py selftest      # CPU: arithmetic, label parser, whole leg against a fake server, verdict scenarios
python3 scripts/headroom_ab.py preflight     # inputs, A now, whether a model server runs; loads nothing
python3 scripts/headroom_ab.py plan          # the eight legs and their exact commands
# GPU, only on the orchestrator's GO naming this folder's frozen commit, as a detached job (AGENTS.md "Long jobs"):
python3 ~/.agents/skills/session-orchestration/scripts/detach.py run --name headroom-ab --cwd "$PWD" -- bash scripts/run_ab.sh
python3 scripts/headroom_ab.py summarize legs   # table.md, summary.json, the verdict
```

`run_ab.sh` skips legs that have a JSON, so it resumes. It stops on a guard stop (rc 3), a Splash refusal or exit (4), or an A10 leg
that broke a safety bound (5), and writes the summary.

## Decision rule (fixed before any run)

Computed by `headroom_ab.py summarize`; nothing else counts. Per cell, per pair: `gain = A14 leg median − A10 leg median`
(sign flipped for tok/s), so positive means A10 is better. *Noise* is the larger of the two arms' ranges of leg medians.

**ADOPT A10 (open a PR with this A/B, changing kHeadroom to the clamp)** only if all hold:

1. at least 3 complete pairs;
2. on a primary cell (turn-2 TTFT, cold-prefill 4K TTFT), A10 is better in at least 3 of 4 pairs (75 %), the median paired gain is above the noise, and it is at least 5 % of the A14 median;
3. no cell, primary or secondary (1K and 16K TTFT, turn-1 TTFT, decode tok/s at T=0 and T=0.6), is *worse* by the same test, mirrored;
4. every A10 leg stays at free + reclaimable ≥ 3 GiB, swap growth ≤ 1 GB, pressure below level 4, with no Splash refusal and no request error.

**KEEP 14 GiB (close #753)** otherwise: unsafe A10, a slowdown anywhere, or no gain beyond noise. **INCONCLUSIVE** with fewer than 3
complete pairs, then run more pairs or nothing changes. A default also needs the PR's own checks from AGENTS.md; this A/B is only
the interleaved measurement. A10 refusing to start, or Splash refusing at A10 where A14 starts, counts under 4.

## What this cannot say

- Anything about A < 31 GiB, where the issue's A−14 floor and refusal behaviour matter most; the 22–31 GiB window is untested.
- Whether a gain would hold with other apps open: the legs run on a quiet Mac, as the harness requires.
- Output equality. Both arms change only how many experts stay resident; if the PR adopts A10 it still carries a temperature-0 token comparison.
