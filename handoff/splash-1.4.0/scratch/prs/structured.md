Size override: one feature in three reviewable commits. The grammar engine (`src/engine/grammar.cpp`, about 2,000 lines)
is a single module with its tests; split across PRs, its first part would be dead code in main. Override approved by
the maintainer for this PR.

## What changed

**1/3. A grammar engine with token masks** (`src/engine/grammar.{h,cpp}`, `tools/grammar_test.cpp`)
- It covers a JSON Schema subset and Qwen's XML tool-call format.
- Masks come from a byte trie over Qwen's 248K-token vocabulary: 6.0 µs per mask on average, p99 39 µs, and equal to
  brute force at 5,471 states.

**2/3. Sampling under masks in the engine**
- Masks apply to the first token, to plain steps, to DFlash2/KFlash verify rows (every draft under its own mask), and to
  batched passes (`logit_mask_rows` kernel, `src/model/logit_mask.cpp`, `engine.cpp`).
- Each mask is computed on the CPU while the GPU runs the layers before the sampler, so the GPU never waits.
- Other drafters (MTP, LiLiCorr, DSpark, prompt lookup) write the constrained part one token per pass and take turns
  rather than batching.

**3/3. The APIs**
- Chat Completions: `response_format` (`json_object`, `json_schema` with `strict`).
- Responses: `text.format`.
- Messages: `output_format`.
- `tool_choice` required / named / any / tool, and strict tools in all three.
- Reasoning stays free until `</think>`.
- Also: `tests/test_structured.py`, `docs/usage.md`, and the CI step.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1, Qwen3.5-4B 4-bit, 1 slot. Base: main
4e0154f. Builds in M B B M order, 2 rounds per cell, rates from the server log. "On" = KFlash; "off" =
`--dflash none --spec 0`.

**Requests without a format, main vs this PR (tok/s).** Every reply has the same tokens on both builds.

| | main | this PR |
| --- | --- | --- |
| Decode, on | 246.6 / 246.2 | 245.9 / 246.5 |
| Decode, off | 89.0 / 88.8 | 88.8 / 88.7 |
| Prefill ~4K, on / off | 1987, 1984 / 1983, 1982 | 1980, 1982 / 1986, 1985 |
| Prefill ~16K, on / off | 1580, 1525 / 1579, 1579 | 1578, 1580 / 1580, 1578 |

**The same tokens with and without a mask:**
- JSON schema, 4 × 512 tokens: 168.6 / 167.7 constrained vs 167.2 / 166.7 unconstrained with KFlash; 89.7 / 90.0 vs
  90.0 / 90.3 without speculation.
- A required tool call: 364.2 / 368.7 vs 364.1 / 364.0 with KFlash.

**Speculation under masks:** 177.6 / 177.0 tok/s with KFlash vs 88.5 / 87.6 with one token per pass, with identical
greedy replies.

## Checks

- **End to end, 4B with KFlash:** 54 of 54 replies schema-valid (greedy and two seeds), and forced tool calls valid
  against their tools' schemas. MTP and Dory: 24 of 24.
- **Grammar engine:** `grammar_test` passes 308 of 308, also under ASan and UBSan.
- **Kernel:** 33 of 33 under `MTL_SHADER_VALIDATION=1`.
- **CPU suite:** `api_test` (102 checks); `test_server.py --cli`, `test_sidecar.py`, `test_update.py`,
  `test_community.py`, `test_reasoning.py`, `test_pr_size.py` and `test_structured.py --cli` pass.
- **Not counted as a regression:** `test_bench.py` fails exactly as on main (frozen `kflash-merge.json`).

## Not verified

- The 27B, the 35B, and the Nemotron-H paths.
- Batched throughput with constrained members. Only their validity, and that they batch, were checked.
- Masked speculation for MTP, LiLiCorr, DSpark, prompt lookup and Dory's drafter: not built.
- Schema features not enforced: string `pattern` and `format`, number bounds, `uniqueItems`.

Design point from Splash 1.4.0 (Apache-2.0): constrained and free requests share batched passes. No code was taken;
Splash uses llguidance, and this engine is new C++ with no dependency.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
