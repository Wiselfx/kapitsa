Part 1 of 3 of the output check. Parts 2 and 3 are stacked on this branch.

## What changed

- **`tools/logit_compare.py`** compares two full-vocabulary logit dumps, in fp64. It reports:
  - top-1 agreement, with each flip's top-2 margin;
  - KL mean, median, p99 and max;
  - the next-token NLL change;
  - rows that are bitwise equal.
- **Gate:** it exits 1 past the floor it is given. KL mean and p99 must be at most twice the floor's, flips at most twice
  its flips, and every flip a near-tie.
- **No hard dependency:** it uses numpy when installed (192 rows in 0.7 s) and otherwise the standard library (19 s,
  the same numbers).
- **Dump formats:** it reads the dump layout of part 2 (Splash's layout, so another engine's dump of the same prefixes
  compares directly) and the raw dumps `tools/nh_vcheck.py` read. So `nh_vcheck.py` is removed, and its references
  (`tools/nh_eval.cpp`, AGENTS.md) point to `logit_compare.py`.
- **Test:** `tests/test_logit_compare.py` on synthetic dumps, added to CI.

## Checks

- `tests/test_logit_compare.py`: 41 checks pass with numpy, and 40 with numpy hidden, as in CI.
- Also passing: `api_test`, `test_server.py --cli`, `test_sidecar.py`, `test_update.py`, `test_reasoning.py`,
  `test_pr_size.py`.

## Not verified

- The CPU path only: this part has no engine or GPU change.
- `test_community.py` / `test_bench.py`, which need jsonschema, not installed on this Mac.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
