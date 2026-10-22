#!/bin/zsh
# The CPU tests from AGENTS.md on the worktree's build (no model, no GPU), with a temporary GPU lock file.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
cd $REPO || exit 1
touch $SP/testlock
export BENCH_GPU_LOCK=$SP/testlock
cmake --build build -j 4 --target kapitsa api_test bench_test update_test 2>&1 | grep -E "error|warning" | head -5
echo "== api_test: $(build/api_test 2>&1 | tail -1)"
echo "== test_server --cli: $(python3 tests/test_server.py --cli build/kapitsa 2>&1 | tail -1)"
echo "== test_sidecar: $(python3 tests/test_sidecar.py build/kapitsa 2>&1 | tail -1)"
echo "== test_bench: $(python3 tests/test_bench.py build 2>&1 | grep -E '^FAIL' | tail -1)"
echo "== test_update: $(python3 tests/test_update.py build 2>&1 | tail -1)"
echo "== test_reasoning: $(python3 tests/test_reasoning.py 2>&1 | tail -1)"
echo "== test_pr_size: $(python3 tests/test_pr_size.py 2>&1 | tail -1)"
rm -f $SP/testlock
echo "== done"
