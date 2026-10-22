#!/bin/bash
# build/ (the worktree's) with every target the tests use, at background QoS
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
taskpolicy -b cmake --build "$W/build" -j 4 --target kapitsa api_test bench_test update_test grammar_test structured_check \
  logit_mask_check > "$S/build-all.log" 2>&1
echo "exit=$?"
grep -E "error|warning:" "$S/build-all.log" | head -5
