#!/bin/bash
# Session 2 (GPU lock): the 4B server with the defaults and the structured-output e2e test (--quick)
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash -c "date; python3 /Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100/bench/harness/gpu_hogs.py 2>&1 | tail -3; bash $S/e2e_inner.sh s2 --quick; date"
