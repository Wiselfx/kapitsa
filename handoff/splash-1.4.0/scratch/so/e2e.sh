#!/bin/bash
# Takes the GPU lock (waits for it), then runs e2e_inner.sh TAG [test args...]
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
python3 /Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100/bench/harness/gpu_hogs.py 2>&1 | tail -4
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash "$S/e2e_inner.sh" "$@"
