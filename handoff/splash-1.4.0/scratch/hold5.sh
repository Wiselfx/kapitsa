#!/bin/zsh
# Lock hold 5: 35B plain decode A/B under moderate memory pressure: a memhog (at most 6 GB, stops at the warning level
# or +128 MB swap) starts 3 s after the model has loaded and runs through the decode.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
echo "==== hold 5 start $(date +%T)"
zsh $SP/hold_decode.sh $SP/r5 /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $REPO/bench/prompts/code_35b.txt 3000 1 "Q W W Q A"
echo "==== hold 5 end $(date +%T)"
