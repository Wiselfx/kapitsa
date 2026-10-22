#!/bin/zsh
# Lock hold 3: 35B plain decode A/B without memory pressure (base / NI_RESIDENCY=queue / default wired), interleaved.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
echo "==== hold 3 start $(date +%T)"
zsh $SP/hold_decode.sh $SP/r3 /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $REPO/bench/prompts/code_35b.txt 2048 0 "A Q W W Q A"
echo "==== hold 3 end $(date +%T)"
