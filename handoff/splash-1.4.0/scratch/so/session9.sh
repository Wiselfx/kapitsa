#!/bin/bash
# Session 9: one GPU lock hold of about 3 minutes. The branch (build/kapitsa), the JSON prompts unconstrained and under
# a closed schema their unconstrained greedy reply follows (the same tokens), speculation on and off, M B B M.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
K=$W/build/kapitsa
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  bash "$S/perf_ab.sh" s9-derived derived 2 "on:$K:-:-" "off:$K:-:--dflash,none,--spec,0"
echo "hold released at $(date)"
