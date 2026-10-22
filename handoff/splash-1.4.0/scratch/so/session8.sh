#!/bin/bash
# Session 8: one GPU lock hold of about 5 minutes. The branch (build/kapitsa), requests whose constrained reply is the
# unconstrained reply's tokens (json_object; tool_choice required on a call the model makes anyway), speculation on and
# off, M B B M: the masks' own cost per token.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
K=$W/build/kapitsa
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  bash "$S/perf_ab.sh" s8-same same 2 "on:$K:-:-" "off:$K:-:--dflash,none,--spec,0"
echo "hold released at $(date)"
