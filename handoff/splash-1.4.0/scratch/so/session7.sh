#!/bin/bash
# Session 7: two GPU lock holds of about 5 minutes, 150 s apart.
# Hold A: unconstrained prefill and decode, origin/main against the branch (build/kapitsa as committed), speculation
# on (KFlash) and off (--dflash none --spec 0), M B B M.
# Hold B: the branch, the same requests unconstrained and constrained (a JSON schema; tool_choice auto and required),
# speculation on and off, M B B M.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
B=$S/build-base/kapitsa
K=$W/build/kapitsa
OFF=--dflash,none,--spec,0
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  bash "$S/perf_ab.sh" s7-free free 2 "main-on:$B:-:-" "branch-on:$K:-:-" "main-off:$B:-:$OFF" "branch-off:$K:-:$OFF"
echo "hold A released at $(date)"
sleep 150
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  bash "$S/perf_ab.sh" s7-pairs pairs 2 "on:$K:-:-" "off:$K:-:$OFF"
echo "hold B released at $(date)"
