#!/bin/bash
# Session 10: one GPU lock hold (hard deadline 9.5 min inside): the merge (build/kapitsa) against origin/main 92da1cd
# (build-main/kapitsa), 4B with NI_BATCH=1, the batch budget on and NI_BATCH_KF=0
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
M=$S/build-main/kapitsa
B=$W/build/kapitsa
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  bash "$S/gpu_check_inner.sh" s10 "main-kf:$M:-" "branch-kf:$B:-" "branch-kf0:$B:NI_BATCH_KF=0" "main-kf0:$M:NI_BATCH_KF=0"
echo "hold released at $(date)"
