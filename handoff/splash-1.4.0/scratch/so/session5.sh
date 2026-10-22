#!/bin/bash
# Session 5 (one GPU lock hold, the final binary): the unconstrained A/B against origin/main and structured output's
# masked KFlash cycles against masked plain steps, each in M B B M order; then requests at once with NI_BATCH=1
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
F=$S/build-final/kapitsa
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash -c "
  bash $S/ab_mbbm.sh s5-free free 2 base:$S/build-base/kapitsa:- branch:$F:-
  bash $S/ab_mbbm.sh s5-schema schema 2 spec:$F:- plain:$F:NI_GRAMMAR_SPEC=0
  KBIN=$F bash $S/batch_inner.sh s5-batch
  date"
