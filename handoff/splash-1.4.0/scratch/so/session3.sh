#!/bin/bash
# Session 3 (one GPU lock hold): (a) the unconstrained A/B, origin/main against this branch, interleaved; (b) structured
# output: masked KFlash cycles against masked plain steps (NI_GRAMMAR_SPEC=0), interleaved; (c) requests at once with
# NI_BATCH=1 (two under a schema, two free)
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash -c "
  bash $S/ab_inner.sh ab-free free 2 base:$S/build-base/kapitsa:- branch:$W/build/kapitsa:-
  bash $S/ab_inner.sh ab-schema schema 2 spec:$W/build/kapitsa:- plain:$W/build/kapitsa:NI_GRAMMAR_SPEC=0
  bash $S/batch_inner.sh batch3
  date"
