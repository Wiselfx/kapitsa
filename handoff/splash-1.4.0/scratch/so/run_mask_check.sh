#!/bin/bash
# The mask kernel's small-size check under Metal validation, holding the GPU lock.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
cd "$W" || exit 1
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" \
  "$W/build/logit_mask_check" > "$S/mask_check.log" 2>&1
echo "exit=$?"
tail -14 "$S/mask_check.log"
