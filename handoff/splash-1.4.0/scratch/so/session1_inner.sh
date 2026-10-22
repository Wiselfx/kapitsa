#!/bin/bash
# Under the GPU lock: (1) the mask kernel at small sizes with Metal validation; only if it passes, (2) the 4B server and
# the structured-output end-to-end test (quick), the server stopped at the end.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
date
python3 "$W/bench/harness/gpu_hogs.py" 2>&1 | tail -3
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 "$W/build/logit_mask_check" > "$S/mask_check.log" 2>&1
rc=$?
echo "mask check exit=$rc"
grep -E "^FAIL|passed" "$S/mask_check.log"
grep -iE "validation|error" "$S/mask_check.log" | head -5
if [ $rc -ne 0 ]; then exit 1; fi
bash "$S/e2e_inner.sh" s1 --quick
date
