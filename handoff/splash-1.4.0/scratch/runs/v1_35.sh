#!/bin/bash
# batch_check under Metal validation: the batch budget's row layouts (non-uniform rows per sequence) and the split pass
cd /private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/runs
B=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/build-b/batch_check
P=@/Users/flaminis/Desktop/kapitsa/.claude/worktrees/splash-1-4-0-analysis-3bbeaf/bench/prompts/p16000.txt
echo "### debug layer + shader validation, 2 sequences $(date +%H:%M:%S)"
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 NI_BATCH_SPLIT=1 $B /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $P 2 "7,1;1,7;5,3;6,2" 4 128 2>&1 | grep -vE "^\[kapitsa\] (NI_|GPU:|memory|Qwen)" | tail -6
echo "### shader validation, 2 sequences $(date +%H:%M:%S)"
MTL_SHADER_VALIDATION=1 NI_BATCH_SPLIT=1 $B /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $P 2 "7,1;1,7;5,3;6,2;4,4;2,6" 12 256 2>&1 | tail -4
echo "### shader validation, 3 sequences $(date +%H:%M:%S)"
MTL_SHADER_VALIDATION=1 NI_BATCH_SPLIT=1 $B /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $P 3 "3,3,2;6,1,1;1,1,6;2,3,3;4,2,2;1,2,5" 12 256 2>&1 | tail -4
echo "### shader validation, 4 sequences $(date +%H:%M:%S)"
MTL_SHADER_VALIDATION=1 NI_BATCH_SPLIT=1 $B /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit $P 4 "2,2,2,2;4,2,1,1;1,1,1,5;5,1,1,1;3,2,2,1;1,3,1,3" 12 256 2>&1 | tail -4
echo "### done $(date +%H:%M:%S)"
