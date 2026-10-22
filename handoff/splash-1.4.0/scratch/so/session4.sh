#!/bin/bash
# Session 4 (one GPU lock hold, the final binary): the full validity run on the 4B (KFlash trees), quick runs with DFlash
# chains and with MTP (--dflash none), and a quick run on Dory 4B (no drafter: masked plain steps)
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
M=$HOME/.cache/kapitsa/models
export KBIN=$S/build-final/kapitsa
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash -c "
  python3 /Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100/bench/harness/gpu_hogs.py 2>&1 | tail -3
  bash $S/e2e_model.sh s4-4b $M/qwen3.5-4b-4bit - -
  bash $S/e2e_model.sh s4-chain $M/qwen3.5-4b-4bit NI_DFLASH_TREE=0 - --quick
  bash $S/e2e_model.sh s4-mtp $M/qwen3.5-4b-4bit - '--dflash none' --quick
  bash $S/e2e_model.sh s4-dory $M/nemotron-labs-dory1.0-4b-nvfp4 - - --quick"
