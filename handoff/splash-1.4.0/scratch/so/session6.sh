#!/bin/bash
# Session 6 (one GPU lock hold, build/kapitsa from the tree as committed): the full validity run on the 4B (KFlash
# trees) and a quick one with MTP (--dflash none), whose reasoning check runs into max_tokens
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
M=$HOME/.cache/kapitsa/models
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash -c "
  python3 /Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100/bench/harness/gpu_hogs.py 2>&1 | tail -3
  memory_pressure | tail -1
  uptime
  bash $S/e2e_model.sh s6-4b $M/qwen3.5-4b-4bit - -
  bash $S/e2e_model.sh s6-mtp $M/qwen3.5-4b-4bit - '--dflash none' --quick"
