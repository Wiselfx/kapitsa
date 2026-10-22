#!/bin/zsh
# precheck.sh OUT: foreign GPU load (4 s), load average, memory (vm_stat, memory_pressure, swap, level), other
# engines' processes; appended to OUT
K=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/kap-main
{
  echo "== $(date '+%Y-%m-%d %H:%M:%S') precheck"
  echo "uptime: $(uptime)"
  echo "power: $(pmset -g batt | head -1) / powermode $(pmset -g | awk '/powermode/ {print $2}')"
  echo "gpu users (4 s):"; python3 $K/bench/harness/gpu_hogs.py 4 2>&1 | head -8
  vm_stat | awk '/Pages (free|active|inactive|speculative|wired down|purgeable|occupied by compressor)/'
  echo "memory_pressure: $(memory_pressure | tail -1)"
  echo "swap: $(sysctl -n vm.swapusage); memorystatus_level $(sysctl -n kern.memorystatus_level)"
  echo "engines:"; ps -axo pid,rss,etime,comm | awk '$4 ~ /kapitsa|splash|strata|llama|ollama|mlx|ni_run|nh_|lms|LM Studio/' | head -10
  echo "disk: $(df -g /System/Volumes/Data | tail -1 | awk '{print $4}') GB free"
} >> $1 2>&1
