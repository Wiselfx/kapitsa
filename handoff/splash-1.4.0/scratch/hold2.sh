#!/bin/zsh
# Lock hold 2: residency semantics with command buffers (residency_probe2 watch scripts, 8 GB), then the first command
# buffer after idle periods with rising memory pressure (gap test, 20 GB), queue vs wired, separate processes,
# interleaved. Run under the GPU lock.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
P=$SP/tools/residency_probe2
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -4
  vm_stat | grep -E "Pages (free|active|inactive|wired|stored in compressor|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
  uptime
}
echo "==== hold 2 start $(date +%T)"
state "before"
for s in "1:req,2:cball" "1:req,2:tiny" "1:addq,1:req,2:cball" "1:addq,1:req,2:cball,6.5:tiny" "1:req,2:cball,6.5:tiny" \
         "1:addq,1:req,2:cball,6.5:cball" "1:req,2:cball,6.5:cball" "1:req,1:beat,2:burst" "1:addq,1:req,2:burst,7.5:tiny"; do
  echo "==== watch $s"
  $P watch $s 8 10
done
state "before gap tests"
for m in queue wired queue wired; do
  echo "==== gap $m 20 GB, 3 rounds"
  PROBE_HOG=$SP/tools/memhog $P gap $m 20 3
done
state "after"
echo "==== hold 2 end $(date +%T)"
