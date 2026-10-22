#!/bin/zsh
# Lock hold 1: Metal residency semantics and the synthetic stall test (residency_probe), with and without memory
# pressure (memhog). Run under: caffeinate -i -s lockf -k -t 3600 ~/.cache/nace-gpu.lock zsh hold1_probe.sh
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
P=$SP/tools/residency_probe
H=$SP/tools/memhog
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -5
  vm_stat | grep -E "Pages (free|active|inactive|wired|stored in compressor|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
  uptime
}
echo "==== hold 1 start $(date +%T)"
state "before"
for m in none request beat cb queue endres; do
  echo "==== watch $m 8 GB"
  $P watch $m 8 9
done
state "before stall (no pressure)"
echo "==== stall 20 GB, no pressure"
$P stall queue,wired,queue,wired,none 20 15 2 4
state "before stall with memhog"
echo "==== stall 20 GB, memhog started during the 25 s delay"
PROBE_DELAY_S=25 $P stall queue,wired,queue,wired 20 15 2 4 &
PROBE=$!
for i in {1..240}; do  # until the probe has allocated (its "ready" line is in this log)
  grep -q "^ready" $SP/hold1.log 2>/dev/null && break
  /bin/sleep 0.25
done
$H 12 120 256 512 2 2> $SP/hold1_memhog.log &
HOG=$!
wait $PROBE
kill $HOG 2>/dev/null
wait $HOG 2>/dev/null
cat $SP/hold1_memhog.log
state "after"
echo "==== hold 1 end $(date +%T)"
