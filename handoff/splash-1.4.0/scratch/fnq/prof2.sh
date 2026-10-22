#!/bin/zsh
# prof2.sh NAME: NI_PROFILE=1 per-kernel GPU time of the prefill (each dispatch in its own command buffer), with one
# decode step so the report prints totals (run under lockf and deadline.py); 12 GB store:
#  1. the 4K prompt, 2048-row chunks   2. the 4K prompt, 4096-row chunks   3. the 16K prompt, 4096-row chunks
set -u
NAME=$1
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
OUT=$Q/runs/$NAME
mkdir -p $OUT
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_HOME=$Q/home
capped() {  # name secs cmd ...
  local n=$1 secs=$2; shift 2
  echo "$(date '+%H:%M:%S') $n: $*" >> $OUT/prof.log
  "$@" > $OUT/$n.txt 2>&1 &
  local P=$!
  local i=0
  while kill -0 $P 2>/dev/null && [ $i -lt $secs ]; do sleep 1; i=$((i + 1)); done
  if kill -0 $P 2>/dev/null; then echo "$(date '+%H:%M:%S') $n: over $secs s, stopped" >> $OUT/prof.log; kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; fi
  wait $P 2>/dev/null
  echo "$(date '+%H:%M:%S') $n: done ($i s)" >> $OUT/prof.log
}
capped pf4_profile 120 env NI_EXPERT_STORE=12 NI_CHUNK=2048 NI_PROFILE=1 $W/build/ni_run $M @$Q/plan/pf4_seed11.ids 2 0
capped pf4_profile_c4096 120 env NI_EXPERT_STORE=12 NI_CHUNK=4096 NI_PROFILE=1 $W/build/ni_run $M @$Q/plan/pf4_seed11.ids 2 0
capped pf16_profile_c4096 220 env NI_EXPERT_STORE=12 NI_CHUNK=4096 NI_PROFILE=1 $W/build/ni_run $M @$Q/plan/pf16_seed33.ids 2 0
echo "prof done" >> $OUT/prof.log
