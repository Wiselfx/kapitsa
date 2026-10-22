#!/bin/zsh
# sweep.sh NAME: the expert store's size and buffers against the GPU's per-command-buffer cost (run under lockf and
# deadline.py). ni_run on the decode prompt, 16 tokens, MTP depth 2 (one command buffer a cycle) and plain.
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
  echo "$(date '+%H:%M:%S') $n: $*" >> $OUT/sweep.log
  { date; vm_stat | grep -E "wired|free:"; } > $OUT/$n.vm 2>&1
  "$@" > $OUT/$n.txt 2>&1 &
  local P=$!
  local i=0
  while kill -0 $P 2>/dev/null && [ $i -lt $secs ]; do
    sleep 1; i=$((i + 1))
    if [ $i -eq 30 ] && kill -0 $P 2>/dev/null; then vm_stat | grep -E "wired|free:" >> $OUT/$n.vm; fi
  done
  if kill -0 $P 2>/dev/null; then echo "$(date '+%H:%M:%S') $n: over $secs s, stopped" >> $OUT/sweep.log; kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; fi
  wait $P 2>/dev/null
  echo "$(date '+%H:%M:%S') $n: done ($i s)" >> $OUT/sweep.log
}
capped s6 75 env NI_EXPERT_STORE=6 $W/build/ni_run $M @$Q/plan/dec1.ids 16 0 --spec 2
capped s15 75 env NI_EXPERT_STORE=15 $W/build/ni_run $M @$Q/plan/dec1.ids 16 0 --spec 2
capped s18c9 75 env NI_EXPERT_STORE=18 NI_EXPERT_CHUNK_GB=9 $W/build/ni_run $M @$Q/plan/dec1.ids 16 0 --spec 2
capped s18 60 env NI_EXPERT_STORE=18 $W/build/ni_run $M @$Q/plan/dec1.ids 8 0 --spec 2
capped s24c12 75 env NI_EXPERT_STORE=24 NI_EXPERT_CHUNK_GB=12 $W/build/ni_run $M @$Q/plan/dec1.ids 16 0 --spec 2
capped s18c9_plain 75 env NI_EXPERT_STORE=18 NI_EXPERT_CHUNK_GB=9 $W/build/ni_run $M @$Q/plan/dec1.ids 16 0
echo "sweep done" >> $OUT/sweep.log
