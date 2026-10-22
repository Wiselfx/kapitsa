#!/bin/zsh
# sweep2.sh NAME: store sizes 4-12 GB for decode (MTP depth 2 and plain, 16 tokens) and the 4K prefill at 6 / 12 GB with
# 2048- and 4096-row chunks (run under lockf and deadline.py).
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
  "$@" > $OUT/$n.txt 2>&1 &
  local P=$!
  local i=0
  while kill -0 $P 2>/dev/null && [ $i -lt $secs ]; do
    sleep 1; i=$((i + 1))
    if [ $i -eq 20 ] && kill -0 $P 2>/dev/null; then { vm_stat | grep -E "wired|free:"; sysctl -n kern.memorystatus_level; } > $OUT/$n.vm; fi
  done
  if kill -0 $P 2>/dev/null; then echo "$(date '+%H:%M:%S') $n: over $secs s, stopped" >> $OUT/sweep.log; kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; fi
  wait $P 2>/dev/null
  echo "$(date '+%H:%M:%S') $n: done ($i s)" >> $OUT/sweep.log
}
for S in 4 8 10 12; do
  capped mtp_s$S 50 env NI_EXPERT_STORE=$S $W/build/ni_run $M @$Q/plan/dec1.ids 16 0 --spec 2
done
for S in 6 10; do
  capped plain_s$S 60 env NI_EXPERT_STORE=$S $W/build/ni_run $M @$Q/plan/dec1.ids 16 0
done
capped pf4_s6_c2048 65 env NI_EXPERT_STORE=6 NI_CHUNK=2048 $W/build/ni_run $M @$Q/plan/pf4_seed11.ids 1 0
capped pf4_s12_c2048 65 env NI_EXPERT_STORE=12 NI_CHUNK=2048 $W/build/ni_run $M @$Q/plan/pf4_seed11.ids 1 0
capped pf4_s12_c4096 65 env NI_EXPERT_STORE=12 NI_CHUNK=4096 $W/build/ni_run $M @$Q/plan/pf4_seed11.ids 1 0
echo "sweep done" >> $OUT/sweep.log
