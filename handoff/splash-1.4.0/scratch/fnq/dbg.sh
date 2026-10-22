#!/bin/zsh
# dbg.sh NAME STORE_GB: why plain decode is slow with streamed experts (run under lockf and deadline.py). ni_run on the
# decode prompt, 24 tokens: plain with steps in flight (as the server), plain one step at a time (--kv-grow), MTP depth
# 2; each capped, with a 5 s `sample` of its threads 45 s in when still running.
set -u
NAME=$1; STORE=$2
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
OUT=$Q/runs/$NAME
mkdir -p $OUT
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_HOME=$Q/home NI_EXPERT_STORE=$STORE
capped() {  # name secs cmd ...
  local n=$1 secs=$2; shift 2
  echo "$(date '+%H:%M:%S') $n: $*" >> $OUT/dbg.log
  "$@" > $OUT/$n.txt 2>&1 &
  local P=$!
  local i=0
  while kill -0 $P 2>/dev/null && [ $i -lt $secs ]; do
    sleep 1; i=$((i + 1))
    if [ $i -eq 45 ] && kill -0 $P 2>/dev/null; then sample $P 5 -file $OUT/$n.sample.txt > /dev/null 2>&1; i=$((i + 5)); fi
  done
  if kill -0 $P 2>/dev/null; then echo "$(date '+%H:%M:%S') $n: over $secs s, stopped" >> $OUT/dbg.log; kill -TERM $P; sleep 3; kill -KILL $P 2>/dev/null; fi
  wait $P 2>/dev/null
  echo "$(date '+%H:%M:%S') $n: done ($i s)" >> $OUT/dbg.log
}
capped plain_ahead 130 $W/build/ni_run $M @$Q/plan/dec1.ids 24 0
capped plain_serial 130 $W/build/ni_run $M @$Q/plan/dec1.ids 24 0 --kv-grow
capped mtp2 90 $W/build/ni_run $M @$Q/plan/dec1.ids 24 0 --spec 2
echo "dbg done" >> $OUT/dbg.log
