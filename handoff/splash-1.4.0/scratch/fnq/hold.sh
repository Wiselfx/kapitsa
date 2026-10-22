#!/bin/zsh
# hold.sh NAME SPEC SEED STORE_GB BUDGET_GB [hbench.py arguments ...]
# One GPU hold (run under lockf and deadline.py): kapitsa serve on the Q2_0 pack (streamed experts, a STORE_GB expert
# store, NI_MEM_BUDGET BUDGET_GB), the seeded harness (hbench.py) against it, then the server stopped. Everything goes
# to runs/NAME: the server's log, the harness's log and result, every request with its reply, memory samples.
set -u
NAME=$1; SPEC=$2; SEED=$3; STORE=$4; BUDGET=$5; shift 5
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
OUT=$Q/runs/$NAME
PORT=8391
mkdir -p $OUT
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_HOME=$Q/home
mkdir -p $Q/home
date '+%s' > $OUT/t_start
{ date; vm_stat; sysctl vm.swapusage kern.memorystatus_level; df -g /System/Volumes/Data | tail -1; } > $OUT/sys_start.txt 2>&1
ps -axo pid,rss,etime,comm | awk '$4 ~ /kapitsa|ni_run|fn_eval|nh_|llama|ollama|mlx|logit_dump|python/' > $OUT/ps_start.txt
# STORE "auto": what vm_stat counts free (free + inactive + speculative + purgeable) less 9 GB (8 GB left free and a
# margin) less 10.5 GB for the rest of the server (weights, MTP head, buffers, KV, host side), 6..30 GB
if [ "$STORE" = auto ]; then
  AVAIL=$(vm_stat | awk '/Pages (free|inactive|speculative|purgeable)/ {gsub("\\.","",$NF); s += $NF} END {printf "%.1f", s * 16384 / 1e9}')
  STORE=$(python3 -c "print(max(6.0, min(30.0, int(($AVAIL - 9 - 10.5) * 2) / 2)))")
  BUDGET=$(python3 -c "print($STORE + 11)")
  echo "auto: $AVAIL GB available -> store $STORE GB, budget $BUDGET GB" >> $OUT/hold.log
fi
echo "store $STORE GB, budget $BUDGET GB, serve extra: ${SERVE_EXTRA:-none}" >> $OUT/hold.log
env NI_EXPERT_STORE=$STORE NI_MEM_BUDGET=$BUDGET NI_LOG=1 $W/build/kapitsa serve --model $M --model-id flash-q20 \
  --dflash none --spec $SPEC --host 127.0.0.1 --port $PORT --ctx 32768 --slots 1 ${=SERVE_EXTRA:-} > $OUT/server.log 2>&1 &
SP=$!
( while kill -0 $SP 2>/dev/null; do echo "$(date '+%s') $(sysctl -n kern.memorystatus_level) $(sysctl -n vm.swapusage | awk '{print $6}')"; sleep 5; done ) > $OUT/memlevel.txt 2>&1 &
n=0
until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; do
  n=$((n + 1))
  if [ $n -gt 300 ] || ! kill -0 $SP 2>/dev/null; then echo "server not ready after $n s" >> $OUT/hold.log; break; fi
  sleep 1
done
date '+%s' > $OUT/t_ready
if kill -0 $SP 2>/dev/null; then
  FNQ_SEED=$SEED FNQ_LOG=$OUT/requests.jsonl python3 $Q/hbench.py --base-url http://127.0.0.1:$PORT/v1 --model flash-q20 \
    --label $NAME --pid $SP --no-gpu-lock --extra-body '{"seed": 1234}' --out $OUT/result.json "$@" > $OUT/bench.log 2>&1
  echo "hbench exit $?" >> $OUT/hold.log
fi
{ date; vm_stat; sysctl vm.swapusage kern.memorystatus_level; } > $OUT/sys_end.txt 2>&1
kill -TERM $SP 2>/dev/null
n=0
while kill -0 $SP 2>/dev/null && [ $n -lt 30 ]; do sleep 1; n=$((n + 1)); done
kill -KILL $SP 2>/dev/null
date '+%s' > $OUT/t_end
echo "hold done" >> $OUT/hold.log
