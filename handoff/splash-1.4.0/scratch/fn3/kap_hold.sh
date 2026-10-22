#!/bin/zsh
# kap_hold.sh NAME SPEC SEED [hbench.py arguments ...]: one round (run under launch.sh): kapitsa serve (origin/main
# build) on the Q2_0 pack with the automatic expert store (NI_EXPERT_STORE unset unless STORE is set), the seeded
# harness against it, then the server stopped. SERVE_EXTRA adds serve flags. Everything to runs/NAME.
set -u
NAME=$1; SPEC=$2; SEED=$3; shift 3
W=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/kap-main
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
OUT=$Q/runs/$NAME
PORT=8391
mkdir -p $OUT $Q/home
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_HOME=$Q/home
/bin/zsh $Q/precheck.sh $OUT/precheck.txt
date '+%s' > $OUT/t_start
STOREENV=()
[ -n "${STORE:-}" ] && STOREENV=(NI_EXPERT_STORE=$STORE)
echo "store ${STORE:-auto}, serve extra: ${SERVE_EXTRA:-none}" >> $OUT/hold.log
echo "cmd: env $STOREENV NI_LOG=1 $W/build/kapitsa serve --model $M --model-id flash-q20 --dflash none --spec $SPEC --host 127.0.0.1 --port $PORT --ctx 32768 --slots 1 ${SERVE_EXTRA:-}" >> $OUT/hold.log
env $STOREENV NI_LOG=1 $W/build/kapitsa serve --model $M --model-id flash-q20 \
  --dflash none --spec $SPEC --host 127.0.0.1 --port $PORT --ctx 32768 --slots 1 ${=SERVE_EXTRA:-} > $OUT/server.log 2>&1 &
SPID=$!
( while kill -0 $SPID 2>/dev/null; do echo "$(date '+%s') $(sysctl -n kern.memorystatus_level) $(sysctl -n vm.swapusage | awk '{print $6}')"; sleep 5; done ) > $OUT/memlevel.txt 2>&1 &
python3 $W/bench/harness/gpu_hogs.py monitor $OUT/gpu.txt $SPID > /dev/null 2>&1 &
MON=$!
n=0
until curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1; do
  n=$((n + 1))
  if [ $n -gt 300 ] || ! kill -0 $SPID 2>/dev/null; then echo "server not ready after $n s" >> $OUT/hold.log; break; fi
  sleep 1
done
date '+%s' > $OUT/t_ready
if kill -0 $SPID 2>/dev/null; then
  FNQ_SEED=$SEED FNQ_LOG=$OUT/requests.jsonl python3 $Q/hbench.py --base-url http://127.0.0.1:$PORT/v1 --model flash-q20 \
    --label $NAME --pid $SPID --no-gpu-lock --extra-body '{"seed": 1234}' --out $OUT/result.json "$@" > $OUT/bench.log 2>&1
  echo "hbench exit $?" >> $OUT/hold.log
fi
kill -TERM $SPID 2>/dev/null
n=0
while kill -0 $SPID 2>/dev/null && [ $n -lt 30 ]; do sleep 1; n=$((n + 1)); done
kill -KILL $SPID 2>/dev/null
kill -TERM $MON 2>/dev/null
date '+%s' > $OUT/t_end
echo "hold done" >> $OUT/hold.log
