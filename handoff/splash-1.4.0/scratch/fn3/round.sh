#!/bin/zsh
# round.sh ENGINE NAME "SEED|hbench args" ...: one GPU round (run under launch.sh). ENGINE is kapitsa (SPEC: --spec,
# default 2; SERVE_EXTRA: more serve flags; the automatic expert store), splash (SPLASH_EXTRA: more serve flags) or
# strata (STRATA_CFG: its run config). The server starts once; each "SEED|args" is one seeded harness invocation
# against it (FNQ_SEED=SEED: the same prompts and nonces as this morning's runs with that seed and those arguments),
# its result in runs/NAME/inv<i>-s<SEED>.json. Then the whole server tree is stopped.
set -u
ENGINE=$1; NAME=$2; shift 2
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
K=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/kap-main
OUT=$Q/runs/$NAME
mkdir -p $OUT
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
/bin/zsh $Q/precheck.sh $OUT/precheck.txt
# wait (up to 5 min) until no foreign process uses more than 5% of the GPU for 15 s; the Claude app this runs from is ambient
BENCH_GPU_AMBIENT="Claude Helper,Claude" python3 $K/bench/harness/gpu_hogs.py wait 5 15 300 >> $OUT/precheck.txt 2>&1
echo "$(date '+%H:%M:%S') quiet wait exit $?" >> $OUT/hold.log
date '+%s' > $OUT/t_start
HUB=$HOME/.cache/flash-next-gguf/hub
CLEAR=()
case $ENGINE in
  kapitsa)
    M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
    PORT=8391
    mkdir -p $Q/home
    CMD=($K/build/kapitsa serve --model $M --model-id flash-q20 --dflash none --spec ${SPEC:-2} --host 127.0.0.1
         --port $PORT --ctx 32768 --slots 1 ${=SERVE_EXTRA:-})
    ENVS=(NI_LOG=1 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_HOME=$Q/home)
    [ -n "${STORE:-}" ] && ENVS+=(NI_EXPERT_STORE=$STORE)
    ;;
  splash)
    S="/Users/flaminis/Desktop/Nace Agent/research/inference/engines-src/splash-release/splash-1.4.0-arm64-macos26"
    DRAFT=$HUB/models--incoai--Qwen3.8-Flash-Next-DFlash2/snapshots/a22a1744e85c17364dcc54a6a420af7135d4c7f6
    PORT=8151
    CMD=("$S/python/bin/python3" -u "$S/install/launcher.py" serve
         --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09
         --draft-model $DRAFT --offline --language-only --host 127.0.0.1 --port $PORT --no-webui
         --max-context 32K --idle-release off --max-cache-disk 0 ${=SPLASH_EXTRA:-})
    ENVS=(HF_HUB_CACHE=$HUB HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1)
    CLEAR=(${(k)parameters[(I)SPLASH_*]})
    ;;
  strata)
    ST=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata
    SRC=${STRATA_CFG:-$Q/strata-q2_0.json}
    CFG=$OUT/strata-config.json
    python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['log']=sys.argv[3]; json.dump(c, open(sys.argv[2],'w'), indent=1)" $SRC $CFG $OUT/engine.log
    PORT=8180
    CMD=($ST/.venv/bin/python $ST/serve/server.py --engine strata --config $CFG --port $PORT)
    ENVS=(PYTHONNOUSERSITE=1)
    CLEAR=(${(k)parameters[(I)STRATA_*]})
    ;;
  *) echo "unknown engine $ENGINE" >> $OUT/hold.log; exit 2 ;;
esac
echo "$(date '+%H:%M:%S') cmd: env ${ENVS[*]} ${CMD[*]}" >> $OUT/hold.log
for v in $CLEAR; do unset $v; done
env $ENVS "${CMD[@]}" > $OUT/server.log 2>&1 &
SPID=$!
echo $SPID > $OUT/server.pid
# memory, swap and free disk (MiB) every 2 s; the soft disk guard: below 3.8 GiB free (swap growth) the server is
# stopped and the round marked; guard.sh holds the hard floor (3 GiB) for everything
( while kill -0 $SPID 2>/dev/null; do
    d=$(df -k / | tail -1 | awk '{printf "%d", $4 / 1024}')
    echo "$(date '+%s') $(sysctl -n kern.memorystatus_level) $(sysctl -n vm.swapusage | awk '{print $6}') $d"
    if [ "$d" -lt 3891 ]; then echo "$(date '+%H:%M:%S') soft disk guard: $d MiB free, stopping the server" >> $OUT/hold.log; touch $OUT/SOFT_GUARD; kill -TERM $SPID; fi
    sleep 2
  done ) > $OUT/memlevel.txt 2>&1 &
python3 $K/bench/harness/gpu_hogs.py monitor $OUT/gpu.txt $SPID > /dev/null 2>&1 &
MON=$!
n=0
MODEL=""
until MODEL=$(curl -sf http://127.0.0.1:$PORT/v1/models 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) && [ -n "$MODEL" ]; do
  n=$((n + 1))
  if [ $n -gt ${READY_S:-420} ] || ! kill -0 $SPID 2>/dev/null; then echo "server not ready after $n s" >> $OUT/hold.log; MODEL=""; break; fi
  sleep 1
done
date '+%s' > $OUT/t_ready
echo "$(date '+%H:%M:%S') ready after $n s, model id: $MODEL" >> $OUT/hold.log
i=0
for inv in "$@"; do
  [ -f $Q/ABORT ] && break
  i=$((i + 1))
  SEED=${inv%%|*}; ARGS=${inv#*|}
  SKIP=0
  if [[ $SEED == *+* ]]; then SKIP=${SEED#*+}; SEED=${SEED%%+*}; fi
  [ -n "$MODEL" ] && kill -0 $SPID 2>/dev/null || break
  [ $ENGINE = splash ] && curl -sf http://127.0.0.1:$PORT/status > $OUT/status_before_inv$i.json 2>/dev/null
  echo "$(date '+%H:%M:%S') inv$i seed $SEED (skip $SKIP draws): $ARGS" >> $OUT/hold.log
  FNQ_SEED=$SEED FNQ_SKIP_DRAWS=$SKIP FNQ_LOG=$OUT/requests_inv$i.jsonl python3 $Q/hbench.py --base-url http://127.0.0.1:$PORT/v1 --model "$MODEL" \
    --label $NAME-inv$i --pid $SPID --no-gpu-lock --extra-body '{"seed": 1234}' --out $OUT/inv$i-s$SEED.json ${=ARGS} > $OUT/bench_inv$i.log 2>&1
  echo "$(date '+%H:%M:%S') inv$i exit $?" >> $OUT/hold.log
done
[ $ENGINE = splash ] && curl -sf http://127.0.0.1:$PORT/status > $OUT/status_end.json 2>/dev/null
tree() { echo $1; for c in $(pgrep -P $1); do tree $c; done; }
PIDS=($(tree $SPID))
echo "$(date '+%H:%M:%S') stopping: ${PIDS[*]}" >> $OUT/hold.log
ps -o pid,ppid,pgid,rss,etime,comm -p ${(j:,:)PIDS} >> $OUT/hold.log 2>&1
kill -TERM $SPID 2>/dev/null
n=0
while kill -0 $SPID 2>/dev/null && [ $n -lt 30 ]; do sleep 1; n=$((n + 1)); done
for p in $PIDS; do kill -0 $p 2>/dev/null && kill -TERM $p 2>/dev/null; done
n=0
while [ $n -lt 10 ]; do alive=0; for p in $PIDS; do kill -0 $p 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 1; n=$((n + 1)); done
for p in $PIDS; do kill -0 $p 2>/dev/null && { echo "KILL $p" >> $OUT/hold.log; kill -KILL $p 2>/dev/null; }; done
kill -TERM $MON 2>/dev/null
date '+%s' > $OUT/t_end
echo "$(date '+%H:%M:%S') hold done" >> $OUT/hold.log
