#!/bin/zsh
# A lock hold of server TTFT runs. Per server (A = base, Q = new with NI_RESIDENCY=queue, W = new default):
#   start; R1 cold request (first use after load); R2 the same request (cached prefix); idle IDLE1 s; R3 (cached,
#   after idle); IDLE2 s idle with a memhog applying pressure, which stays on through R4 (cached, after idle under
#   pressure); stop.
#   hold_server.sh <outdir> <model dir> <idle1 s> <idle2 s> <order: e.g. "A W W A">
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
OUT=$1; MODEL=$2; IDLE1=$3; IDLE2=$4; ORDER=(${=5})
PORT=8391
mkdir -p $OUT
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -4
  vm_stat | grep -E "Pages (free|wired|stored in compressor|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
}
echo "==== server hold start $(date +%T): $MODEL idle $IDLE1 / $IDLE2 (pressure) order $ORDER"
i=0
for lab in $ORDER; do
  i=$((i+1))
  if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT in use: stopping"; break; fi
  state "before server $i ($lab)"
  case $lab in
    A) BIN=$SP/base/build/kapitsa; ENVS=() ;;
    Q) BIN=$REPO/build/kapitsa; ENVS=(NI_RESIDENCY=queue NI_CB_LAT=$OUT/$i-Q.cblat) ;;
    W) BIN=$REPO/build/kapitsa; ENVS=(NI_CB_LAT=$OUT/$i-W.cblat) ;;
  esac
  rm -f $OUT/$i-$lab.*
  t0=$(date +%s)
  env NI_LOG=1 NI_HISTORY_PREFILL=0 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1 $ENVS $BIN serve --model $MODEL \
    --port $PORT --slots 1 --context 32k > $OUT/$i-$lab.srv 2>&1 &
  SRV=$!
  ok=0
  for k in {1..360}; do
    grep -q "^Ready:" $OUT/$i-$lab.srv 2>/dev/null && { ok=1; break; }
    kill -0 $SRV 2>/dev/null || break
    /bin/sleep 0.5
  done
  echo "==== server $i $lab ready=$ok after $(( $(date +%s) - t0 )) s"
  if [[ $ok == 1 ]]; then
    python3 $SP/tools/ttft_client.py $PORT 9000 16 "$i-$lab R1 cold"
    python3 $SP/tools/ttft_client.py $PORT 9000 16 "$i-$lab R2 cached"
    /bin/sleep $IDLE1
    python3 $SP/tools/ttft_client.py $PORT 9000 16 "$i-$lab R3 cached after ${IDLE1}s idle"
    $SP/tools/memhog 12 $((IDLE2 + 60)) 256 512 2 2> $OUT/$i-$lab.hog &
    HOG=$!
    /bin/sleep $IDLE2
    grep -E "holding" $OUT/$i-$lab.hog
    python3 $SP/tools/ttft_client.py $PORT 9000 16 "$i-$lab R4 cached after ${IDLE2}s idle under pressure"
    kill $HOG 2>/dev/null; wait $HOG 2>/dev/null
    grep -E "done|swap grew" $OUT/$i-$lab.hog
  fi
  kill $SRV 2>/dev/null
  for k in {1..60}; do kill -0 $SRV 2>/dev/null || break; /bin/sleep 0.5; done
  kill -9 $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  grep -E "chat: prompt" $OUT/$i-$lab.srv | sed 's/^.*\(chat: prompt\)/\1/' | cut -c1-260
  [[ -f $OUT/$i-$lab.cblat ]] && python3 $SP/tools/cblat.py $OUT/$i-$lab.cblat
done
state "after"
echo "==== server hold end $(date +%T)"
