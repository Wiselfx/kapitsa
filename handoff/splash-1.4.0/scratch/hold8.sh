#!/bin/zsh
# Lock hold 8: the memory-pressure path, simulated with NI_RESIDENCY_PRESSURE_TEST (no real pressure, no hog).
# One 35B server (new build): R1 cold, R2 cached, R3 after 10 s idle (normal); "warn" written; R4 after 10 s idle;
# R5 after 10 s more (still warn); "normal" written; R6 (restarts the heartbeat); R7 after 10 s idle.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
MODEL=/Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit
OUT=$SP/r8
PORT=8391
mkdir -p $OUT
rm -f $OUT/*
F=$OUT/pressure
echo normal > $F
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -4
  vm_stat | grep -E "Pages (free|inactive|wired|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
  uptime
}
mem() {
  echo "  [$(date +%T)] wired $(vm_stat | grep 'wired down' | awk '{printf "%.1f GB", $4 * 16384 / 1e9}'), file says $(cat $F)"
}
echo "==== hold 8 start $(date +%T)"
state "before"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT in use"; exit 1; fi
env NI_LOG=1 NI_HISTORY_PREFILL=0 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1 NI_CB_LAT=$OUT/W.cblat \
  NI_RESIDENCY_PRESSURE_TEST=$F $REPO/build/kapitsa serve --model $MODEL --port $PORT --slots 1 --context 32k \
  > $OUT/W.srv 2>&1 &
SRV=$!
ok=0
for k in {1..360}; do
  grep -q "^Ready:" $OUT/W.srv 2>/dev/null && { ok=1; break; }
  kill -0 $SRV 2>/dev/null || break
  /bin/sleep 0.5
done
echo "==== server ready=$ok"
if [[ $ok == 1 ]]; then
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R1 cold"
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R2 cached"
  /bin/sleep 10
  mem
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R3 normal, 10 s after R2"
  /bin/sleep 2
  echo warn > $F
  echo "  [$(date +%T)] wrote warn"
  for k in 1 2 3 4 5; do /bin/sleep 1; mem; done
  grep "memory pressure" $OUT/W.srv
  /bin/sleep 3
  mem
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R4 warn, 10 s after R3"
  /bin/sleep 10
  mem
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R5 still warn, 10 s after R4"
  /bin/sleep 2
  mem
  echo normal > $F
  echo "  [$(date +%T)] wrote normal"
  /bin/sleep 2
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R6 normal again, 4 s after R5"
  /bin/sleep 10
  mem
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R7 normal, 10 s after R6"
fi
kill $SRV 2>/dev/null
for k in {1..60}; do kill -0 $SRV 2>/dev/null || break; /bin/sleep 0.5; done
kill -9 $SRV 2>/dev/null
wait $SRV 2>/dev/null
grep -E "chat: prompt|memory pressure" $OUT/W.srv | sed 's/^.*\(chat: prompt\)/\1/' | cut -c1-200
python3 $SP/tools/cblat.py $OUT/W.cblat | tail -1
state "after"
echo "==== hold 8 end $(date +%T)"
