#!/bin/zsh
# Lock hold 6: the memory-pressure source. One 35B server (new build, default): R1 cold, R2 cached, R3 cached after
# 10 s idle (pressure normal); then a memhog (at most 6 GB, stops at the warning level, gives that step back, +128 MB
# swap guard) with the server idle, 8 s more idle, R4; the hog stops; R5 and R6 each after 10 s idle.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
MODEL=/Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit
OUT=$SP/r6
PORT=8391
mkdir -p $OUT
rm -f $OUT/*
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -4
  vm_stat | grep -E "Pages (free|inactive|speculative|wired|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
  uptime
}
mem() {
  echo "  [$(date +%T)] wired $(vm_stat | grep 'wired down' | awk '{printf "%.1f GB", $4 * 16384 / 1e9}'), pressure level $(sysctl -n kern.memorystatus_vm_pressure_level), swap $(sysctl -n vm.swapusage | awk '{print $6}')"
}
echo "==== hold 6 start $(date +%T)"
state "before"
if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then echo "port $PORT in use"; exit 1; fi
env NI_LOG=1 NI_HISTORY_PREFILL=0 KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1 NI_CB_LAT=$OUT/W.cblat $REPO/build/kapitsa \
  serve --model $MODEL --port $PORT --slots 1 --context 32k > $OUT/W.srv 2>&1 &
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
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R3 cached after 10 s idle, pressure normal"
  /bin/sleep 3
  mem
  $SP/tools/memhog 6 90 128 512 2 2> $OUT/hog.log &
  HOG=$!
  for k in {1..60}; do
    grep -q "memhog holding\|memhog done" $OUT/hog.log 2>/dev/null && break
    /bin/sleep 0.5
  done
  grep -E "holding|swap grew" $OUT/hog.log
  mem
  /bin/sleep 4
  mem
  /bin/sleep 4
  mem
  grep "memory pressure" $OUT/W.srv
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R4 cached, idle server after the hog reached the warning level"
  kill $HOG 2>/dev/null; wait $HOG 2>/dev/null
  grep -E "done|swap grew" $OUT/hog.log
  /bin/sleep 10
  mem
  grep "memory pressure" $OUT/W.srv
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R5 cached after 10 s idle, hog gone"
  /bin/sleep 10
  mem
  python3 $SP/tools/ttft_client.py $PORT 9000 16 "R6 cached after 10 s idle"
fi
kill $SRV 2>/dev/null
for k in {1..60}; do kill -0 $SRV 2>/dev/null || break; /bin/sleep 0.5; done
kill -9 $SRV 2>/dev/null
wait $SRV 2>/dev/null
grep -E "chat: prompt|memory pressure" $OUT/W.srv | sed 's/^.*\(chat: prompt\)/\1/' | cut -c1-230
python3 $SP/tools/cblat.py $OUT/W.cblat | tail -1
state "after"
echo "==== hold 6 end $(date +%T)"
