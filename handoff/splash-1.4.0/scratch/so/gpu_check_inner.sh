#!/bin/bash
# Under the GPU lock, a hard deadline of 9.5 minutes for the hold: per config (name:binary:env), the 4B server with
# NI_BATCH=1 and 4 slots, gpu_check_client.py, the server stopped. A watchdog stops the server (and this script) at the
# deadline; a config is not started with less than 75 s left.
# usage: gpu_check_inner.sh RUN CONFIG...
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
RUN=$1
shift
DEADLINE=$(( $(date +%s) + 565 ))
PORT=8477
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
mkdir -p "$S/$RUN"
date
python3 "$W/bench/harness/gpu_hogs.py" 2>&1 | tail -3
memory_pressure | tail -1
uptime
for cfg in "$@"; do
  IFS=: read -r name bin envs <<< "$cfg"
  [ "$envs" = "-" ] && envs=""
  left=$(( DEADLINE - $(date +%s) ))
  if [ $left -lt 75 ]; then echo "skipped $name: $left s left"; continue; fi
  log="$S/$RUN/server-$name.log"
  env NI_BATCH=1 NI_LOG=2 $(echo "$envs" | tr ',' ' ') "$bin" serve --model "$HOME/.cache/kapitsa/models/qwen3.5-4b-4bit" \
    --port $PORT --slots 4 --ctx 16384 > "$log" 2>&1 &
  SRV=$!
  ( sleep "$left"; echo "hard deadline: stopping $name"; kill -TERM $SRV 2>/dev/null; sleep 5; kill -KILL $SRV 2>/dev/null ) &
  WD=$!
  n=0
  until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
    n=$((n + 1))
    if [ $n -gt 120 ] || ! kill -0 $SRV 2>/dev/null; then echo "server $name did not start"; break; fi
    sleep 1
  done
  check=1
  case "$name" in main*) check=0 ;; esac
  "$S/venv/bin/python" "$S/gpu_check_client.py" "http://127.0.0.1:$PORT" "$S/$RUN/out-$name.json" $check \
    > "$S/$RUN/client-$name.log" 2>&1
  echo "$name: client exit=$? $(tail -1 "$S/$RUN/client-$name.log")"
  kill $SRV 2>/dev/null
  wait $SRV 2>/dev/null
  kill $WD 2>/dev/null
  wait $WD 2>/dev/null
  echo "$name: batched cycles $(grep -c 'batch: ' "$log"), budget summaries $(grep -c 'KFlash batch:' "$log"), crashes $(grep -c -i 'fatal\|GPU error\|abort' "$log")"
done
python3 "$W/bench/harness/gpu_hogs.py" 2>&1 | tail -3
date
