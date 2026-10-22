#!/bin/bash
# Under the GPU lock. Rounds in M B B M order (odd rounds the configs as given, even rounds reversed, so drift within
# the session is charged to neither): per config, the 4B server, the client's requests, the server stopped.
# usage: ab_mbbm.sh RUN MODE ROUNDS CONFIG...   CONFIG = name:binary:env (env: VAR=v,VAR=v or -)
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
RUN=$1 MODE=$2 ROUNDS=$3
shift 3
CONFIGS=("$@")
PORT=8475
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
mkdir -p "$S/$RUN"
date
python3 "$W/bench/harness/gpu_hogs.py" 2>&1 | tail -3
memory_pressure | tail -1
uptime
for round in $(seq 1 "$ROUNDS"); do
  order=("${CONFIGS[@]}")
  if [ $((round % 2)) -eq 0 ]; then
    order=()
    for ((k = ${#CONFIGS[@]} - 1; k >= 0; k--)); do order+=("${CONFIGS[$k]}"); done
  fi
  for cfg in "${order[@]}"; do
    name=${cfg%%:*}
    rest=${cfg#*:}
    bin=${rest%%:*}
    envs=${rest#*:}
    [ "$envs" = "-" ] && envs=""
    log="$S/$RUN/server-$name-$round.log"
    env $(echo "$envs" | tr ',' ' ') "$bin" serve --model "$HOME/.cache/kapitsa/models/qwen3.5-4b-4bit" --port $PORT \
      --slots 2 --ctx 16384 > "$log" 2>&1 &
    SRV=$!
    n=0
    until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
      n=$((n + 1))
      if [ $n -gt 180 ] || ! kill -0 $SRV 2>/dev/null; then echo "server $name did not start"; break; fi
      sleep 1
    done
    "$S/venv/bin/python" "$S/ab_client.py" "http://127.0.0.1:$PORT" "$S/$RUN/out-$name-$round.json" "$MODE" \
      || echo "client failed: $name $round"
    kill $SRV
    wait $SRV 2>/dev/null
    echo "round $round $name: $(grep -c 'tok/s' "$log") requests"
  done
done
python3 "$W/bench/harness/gpu_hogs.py" 2>&1 | tail -3
uptime
date
