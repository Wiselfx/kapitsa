#!/bin/bash
# Under the GPU lock: the 4B server with NI_BATCH=1 (its DFlash drafter batches too), requests at once: two under a
# schema, two free; then the server's batch counts.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
TAG=${1:-batch}
PORT=8473
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
NI_BATCH=1 NI_LOG=2 "${KBIN:-$W/build/kapitsa}" serve --model "$HOME/.cache/kapitsa/models/qwen3.5-4b-4bit" --port $PORT --slots 4 \
  --ctx 16384 > "$S/server-$TAG.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
n=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  n=$((n + 1))
  if [ $n -gt 180 ] || ! kill -0 $SRV 2>/dev/null; then echo "server did not start"; exit 1; fi
  sleep 1
done
"$S/venv/bin/python" "$S/batch_client.py" "http://127.0.0.1:$PORT"
echo "batch client exit=$?"
grep -E "cycles batched|structured output|batch:" "$S/server-$TAG.log" | cut -c1-260 | tail -20
