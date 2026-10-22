#!/bin/bash
# Runs under the GPU lock: the 4B server (this branch), the structured-output end-to-end test, then the server stops.
# usage: e2e_inner.sh TAG [test args...]
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
TAG=$1
shift
PORT=8471
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
"$W/build/kapitsa" serve --model "$HOME/.cache/kapitsa/models/qwen3.5-4b-4bit" --port $PORT --slots 2 --ctx 16384 \
  > "$S/server-$TAG.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
n=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  n=$((n + 1))
  if [ $n -gt 180 ] || ! kill -0 $SRV 2>/dev/null; then echo "server did not start"; exit 1; fi
  sleep 1
done
"$S/venv/bin/python" "$W/tests/test_structured.py" "http://127.0.0.1:$PORT" "$@" > "$S/e2e-$TAG.log" 2>&1
echo "e2e exit=$?"
grep -E "^FAIL|schema-valid|ALL PASSED|FAILED" "$S/e2e-$TAG.log" | cut -c1-400
