#!/bin/bash
# Under the GPU lock: a model's server (this branch) and the structured-output e2e test, then the server stops.
# usage: e2e_model.sh TAG MODEL_DIR ENV SERVE_ARGS [test args...]   (ENV: VAR=v,VAR=v or -; SERVE_ARGS: "--x y" or -)
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
TAG=$1
MODEL=$2
ENVS=$3
ARGS=$4
shift 4
[ "$ENVS" = "-" ] && ENVS=""
[ "$ARGS" = "-" ] && ARGS=""
PORT=8474
export NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1
date
env $(echo "$ENVS" | tr ',' ' ') "${KBIN:-$W/build/kapitsa}" serve --model "$MODEL" --port $PORT --slots 2 --ctx 16384 $ARGS \
  > "$S/server-$TAG.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
n=0
until curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models"; do
  n=$((n + 1))
  if [ $n -gt 240 ] || ! kill -0 $SRV 2>/dev/null; then echo "server did not start"; exit 1; fi
  sleep 1
done
grep "engine ready" "$S/server-$TAG.log" | cut -c1-200
"$S/venv/bin/python" "$W/tests/test_structured.py" "http://127.0.0.1:$PORT" "$@" > "$S/e2e-$TAG.log" 2>&1
echo "e2e $TAG exit=$?"
grep -E "^FAIL|schema-valid|ALL PASSED|FAILED" "$S/e2e-$TAG.log" | cut -c1-400
date
