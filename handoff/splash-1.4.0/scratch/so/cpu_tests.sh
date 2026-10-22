#!/bin/bash
# AGENTS.md's CPU tests (no model on the GPU), at background QoS; jsonschema from the scratch venv
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
PY=$S/venv/bin/python
cd "$W" || exit 1
run() {
  local name=$1
  shift
  local t0=$(date +%s)
  taskpolicy -b "$@" > "$S/cpu-$name.log" 2>&1
  local rc=$?
  echo "$name: exit $rc ($(( $(date +%s) - t0 )) s) $(tail -1 "$S/cpu-$name.log" | cut -c1-120)"
}
run api_test build/api_test
run test_server_cli "$PY" tests/test_server.py --cli build/kapitsa
run test_sidecar "$PY" tests/test_sidecar.py build/kapitsa
run test_bench "$PY" tests/test_bench.py build
run test_update "$PY" tests/test_update.py build
run test_community "$PY" tests/test_community.py build
run test_reasoning "$PY" tests/test_reasoning.py
run test_pr_size "$PY" tests/test_pr_size.py
run test_structured "$PY" tests/test_structured.py --cli build
