#!/bin/zsh
# Local CI parity for the PR branches: incremental Release build of the CI targets, then the CI's CPU tests
# (test_community.py skipped: it needs jsonschema). Builds run in parallel, tests one branch at a time.
WT=/Users/flaminis/Desktop/kapitsa/.claude/worktrees
OUT=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/prs/verify
rm -rf $OUT && mkdir -p $OUT
: > $OUT/summary.txt

verify() {
  local w=$1 name=$2 targets=$3 extra=$4
  {
    cd $WT/$w || exit 1
    echo "== $name $(git rev-parse --short HEAD) $(date +%T)"
    [ -f build/CMakeCache.txt ] || cmake -S . -B build -DCMAKE_BUILD_TYPE=Release || exit 2
    cmake --build build -j3 --target ${=targets} || exit 3
    echo "== build ok $(date +%T)"
    lockf -k $OUT/tests.lock zsh -c "
      set -e
      build/api_test
      python3 tests/test_server.py --cli build/kapitsa
      python3 tests/test_sidecar.py build/kapitsa
      python3 tests/test_update.py build
      python3 tests/test_reasoning.py
      $extra
    " || exit 4
    echo "== tests ok $(date +%T)"
  } > $OUT/$name.log 2>&1
  echo "$name rc=$?" >> $OUT/summary.txt
}

verify agent-a1bca8900e5443872 residency "kapitsa api_test bench_test update_test" "" &
verify splash-1-4-0-analysis-3bbeaf batch "kapitsa api_test bench_test update_test" "" &
verify agent-ad2390d3db3fba8da logit "kapitsa api_test bench_test update_test" "python3 tests/test_logit_compare.py" &
verify agent-a85679af4f1631100 structured "kapitsa api_test bench_test update_test grammar_test structured_check" "python3 tests/test_structured.py --cli build" &
verify agent-a6f34d21761077bfc ssd "kapitsa api_test bench_test update_test prefix_cache_test" "build/prefix_cache_test" &
wait
cat $OUT/summary.txt
