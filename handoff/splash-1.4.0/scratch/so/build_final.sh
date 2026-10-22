#!/bin/bash
# The branch's current source built into so/build-final at background QoS (build/ stays as session 3 uses it)
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a85679af4f1631100
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
taskpolicy -b cmake -S "$W" -B "$S/build-final" -DCMAKE_BUILD_TYPE=Release > "$S/final-cmake.log" 2>&1 || exit 1
taskpolicy -b cmake --build "$S/build-final" -j 4 --target kapitsa > "$S/final-build.log" 2>&1
echo "exit=$?"
ls -la "$S/build-final/kapitsa"
