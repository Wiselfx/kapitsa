#!/bin/zsh
# Builds the origin/main baseline (exported with git archive into ./base) for the A/B.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
cd "$SP/base" || exit 1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=0553f55 > "$SP/base-cmake.log" 2>&1 || { echo "cmake failed" >> "$SP/base-build.log"; exit 1; }
cmake --build build -j 4 --target kapitsa ni_run > "$SP/base-build.log" 2>&1
echo "exit $?" >> "$SP/base-build.log"
