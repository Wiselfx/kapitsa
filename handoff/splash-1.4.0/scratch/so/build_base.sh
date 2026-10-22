#!/bin/bash
# The A/B baseline: origin/main (4e0154f), exported to so/base-src, built into so/build-base (kapitsa only)
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
cmake -S "$S/base-src" -B "$S/build-base" -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=4e0154f > "$S/base-cmake.log" 2>&1 || exit 1
cmake --build "$S/build-base" -j 4 --target kapitsa > "$S/base-build.log" 2>&1
echo "exit=$?"
ls -la "$S/build-base/kapitsa"
