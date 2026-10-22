#!/bin/bash
# origin/main 92da1cd (exported to main-src): the CLI alone, for the baseline replies
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
cmake -S "$S/main-src" -B "$S/build-main" -DCMAKE_BUILD_TYPE=Release > "$S/main-cmake.log" 2>&1 || { echo "cmake failed"; exit 1; }
cmake --build "$S/build-main" -j 4 --target kapitsa > "$S/main-build.log" 2>&1
echo "main build exit=$? $(ls -la "$S/build-main/kapitsa" 2>&1 | cut -c1-80)"
