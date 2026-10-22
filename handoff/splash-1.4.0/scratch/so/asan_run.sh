#!/bin/bash
# grammar_test (quick) under ASan and UBSan, halting on the first UBSan report
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 "$S/asan/grammar_test" --quick > "$S/asan/run2.log" 2>&1
echo "exit=$?"
tail -3 "$S/asan/run2.log"
echo "sanitizer reports: $(grep -c -E 'runtime error|ERROR: AddressSanitizer' "$S/asan/run2.log")"
