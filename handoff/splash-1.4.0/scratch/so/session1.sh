#!/bin/bash
S=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/so
caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" bash "$S/session1_inner.sh"
