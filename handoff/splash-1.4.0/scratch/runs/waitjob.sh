#!/bin/bash
# Waits (no polling) until the background GPU job <name> has ended: its private lock (taken by gpujob.sh for the job's
# whole run) becomes free, or <secs> pass (exit 75).
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
lockf -s -t "${2:-540}" "$SCR/runs/$1.plock" true
