#!/bin/bash
# usage: gpujob.sh <name> <command...>: under a private lock (waitjob.sh waits on it), the GPU lock, caffeinate
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
name="$1"; shift
exec lockf -k "$SCR/runs/$name.plock" caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" "$@"
