#!/bin/bash
# usage: gpuhold.sh <secs> <command...>: one GPU lock hold with a hard deadline (the command's process group is stopped
# at <secs>; the lock is released when it ends)
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
secs="$1"; shift
exec caffeinate -i -s lockf -k -t 3600 "$HOME/.cache/nace-gpu.lock" "$SCR/runs/deadline.pl" "$secs" "$@"
