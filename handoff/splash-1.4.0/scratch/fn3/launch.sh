#!/bin/zsh
# launch.sh LIMIT_S cmd ...: at least 120 s after this agent's previous hold ended (wall clock), takes the GPU lock
# (caffeinate + lockf, waiting up to an hour) and runs cmd under deadline.py (TERM to its process group at LIMIT_S,
# KILL 15 s later). Records when the hold ended.
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
LIMIT=$1; shift
LAST=$Q/last_hold_end
if [ -f $LAST ]; then
  until [ $(( $(date +%s) - $(cat $LAST) )) -ge 120 ]; do sleep 5; done
fi
echo "$(date '+%H:%M:%S') launch: waiting for the GPU lock: $*" >> $Q/holds.log
caffeinate -i -s lockf -k -t 3600 $HOME/.cache/nace-gpu.lock python3 $Q/deadline.py $LIMIT $Q/holds.log "$@"
rc=$?
date '+%s' > $LAST
echo "$(date '+%H:%M:%S') launch: lock released, exit $rc" >> $Q/holds.log
exit $rc
