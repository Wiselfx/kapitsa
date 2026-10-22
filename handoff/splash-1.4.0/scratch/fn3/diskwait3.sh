#!/bin/zsh
# diskwait3.sh: every 5 minutes (wall clock) the free space of / in GiB; exits 0 once it is at least 71.5 GiB (the Trash's
# ~7.4 GiB freed: then the 74 GiB mark is checked by hand), 1 once the deadline in disk_deadline (4 hours after the pack
# was deleted) has passed. One line per check.
DL=$(cat /private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3/disk_deadline)
while true; do
  g=$(df -k / | tail -1 | awk '{printf "%.2f", $4 / 1048576}')
  now=$(date +%s)
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 71.5 else 1)" $g; then echo "$(date '+%H:%M:%S') free $g GiB: the Trash was emptied"; exit 0; fi
  if [ $now -ge $DL ]; then echo "$(date '+%H:%M:%S') free $g GiB: gave up (4 h)"; exit 1; fi
  echo "$(date '+%H:%M:%S') free $g GiB, waiting for the Trash (74 GiB mark); $(( (DL - now) / 60 )) min left"
  nxt=$(( now + 300 ))
  while [ $(date +%s) -lt $nxt ]; do sleep 10; done
done
