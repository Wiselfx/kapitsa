#!/bin/zsh
# diskwait4.sh: every 5 minutes (wall clock) the free space of / in GiB; exits 0 at 65 GiB or more (the 35B's Trash
# emptied), 1 once the deadline in disk_deadline2 (4 hours from 11:10 on 2026-10-22) has passed. One line per check.
DL=$(cat /private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3/disk_deadline2)
while true; do
  g=$(df -k / | tail -1 | awk '{printf "%.2f", $4 / 1048576}')
  now=$(date +%s)
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 65 else 1)" $g; then echo "$(date '+%H:%M:%S') free $g GiB: enough"; exit 0; fi
  if [ $now -ge $DL ]; then echo "$(date '+%H:%M:%S') free $g GiB: gave up (4 h)"; exit 1; fi
  echo "$(date '+%H:%M:%S') free $g GiB, need 65; $(( (DL - now) / 60 )) min left"
  nxt=$(( now + 300 ))
  while [ $(date +%s) -lt $nxt ]; do sleep 10; done
done
