#!/bin/zsh
# diskwait2.sh: every 5 minutes (wall clock) the free space of /; exits 0 once the GGUF's 66.42 GB still to fetch would
# leave at least 11 GB (77.5 GB free: the 74 GiB mark less the 1.21 GB draft already fetched), 1 once the deadline in
# disk_deadline (4 hours after the pack was deleted) has passed. One line per check.
DL=$(cat /private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3/disk_deadline)
while true; do
  f=$(df -k / | tail -1 | awk '{printf "%d", $4 * 1024 / 1e6}')
  g=$(df -g / | tail -1 | awk '{print $4}')
  now=$(date +%s)
  if [ "$f" -ge 77500 ]; then echo "$(date '+%H:%M:%S') free $f MB ($g GiB): enough"; exit 0; fi
  if [ $now -ge $DL ]; then echo "$(date '+%H:%M:%S') free $f MB ($g GiB): gave up (4 h)"; exit 1; fi
  echo "$(date '+%H:%M:%S') free $f MB ($g GiB), need 77500 MB; $(( (DL - now) / 60 )) min left"
  nxt=$(( now + 300 ))
  while [ $(date +%s) -lt $nxt ]; do sleep 10; done
done
