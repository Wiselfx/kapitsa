#!/bin/bash
# queue.sh <queue file>: runs the queue's holds one after another (one line each: "NAME|ENV|MODEL|ORDER|SPEC"), each
# under the GPU lock with a hard 570 s deadline (gpuhold.sh), at least 125 s after the previous hold ended. Stops at a
# hold that hit its deadline or failed, or when the file STOP exists beside the queue.
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
Q="$1"
last=$(cat $SCR/runs/last_hold_end 2>/dev/null || echo 0)
while IFS='|' read -r name envs model order spec; do
  [ -z "$name" ] && continue
  [ -f $SCR/runs/STOP ] && { echo "STOP before $name"; exit 0; }
  t=$(( last + 125 ))
  while [ $(date +%s) -lt $t ]; do sleep 5; done
  echo "$name: queued $(date +%H:%M:%S)"
  env $envs BUILD_TIMEOUT=540 $SCR/runs/gpuhold.sh 570 $SCR/runs/round.sh "$name" "$model" "$order" "$spec" > $SCR/runs/$name.out 2>&1
  rc=$?
  last=$(date +%s); echo $last > $SCR/runs/last_hold_end
  echo "$name: done $(date +%H:%M:%S) rc $rc"
  if [ $rc -ne 0 ]; then echo "$name failed or hit its deadline: queue stopped"; exit 1; fi
done < "$Q"
echo "queue done"
