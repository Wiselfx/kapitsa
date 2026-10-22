#!/bin/zsh
# conv.sh [converter arguments ...]: tools/qwen4_exp_q20.py with a disk guard: every 10 s the free space of the data
# volume is checked, and below 10 GB the converter is stopped (TERM, then KILL). Output to conv.log.
W=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
LOG=$Q/conv2.log
echo "$(date '+%H:%M:%S') conv: $* ($(df -g /System/Volumes/Data | tail -1 | awk '{print $4}') GB free)" >> $LOG
python3 -u $W/tools/qwen4_exp_q20.py "$@" >> $LOG 2>&1 &
CP=$!
while kill -0 $CP 2>/dev/null; do
  free=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  if [ "$free" -lt 10 ]; then
    echo "$(date '+%H:%M:%S') disk guard: $free GB free, stopping the converter" >> $LOG
    kill -TERM $CP
    sleep 5
    kill -KILL $CP 2>/dev/null
    break
  fi
  sleep 10
done
wait $CP
rc=$?
echo "$(date '+%H:%M:%S') conv: exit $rc ($(df -g /System/Volumes/Data | tail -1 | awk '{print $4}') GB free)" >> $LOG
exit $rc
