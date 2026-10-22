#!/bin/zsh
# cover.sh TAG CONNS RUN_DIR ...: the token ids of every request the runs logged (prompt + reply) and their n-gram rows
# added to the pack's sparse table (rows already there skipped), with a disk guard (stops below 11 GB free).
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
M=$HOME/.cache/kapitsa/models/qwen3.8-flash-next-q20
TAG=$1; CONNS=$2; shift 2
mkdir -p $Q/cover
logs=()
for d in "$@"; do logs+=($d/requests.jsonl); done
python3 $Q/ids.py $Q/tok $Q/cover/$TAG.ids $logs || exit 1
echo "$(date '+%H:%M:%S') cover $TAG: before" ; python3 $Q/coverage.py $M $Q/cover/$TAG.ids $Q/cover/$TAG.ids.prompts | tail -1
python3 -u $Q/rows_file.py $M/ple.safetensors $CONNS @$Q/cover/$TAG.ids &
RP=$!
while kill -0 $RP 2>/dev/null; do
  f=$(df -g /System/Volumes/Data | tail -1 | awk '{print $4}')
  if [ "$f" -lt 11 ]; then echo "disk guard: $f GB free, stopping the row fetch"; kill -TERM $RP; break; fi
  sleep 10
done
wait $RP
echo "$(date '+%H:%M:%S') cover $TAG: after (exit $?)"; python3 $Q/coverage.py $M $Q/cover/$TAG.ids $Q/cover/$TAG.ids.prompts | tail -1
