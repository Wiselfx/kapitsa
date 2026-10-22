#!/bin/bash
# One measured round under the GPU lock: for each build in ORDER, a server (serve_run.sh) and the client commands.
# usage: round.sh <tag> <model_dir> "<order: a b>" <client spec file>  (each line of the spec: a client command, BASE = URL)
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
WT=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/splash-1-4-0-analysis-3bbeaf
TAG="$1"; MODEL="$2"; ORDER="$3"; SPEC="$4"
OUT=$SCR/runs/$TAG; mkdir -p $OUT
python3 $WT/bench/harness/gpu_hogs.py monitor $OUT/hogs.txt $$ &
MON=$!
for b in $ORDER; do
  {
    echo "### build $b $(date '+%H:%M:%S')"
    uptime
    vm_stat | grep -E "Pages free|Pages active|Pages wired|Pageouts|Swapouts|compressor"
    memory_pressure 2>/dev/null | grep -E "System-wide memory free percentage" 
  } >> $OUT/sys.txt
  echo "start $b $(date +%s)" >> $OUT/hogs_marks.txt
  args=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    [ ${#args[@]} -gt 0 ] && args+=("--")
    line=${line//BUILD/$b}
    line=${line//OUTDIR/$OUT}
    # shellcheck disable=SC2206
    args+=($line)
  done < "$SPEC"
  if [ -n "${TRACE:-}" ]; then export NI_TRACE_TOKENS=$OUT/trace_$b.jsonl; fi
  # variants of build B: f = a floor of one draft per member, t = the fixed table (NI_BATCH_KF=0)
  bin=$SCR/build-$b/kapitsa
  unset NI_BATCH_KF_FLOOR NI_BATCH_KF
  if [ "$b" = f ]; then bin=$SCR/build-b/kapitsa; export NI_BATCH_KF_FLOOR=1; fi
  if [ "$b" = t ]; then bin=$SCR/build-b/kapitsa; export NI_BATCH_KF=0; fi
  # x, y: build B with another home (x: a copy of B's KFlash caches, y: of A's): the single-stream budget's state apart
  if [ "$b" = x ] || [ "$b" = y ] || [ "$b" = v ]; then bin=$SCR/build-b/kapitsa; fi
  KAPITSA_HOME=$SCR/home-$b KAPITSA_NO_AUTO_UPDATE=1 SLOTS=${SLOTS:-4} PORT=${PORT:-8137} NI_LOG=${NI_LOG:-2} \
    perl -e 'alarm shift; exec @ARGV' ${BUILD_TIMEOUT:-280} $WT/bench/batch/serve_run.sh $bin "$MODEL" $OUT/serve_$b.log "${args[@]}" > $OUT/client_$b.txt 2>&1
  echo "end $b $(date +%s) rc $?" >> $OUT/hogs_marks.txt
  echo "### build $b done $(date '+%H:%M:%S')" >> $OUT/sys.txt
done
kill $MON 2>/dev/null; wait $MON 2>/dev/null
echo ROUND_DONE
