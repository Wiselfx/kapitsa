#!/bin/zsh
# A lock hold of ni_run decode A/B runs on one model: base (origin/main), new with NI_RESIDENCY=queue (the old
# behaviour, instrumented), new with the default (wired), in the order given.
#   hold_decode.sh <outdir> <model dir> <prompt file> <n_gen> <pressure 0|1> <order: e.g. "A Q W W Q A">
# With pressure=1 a memhog starts 3 s after the model has loaded (stops growing at the warning level or +256 MB
# swap) and is stopped when the run ends. Run under the GPU lock.
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
REPO=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872
OUT=$1; MODEL=$2; PROMPT=$3; NGEN=$4; PRESSURE=$5; ORDER=(${=6})
EXTRA=(${=7})
mkdir -p $OUT
state() {
  echo "---- state $(date +%T): $1"
  python3 $REPO/bench/harness/gpu_hogs.py 2 2>&1 | head -4
  vm_stat | grep -E "Pages (free|wired|stored in compressor|occupied by compressor)"
  sysctl -n vm.swapusage
  memory_pressure 2>/dev/null | tail -1
  uptime
}
echo "==== hold start $(date +%T): $MODEL n_gen $NGEN pressure $PRESSURE order $ORDER extra $EXTRA"
i=0
for lab in $ORDER; do
  i=$((i+1))
  state "before run $i ($lab)"
  case $lab in
    A) BIN=$SP/base/build/ni_run; ENVS=() ;;
    Q) BIN=$REPO/build/ni_run; ENVS=(NI_RESIDENCY=queue NI_CB_LAT=$OUT/$i-Q.cblat) ;;
    W) BIN=$REPO/build/ni_run; ENVS=(NI_CB_LAT=$OUT/$i-W.cblat) ;;
  esac
  rm -f $OUT/$i-$lab.cblat $OUT/$i-$lab.err $OUT/$i-$lab.hog
  t0=$(date +%s)
  env NI_MEM=1 $ENVS $BIN $MODEL @$PROMPT $NGEN 0 --ignore-eos $EXTRA > $OUT/$i-$lab.out 2> $OUT/$i-$lab.err &
  RUN=$!
  HOG=
  if [[ $PRESSURE == 1 ]]; then
    for k in {1..240}; do
      grep -q "loaded " $OUT/$i-$lab.err 2>/dev/null && break
      kill -0 $RUN 2>/dev/null || break
      /bin/sleep 0.5
    done
    /bin/sleep 3
    $SP/tools/memhog 12 300 256 512 2 2> $OUT/$i-$lab.hog &
    HOG=$!
  fi
  wait $RUN
  rc=$?
  [[ -n $HOG ]] && { kill $HOG 2>/dev/null; wait $HOG 2>/dev/null; }
  echo "==== run $i $lab rc $rc $(( $(date +%s) - t0 )) s"
  grep -E "loaded |decode |mean GPU|dflash decode|memory: Metal|cycle trace|drafting pass" $OUT/$i-$lab.err | cut -c1-400
  [[ -f $OUT/$i-$lab.hog ]] && grep -E "holding|done|swap grew" $OUT/$i-$lab.hog
  if [[ -f $OUT/$i-$lab.cblat ]]; then
    python3 $SP/tools/cblat.py $OUT/$i-$lab.cblat
    if [[ -f $OUT/$i-$lab.hog ]]; then
      HS=$(grep -o "memhog start: host_ms [0-9.]*" $OUT/$i-$lab.hog | awk '{print $4}')
      if [[ -n $HS ]]; then
        echo "  -- before the memhog started:"; python3 $SP/tools/cblat.py $OUT/$i-$lab.cblat --to-ms $HS | sed 's/^/  /'
        echo "  -- from the memhog's start:"; python3 $SP/tools/cblat.py $OUT/$i-$lab.cblat --from-ms $HS | sed 's/^/  /'
      fi
    fi
  fi
done
state "after"
echo "==== hold end $(date +%T)"
