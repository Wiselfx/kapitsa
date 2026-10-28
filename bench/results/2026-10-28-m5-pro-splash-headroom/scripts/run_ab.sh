#!/bin/bash
# Driver for issue #753. Eight legs, four per arm, interleaved A14 A10 | A10 A14 | A14 A10 | A10 A14, one Splash start per leg under the
# GPU lock. Legs that already have a JSON are skipped, so a stopped run resumes. Stops (does not go on) when a leg ends 3 (memory guard),
# 4 (Splash refused or exited), 5 (an A10 leg broke a safety bound) or 6 (not on AC power, at the start or the end of a leg): the verdict is then already KEEP 14 GiB or the Mac needs a look.
# Needs: the orchestrator's GO naming this folder's frozen commit. Do not run it before that.
D=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$D/legs}
PAIRS=${PAIRS:-4}
REPS=${REPS:-3}
H="python3 $D/scripts/headroom_ab.py"
LOCK=${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}
mkdir -p "$OUT"
$H preflight || exit 1
leg() { # idx arm pair
  [ -f "$OUT/leg$1-$2.json" ] && return 0
  echo "=== leg $1 $2 pair $3 $(date -u +%FT%TZ)"
  local rc
  for try in $(seq 1 40); do  # A below 31 GiB, or another worker's model start: wait a minute and ask again (40 tries, about 40 min)
    nice -n 5 lockf -k -t 120 "$LOCK" $H leg --arm "$2" --pair "$3" --idx "$1" --out "$OUT" --reps "$REPS" > "$OUT/leg$1-$2.out" 2>&1
    rc=$?
    [ $rc -eq 75 ] || break
    echo "  not started (try $try): $(tail -1 "$OUT/leg$1-$2.out")"; sleep 60
  done
  echo "leg $1 $2 rc=$rc $(date -u +%FT%TZ)"; tail -2 "$OUT/leg$1-$2.out"
  case $rc in 0) ;; 3|4|5|6) echo "stopping: rc=$rc"; $H summarize "$OUT"; exit $rc ;; *) echo "stopping: rc=$rc (not started in 40 tries, or an error)"; exit $rc ;; esac
}
for ((p = 0; p < PAIRS; p++)); do
  if ((p % 2 == 0)); then first=A14 second=A10; else first=A10 second=A14; fi
  leg $((2 * p)) $first $p
  leg $((2 * p + 1)) $second $p
done
$H summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
