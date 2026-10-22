#!/bin/zsh
# guard.sh: the hard disk floor for the GGUF phase. Every 2 s the free space of /; below 3 GiB: stops the chain
# (run_mix, launch), every round's process group (servers and harness), Strata's engine and server, Splash's server and
# the downloads at once, deletes the GGUF folder, writes ABORT and exits 2. Exits 0 when STOP_GUARD exists.
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
GG=$HOME/.cache/flash-next-gguf/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF
log() { echo "$(date '+%H:%M:%S') guard: $*" >> $Q/guard.log; }
log "started (floor 3 GiB)"
while [ ! -f $Q/STOP_GUARD ]; do
  f=$(df -k / | tail -1 | awk '{print $4}')
  if [ "$f" -lt 3145728 ]; then
    log "free $(( f / 1024 )) MB < 3 GiB: stopping everything, deleting the GGUF"
    touch $Q/ABORT
    for sig in TERM KILL; do
      pkill -$sig -f "$Q/run_mix.sh"; pkill -$sig -f "$Q/launch.sh"; pkill -$sig -f "$Q/files.sh"
      for p in $(pgrep -f "$Q/round.sh"); do pg=$(ps -o pgid= -p $p | tr -d ' '); [ -n "$pg" ] && kill -$sig -- -$pg; done
      pkill -$sig -f "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata/build-metal/strata"; pkill -$sig -f "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata/serve/server.py"
      pkill -$sig -f "launcher.py serve --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 --revision ed59f92082b1e93c0e96d60a8b11aab089b52f09 --draft-model /Users/flaminis/.cache/flash-next-gguf"; pkill -$sig -f "$Q/fetch.py"; pkill -$sig -f "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata/tools/iq_pack.py"
      [ $sig = TERM ] && sleep 3
    done
    rm -rf $GG
    log "done: $(df -k / | tail -1 | awk '{printf "%.2f GiB", $4 / 1048576}') free; left: $(pgrep -fl 'round.sh|build-metal/strata|serve/server.py|ISTA-DASLab|fetch.py' | tr '\n' ' ')"
    exit 2
  fi
  sleep 2
done
log "stopped (STOP_GUARD)"
exit 0
