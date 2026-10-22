#!/bin/zsh
# run_mix.sh STEP...: Splash and Strata rounds interleaved, each its own GPU hold (launch.sh: 570 s deadline, 120 s
# between holds). The disk holds Strata's pack or Splash's draft, not both: files.sh switches them between holds.
# Steps: t-dec t-dec-sfx t-pf-a t-pf-b t-pf4096-a t-pf4096-b s-dec s-pf-a s-pf-b (a suffix -rN repeats one)
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
ST=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata
W_A="999|--warmup full --adaptive-warm --skip decode,greedy,prefill,tools,agent"
W_F="999|--warmup full --skip decode,greedy,prefill,tools,agent"
DEC="11|--decode-runs 3 --greedy-runs 3 --skip prefill,tools,agent --warmup light"
PF_A=("11+6|--sizes 4000 --skip decode,greedy,tools,agent --warmup light" "33|--sizes 16000 --skip decode,greedy,tools,agent --warmup light")
PF_B=("44|--sizes 4000,16000 --skip decode,greedy,tools,agent --warmup light")
need() {  # pack | draft
  if [ $1 = pack ]; then [ -d $HOME/.cache/flash-next-gguf/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2 ] && /bin/zsh $Q/files.sh nodraft
    /bin/zsh $Q/files.sh pack
  else [ -d /private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata-data/packs/q2_0 ] && /bin/zsh $Q/files.sh nopack
    /bin/zsh $Q/files.sh draft
  fi
}
for step in "$@"; do
  base=${step%-r[0-9]*}
  case $base in
    t-dec)      need pack;  STRATA_CFG=$Q/strata-q2_0.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_A" "$DEC" ;;
    t-dec-sfx)  need pack;  STRATA_CFG=$Q/strata-q2_0-sfx.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_A" "$DEC" ;;
    t-pf-a)     need pack;  STRATA_CFG=$Q/strata-q2_0.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_F" "${PF_A[@]}" ;;
    t-pf-b)     need pack;  STRATA_CFG=$Q/strata-q2_0.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_F" "${PF_B[@]}" ;;
    t-pf4096-a) need pack;  STRATA_CFG=$Q/strata-q2_0-c4096.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_F" "${PF_A[@]}" ;;
    t-pf4096-b) need pack;  STRATA_CFG=$Q/strata-q2_0-c4096.json DEADLINE_PKILL=$ST/build-metal/strata /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh strata $step "$W_F" "${PF_B[@]}" ;;
    s-dec)      need draft; /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh splash $step "$W_A" "$DEC" ;;
    s-pf-a)     need draft; /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh splash $step "$W_F" "${PF_A[@]}" ;;
    s-pf-b)     need draft; /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh splash $step "$W_F" "${PF_B[@]}" ;;
    *) echo "unknown step $step" >> $Q/holds.log ;;
  esac
  echo "$(date '+%H:%M:%S') run_mix: $step finished" >> $Q/holds.log
done
echo "run_mix done $(date '+%H:%M:%S')" >> $Q/holds.log
