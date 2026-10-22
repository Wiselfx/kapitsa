#!/bin/zsh
# run_kap.sh SUFFIX: the three Kapitsa rounds, each its own GPU hold (launch.sh: 570 s deadline, 120 s between holds)
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/kap_hold.sh k-mtp$1 2 11 --decode-runs 3 --greedy-runs 3 --sizes 4000 --skip tools,agent --warmup light
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/kap_hold.sh k-plain$1 0 11 --decode-runs 3 --greedy-runs 3 --skip prefill,tools,agent --warmup light
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/k_pf.sh $1
echo "run_kap$1 done $(date '+%H:%M:%S')" >> $Q/holds.log
