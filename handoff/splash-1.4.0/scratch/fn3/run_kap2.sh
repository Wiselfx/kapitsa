#!/bin/zsh
# run_kap2.sh: after k-c4096-r2 ends, a second MTP round and a second plain round (light warm-up, seed 11), each its own hold
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
end=$(( $(date +%s) + 1500 ))
until grep -q "k-c4096-r2" $Q/holds.log && [ -f $Q/runs/k-c4096-r2/t_end ] && [ $(cat $Q/last_hold_end) -ge $(cat $Q/runs/k-c4096-r2/t_end) ] || [ $(date +%s) -ge $end ]; do sleep 5; done
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh kapitsa k-mtp-r2 "11|--decode-runs 3 --greedy-runs 3 --skip prefill,tools,agent --warmup light"
SPEC=0 /bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh kapitsa k-plain-r2 "11|--decode-runs 3 --greedy-runs 3 --skip prefill,tools,agent --warmup light"
echo "run_kap2 done $(date '+%H:%M:%S')" >> $Q/holds.log
