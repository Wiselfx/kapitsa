#!/bin/zsh
# run_splash.sh SUFFIX: Splash 1.4.0's rounds, each its own GPU hold (launch.sh: 570 s deadline, 120 s between holds):
#   s-dec: the harness's adaptive warm-up (seed 999: a 2.9K prefill, a tools turn, three 512-token decodes), then seed 11's
#          decode runs (3 at T=0.6, 3 greedy: this morning's prompts and nonces)
#   s-pf:  the full warm-up (seed 999), then the 4K prompt of seed 11 (its nonce after the 6 decode draws), the 16K of
#          seed 33 and the 4K and 16K of seed 44, each with its 3 follow-ups
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
W_A="999|--warmup full --adaptive-warm --skip decode,greedy,prefill,tools,agent"
W_F="999|--warmup full --skip decode,greedy,prefill,tools,agent"
DEC="11|--decode-runs 3 --greedy-runs 3 --skip prefill,tools,agent --warmup light"
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh splash s-dec$1 "$W_A" "$DEC"
/bin/zsh $Q/launch.sh 570 /bin/zsh $Q/round.sh splash s-pf$1 "$W_F" "11+6|--sizes 4000 --skip decode,greedy,tools,agent --warmup light" "33|--sizes 16000 --skip decode,greedy,tools,agent --warmup light" "44|--sizes 4000,16000 --skip decode,greedy,tools,agent --warmup light"
echo "run_splash$1 done $(date '+%H:%M:%S')" >> $Q/holds.log
