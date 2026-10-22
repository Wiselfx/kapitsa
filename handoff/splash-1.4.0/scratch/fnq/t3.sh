#!/bin/zsh
# t3.sh: the 16K prefill at the server's default 2048-row chunks, then the 4K and 16K prefills with 4096-row chunks
# (run under lockf and deadline.py)
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fnq
/bin/zsh $Q/hold.sh t3-pf16 2 33 12 23 --sizes 16000 --skip decode,greedy,tools,agent --warmup light
SERVE_EXTRA="--prefill-chunk 4096" /bin/zsh $Q/hold.sh t3-c4096 2 44 12 23 --sizes 4000,16000 --skip decode,greedy,tools,agent --warmup light
