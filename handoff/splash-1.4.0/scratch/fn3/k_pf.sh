#!/bin/zsh
# k_pf.sh SUFFIX: the 16K prefill at the default 2048-row chunks (seed 33), then the 4K and 16K prefills with
# --prefill-chunk 4096 (seed 44); this morning's t3.sh prompts (run under launch.sh)
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
/bin/zsh $Q/kap_hold.sh k-pf16$1 2 33 --sizes 16000 --skip decode,greedy,tools,agent --warmup light
SERVE_EXTRA="--prefill-chunk 4096" /bin/zsh $Q/kap_hold.sh k-c4096$1 2 44 --sizes 4000,16000 --skip decode,greedy,tools,agent --warmup light
