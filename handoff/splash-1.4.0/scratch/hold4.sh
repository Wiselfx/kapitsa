#!/bin/zsh
# Lock hold 4: 35B server TTFT: cold (first request after load), cached, cached after 10 s idle, cached after 20 s idle
# with a modest memhog. The new binary first (W, Q alternating: one Metal compiler cache entry per executable name, so
# the first start compiles), then the base binary twice (A).
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
echo "==== hold 4 start $(date +%T)"
zsh $SP/hold_server.sh $SP/r4 /Users/flaminis/.cache/kapitsa/models/qwen3.6-35b-a3b-4bit 10 20 "W Q W Q A A"
echo "==== hold 4 end $(date +%T)"
