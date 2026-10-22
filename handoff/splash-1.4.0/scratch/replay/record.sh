#!/bin/bash
# usage: record.sh <model_dir> <tag> <n_prompts>: lattice recordings of the coding prompt variants (greedy, 512 tokens)
SCR=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad
M="$1"; TAG="$2"; N="${3:-6}"
for i in $(seq 0 $((N-1))); do
  NI_DFLASH_LAT=$SCR/replay/${TAG}_$i.lat NI_DFLASH_STEP1=1 $SCR/build-a/ni_run "$M" @$SCR/replay/code_$i.ids 512 0 --dflash auto --ignore-eos > $SCR/replay/${TAG}_$i.log 2>&1
  echo "rec $TAG $i exit $? $(ls -la $SCR/replay/${TAG}_$i.lat | awk '{print $5}') bytes"
done
