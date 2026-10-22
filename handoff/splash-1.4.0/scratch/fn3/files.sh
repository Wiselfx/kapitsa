#!/bin/zsh
# files.sh pack|nopack|draft|nodraft: the disk holds either Strata's pack (1.5 GB) or Splash's DFlash2 draft (1.2 GB), not
# both (the GGUF leaves ~12 GiB): makes or removes one, with a line in files.log.
Q=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/fn3
ST=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata
PACK=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata-data/packs/q2_0
G=$HOME/.cache/flash-next-gguf/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0
D=$HOME/.cache/flash-next-gguf/hub/models--incoai--Qwen3.8-Flash-Next-DFlash2
free() { df -k / | tail -1 | awk '{printf "%.2f GiB", $4 / 1048576}'; }
t0=$(date +%s)
case $1 in
  pack)
    if [ $(df -k / | tail -1 | awk '{print $4}') -lt 5242880 ]; then
      echo "$(date '+%H:%M:%S') pack: under 5 GiB free ($(free)), not written" >> $Q/files.log; exit 3; fi
    if [ ! -f $PACK/native_experts.txt ] || [ ! -f $PACK/tokenizer/vocab.json ]; then
      mkdir -p $(dirname $PACK)
      STRATA_GGUF_PY=$ST/third_party/llama.cpp/gguf-py nice -n 5 $ST/.venv/bin/python $ST/tools/iq_pack.py \
        --gguf $G/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --out $PACK >> $Q/files-pack.log 2>&1
      rc=$?
    else rc=0; fi
    echo "$(date '+%H:%M:%S') pack: exit $rc in $(( $(date +%s) - t0 )) s, $(du -sh $PACK 2>/dev/null | cut -f1), $(free) free" >> $Q/files.log ;;
  nopack)
    rm -rf $PACK
    echo "$(date '+%H:%M:%S') pack removed, $(free) free" >> $Q/files.log ;;
  draft)
    python3 $Q/fetch.py $HOME/.cache/flash-next-gguf/hub incoai/Qwen3.8-Flash-Next-DFlash2 a22a1744e85c17364dcc54a6a420af7135d4c7f6 \
      config.json model.safetensors --workers 16 --min-free-gb 3.5 >> $Q/files-draft.log 2>&1
    rc=$?
    echo "$(date '+%H:%M:%S') draft: exit $rc in $(( $(date +%s) - t0 )) s, $(free) free" >> $Q/files.log ;;
  nodraft)
    rm -rf $D
    echo "$(date '+%H:%M:%S') draft removed, $(free) free" >> $Q/files.log ;;
esac
exit ${rc:-0}
