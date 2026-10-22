#!/usr/bin/env python3
"""strata_cfg.py OUT.json [SUFFIX_DRAFT] [extra engine args ...]: Strata's run config for this Mac as metal/setup_mac.py
writes it for a 48 GB Mac (strata_control/mac_fit.engine_args with stream=True: no MTP draft layer, --spec 2,
--suffix-draft 0, --expert-cache 64, --prefill auto, --max-context 32768, --mmap-experts; env STRATA_METAL_STREAM=1),
except SUFFIX_DRAFT (default 0, setup's value for a streaming Mac) and the log path."""
import json, sys
ST = "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata"
G = "/Users/flaminis/.cache/flash-next-gguf/hub/models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF/snapshots/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0"
PACK = "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/strata-src/Strata-data/packs/q2_0"
out = sys.argv[1]
sfx = sys.argv[2] if len(sys.argv) > 2 else "0"
extra = sys.argv[3:]
args = ["--pack", PACK, "--native", G + "/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf",
        "--ple-gguf", G + "/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf",
        "--expert-profile", ST + "/data/expert-profile.bin", "--expert-cache", "64",
        "--prefill", "auto", "--spec", "2", "--suffix-draft", sfx, *extra,
        "--max-context", "32768", "--mmap-experts"]
cfg = {"exe": ST + "/build-metal/strata", "args": args, "cwd": ST, "tokenizer": PACK + "/tokenizer",
       "model_name": "qwen3.8-flash-next-q2_0", "log": out.replace(".json", ".log"), "lib_dirs": [], "port": 8180,
       "backend": "metal", "env": {"STRATA_METAL_STREAM": "1"}, "open_browser": False}
json.dump(cfg, open(out, "w"), indent=1)
print(json.dumps(cfg, indent=1))
