#!/usr/bin/env python3
"""The pack's tokenizer and chat template (small files) into DIR, for prompt ids before the conversion ends."""
import os
import sys

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab/tools")
import qwen4_exp_pack as P  # noqa: E402
import qwen4_exp_q20 as Q  # noqa: E402

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
repo, rev, _ = Q.parse_repo(None, Q.PACK_REPO)
r = P.Remote(repo, rev)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"):
    if f in r.siblings:
        b = r.read(f)
        open(os.path.join(out, f), "wb").write(b)
        print(f, len(b))
