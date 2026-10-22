#!/usr/bin/env python3
"""ids.py MODEL_DIR OUT_IDS LOG.jsonl ... : the token ids of every logged request (prompt + its reply re-encoded), one
sequence a line, for `qwen4_exp_q20.py rows DIR @OUT_IDS`. Also writes OUT_IDS.prompts (the prompts alone)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model, out, logs = sys.argv[1], sys.argv[2], sys.argv[3:]
    bodies = out + ".bodies.jsonl"
    n = 0
    with open(bodies, "w") as f, open(bodies + ".prompts", "w") as fp:
        for path in logs:
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                b = dict(rec["body"])
                fp.write(json.dumps(b) + "\n")
                if rec.get("reply"):
                    b["reply"] = rec["reply"]
                f.write(json.dumps(b) + "\n")
                n += 1
    tool = os.path.join(HERE, "prompt_ids")
    with open(out, "w") as fo:
        subprocess.check_call([tool, model, bodies], stdout=fo)
    with open(out + ".prompts", "w") as fo:
        subprocess.check_call([tool, model, bodies + ".prompts"], stdout=fo)
    lens = [len(x.split(",")) for x in open(out) if x.strip()]
    print(f"{n} requests -> {out}: {sum(lens)} tokens ({min(lens) if lens else 0}..{max(lens) if lens else 0} a sequence)")


if __name__ == "__main__":
    main()
