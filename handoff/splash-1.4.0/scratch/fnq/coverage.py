#!/usr/bin/env python3
"""coverage.py MODEL_DIR|TABLE IDS [PROMPT_IDS]: how much of each sequence's n-gram table rows the sparse table holds.
Per sequence: tokens, positions whose 16 rows are all present, the first position missing one, and with PROMPT_IDS
(the same sequences' prompts) the reply's share. A row is present when its scales are nonzero (holes read zero)."""
import bisect
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab/tools")
import qwen4_exp_pack as P  # noqa: E402


def layout(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    shards = sorted({k.rsplit(".", 1)[0] for k in h if k != "__metadata__"}, key=lambda s: int(s.rsplit(".", 1)[1]))
    first, out = 0, []
    for s in shards:
        sc = h[s + ".scales"]
        nr = sc["shape"][0]
        out.append((first, nr, 8 + n + sc["data_offsets"][0], sc["shape"][1] * 2))
        first += nr
    return out


def main():
    model, ids_path = sys.argv[1], sys.argv[2]
    prompts = [len(x.split(",")) for x in open(sys.argv[3]) if x.strip()] if len(sys.argv) > 3 else None
    table = model if os.path.isfile(model) else os.path.join(model, "ple.safetensors")
    lay = layout(table)
    starts = [x[0] for x in lay]
    fd = os.open(table, os.O_RDONLY)
    cache = {}

    def present(r):
        if r not in cache:
            k = bisect.bisect_right(starts, r) - 1
            f0, nr, so, gb = lay[k]
            cache[r] = any(os.pread(fd, gb, so + (r - f0) * gb))
        return cache[r]
    tot_pos = tot_ok = rep_pos = rep_ok = 0
    for i, line in enumerate(open(ids_path)):
        line = line.strip()
        if not line:
            continue
        seq = [int(x) for x in line.split(",")]
        ids = P.ngram_ids(seq)
        ok = np.array([all(present(int(r)) for r in row) for row in ids])
        miss = np.flatnonzero(~ok)
        p = prompts[i] if prompts else None
        tot_pos += len(seq)
        tot_ok += int(ok.sum())
        msg = f"seq {i}: {len(seq)} tokens, {int(ok.sum())} with every row present"
        if p is not None:
            r_ok = int(ok[p:].sum())
            rep_pos += len(seq) - p
            rep_ok += r_ok
            first_rep = next((int(m) - p for m in miss if m >= p), None)
            msg += (f"; prompt {p}: {int(ok[:p].sum())} present; reply {len(seq) - p}: {r_ok} present, first missing at "
                    f"reply token {first_rep}")
        else:
            msg += f", first missing at {int(miss[0]) if len(miss) else None}"
        print(msg)
    print(f"all: {tot_ok}/{tot_pos} positions present" + (f"; replies {rep_ok}/{rep_pos}" if prompts else ""))


if __name__ == "__main__":
    main()
