#!/usr/bin/env python3
"""rows_file.py TABLE CONNS [--stream] @ids ...: the rows the token sequences read, into a sparse copy of the pack's
n-gram table at TABLE (created when missing; rows already there are skipped), from the pinned remote pack."""
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab/tools")
import qwen4_exp_q20 as Q  # noqa: E402

table, conns = sys.argv[1], int(sys.argv[2])
stream = "--stream" in sys.argv[3:]
ids = [x for x in sys.argv[3:] if x.startswith("@")]
t0 = time.time()
rows = Q.sparse_rows(ids)
with ThreadPoolExecutor(conns) as pool:
    pack = Q.PackSource(types.SimpleNamespace(pack=None, pack_repo=Q.PACK_REPO), pool)
    size, n, wrote = Q.write_sparse_table(pack, table, rows, add=os.path.exists(table), stream=stream)
print(f"{table}: {len(rows)} rows read by the sequences, {wrote} written, {time.time() - t0:.0f} s", flush=True)
