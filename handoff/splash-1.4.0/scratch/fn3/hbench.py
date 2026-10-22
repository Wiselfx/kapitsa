#!/usr/bin/env python3
"""bench/harness/bench.py, made repeatable: Python's random seeded (FNQ_SEED), so the run tags and session nonces are
the same in every invocation with the same arguments, and every request body logged with its reply (FNQ_LOG, JSONL),
so the sparse n-gram table can be given the rows those prompts and answers read. Arguments are bench.py's."""
import json
import os
import random
import sys
import time

HARNESS = os.environ.get("FNQ_HARNESS", "/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-splash-1-4-0-analysis-3bbeaf/4668b80e-9efe-498c-aead-31aa6884d51c/scratchpad/kap-main/bench/harness")
sys.path.insert(0, HARNESS)
import bench  # noqa: E402

random.seed(int(os.environ.get("FNQ_SEED", "1")))
# FNQ_SKIP_DRAWS=N: N random.random() draws first, as N skipped decode runs would take (each decode prompt draws one),
# so a prefill-only invocation gets the nonces of the full one with the same seed
for _ in range(int(os.environ.get("FNQ_SKIP_DRAWS", "0"))):
    random.random()
LOG = os.environ.get("FNQ_LOG")
_orig = bench.stream_chat


def dry(base, body, session=None, timeout=1800):
    """FNQ_DRY=1: no server; an empty answer (so no follow-ups), to log the first turns' bodies"""
    return {"ttft": 0.1, "gen_time": 0.1, "total": 0.2, "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "timings": None, "content": "", "reasoning": "", "tool_calls": [], "n_chunks": 0, "finish": "stop",
            "ev_t": [], "ev_c": [], "ev_k": []}


if os.environ.get("FNQ_DRY"):
    _orig = dry


def logged(base, body, session=None, timeout=1800):
    r = _orig(base, body, session=session, timeout=timeout)
    if LOG:
        ct = (r.get("usage") or {}).get("completion_tokens")
        try:
            mid, e2e, _ = bench.decode_rates(r, ct, 16, 4)
        except Exception:
            mid = e2e = None
        with open(LOG, "a") as f:
            f.write(json.dumps({"body": body, "reply": r.get("content"), "reasoning": r.get("reasoning"),
                                "usage": r.get("usage"), "ttft": r.get("ttft"), "total": r.get("total"),
                                "finish": r.get("finish"), "decode_tps": mid, "decode_tps_e2e": e2e,
                                "t_end": time.time()}) + "\n")
    return r


bench.stream_chat = logged
sys.argv = ["bench.py"] + sys.argv[1:]
bench.main()
