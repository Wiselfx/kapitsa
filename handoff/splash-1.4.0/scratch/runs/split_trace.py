#!/usr/bin/env python3
"""Splits a server's NI_TRACE_TOKENS file of an exactness session (each prompt run alone first, then together) into
the first and the second occurrence of each prompt: SOLO and BATCHED files for bench/batch/match.py.

  split_trace.py TRACE.jsonl SOLO.jsonl BATCHED.jsonl
"""
import json
import sys

seen = {}
solo, bat = [], []
for line in open(sys.argv[1]):
    if not line.strip():
        continue
    j = json.loads(line)
    h = j["prompt_hash"]
    seen[h] = seen.get(h, 0) + 1
    (solo if seen[h] == 1 else bat if seen[h] == 2 else []).append(line)
open(sys.argv[2], "w").writelines(solo)
open(sys.argv[3], "w").writelines(bat)
print(f"{len(solo)} first, {len(bat)} second occurrences")
