#!/usr/bin/env python3
"""One chat request to a local Kapitsa server; prints the client's time to the first streamed token and the total.

  ttft_client.py PORT CORPUS_BYTES MAX_TOKENS TAG

The prompt is the first CORPUS_BYTES bytes of bench/corpus.txt (read only) and a fixed question, greedy.
"""
import json
import sys
import time
import urllib.request

port, nbytes, max_tokens, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
corpus = open("/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a1bca8900e5443872/bench/corpus.txt",
              encoding="utf-8").read()[:nbytes]
body = {
    "model": "m",
    "messages": [{"role": "user", "content": corpus + "\n\nSummarize the code above in three sentences."}],
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
first = None
n = 0
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        try:
            d = json.loads(line[5:])
        except ValueError:
            continue
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            if delta.get("content") or delta.get("reasoning_content"):
                n += 1
                if first is None:
                    first = time.perf_counter() - t0
t1 = time.perf_counter() - t0
print(f"client {tag}: first token {first * 1000 if first else -1:.1f} ms, total {t1 * 1000:.1f} ms, {n} chunks")
