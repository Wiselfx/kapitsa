#!/usr/bin/env python3
"""The network as the converter sees it: one stream and parallel streams of GGUF ranges (MB/s), small row reads of
the pack's n-gram table (requests/s) at 64 and 128 connections."""
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/agent-a9a95c76c733c6cab/tools")
import qwen4_exp_pack as P  # noqa: E402
import qwen4_exp_q20 as Q  # noqa: E402

grepo, grev, gpath = Q.parse_repo(None, Q.GGUF_REPO)
g = P.Remote(grepo, grev)
base = 3 << 30  # somewhere in the expert data
t = time.time()
n = 0
for i in range(3):
    n += len(g.read(gpath, base + i * (16 << 20), base + (i + 1) * (16 << 20)))
dt = time.time() - t
print(f"GGUF, one stream: {n / 1e6:.0f} MB in {dt:.1f} s = {n / 1e6 / dt:.1f} MB/s", flush=True)
for conns, piece in ((16, 16 << 20), (32, 8 << 20)):
    jobs = [(base + (64 << 20) + i * piece, base + (64 << 20) + (i + 1) * piece) for i in range(conns)]
    t = time.time()
    with ThreadPoolExecutor(conns) as ex:
        n = sum(len(x) for x in ex.map(lambda ab: g.read(gpath, ab[0], ab[1]), jobs))
    dt = time.time() - t
    print(f"GGUF, {conns} streams of {piece >> 20} MB: {n / 1e6:.0f} MB in {dt:.1f} s = {n / 1e6 / dt:.1f} MB/s",
          flush=True)
    base += 1 << 30

prepo, prev, _ = Q.parse_repo(None, Q.PACK_REPO)
pk = P.Remote(prepo, prev)
name = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}.weight"
rng = random.Random(1)
for conns, reqs in ((64, 640), (128, 1280)):
    jobs = []
    for _ in range(reqs):
        f, off, nb, dt_, shape = pk.index[name.format(rng.randrange(128))]
        r = rng.randrange(shape[0])
        jobs.append((f, off + r * 80, off + (r + 1) * 80))
    t = time.time()
    with ThreadPoolExecutor(conns) as ex:
        n = sum(len(x) for x in ex.map(lambda j: pk.read_small(j[0], j[1], j[2]), jobs))
    dt = time.time() - t
    print(f"table rows, {conns} connections: {reqs} reads in {dt:.1f} s = {reqs / dt:.0f} reads/s", flush=True)
