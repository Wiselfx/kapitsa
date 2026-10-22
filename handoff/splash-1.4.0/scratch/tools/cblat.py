#!/usr/bin/env python3
"""Summarizes an NI_CB_LAT file: "n commit start end" per command buffer (ms, host clock).

  cblat.py FILE [--skip N] [--from-ms T]

start-commit: the command buffer's wait from its commit. start-ready: its wait with nothing in front of it on the
queue (ready = the later of its commit and the end of every command buffer committed before it). Also the
start-to-start interval of consecutive command buffers (per-step time when each step is one command buffer).
"""
import sys


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def summarize(name, v):
    th = (1, 5, 20, 50, 100)
    over = [sum(1 for x in v if x > t) for t in th]
    return (f"{name} n={len(v)} p50 {pct(v, .5):.3f} p90 {pct(v, .9):.3f} p99 {pct(v, .99):.3f} "
            f"p99.9 {pct(v, .999):.3f} max {max(v) if v else float('nan'):.2f} ms; >1/5/20/50/100 ms: "
            + "/".join(str(o) for o in over))


def main():
    path = sys.argv[1]
    skip = 0
    if "--skip" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--skip") + 1])
    recs = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            recs.append((int(p[0]), float(p[1]), float(p[2]), float(p[3])))
    recs.sort()
    recs = recs[skip:]
    if "--from-ms" in sys.argv:
        t = float(sys.argv[sys.argv.index("--from-ms") + 1])
        recs = [r for r in recs if r[1] >= t]
    if "--to-ms" in sys.argv:
        t = float(sys.argv[sys.argv.index("--to-ms") + 1])
        recs = [r for r in recs if r[1] < t]
    from_commit, from_ready, s2s, gpu = [], [], [], []
    prev_end, prev_start = None, None
    worst = []
    for n, c, s, e in recs:
        if s <= 0:  # (no GPU times: an error or an empty command buffer)
            continue
        from_commit.append(s - c)
        ready = c if prev_end is None else max(c, prev_end)
        w = max(0.0, s - ready)
        from_ready.append(w)
        worst.append((w, n))
        gpu.append(e - s)
        if prev_start is not None:
            s2s.append(s - prev_start)
        prev_start = s
        prev_end = e if prev_end is None else max(prev_end, e)
    print(summarize("start-commit", from_commit))
    print(summarize("start-ready ", from_ready))
    print(summarize("start-start ", s2s))
    print(summarize("gpu         ", gpu))
    worst.sort(reverse=True)
    print("largest start-ready waits (ms@n):", " ".join(f"{w:.2f}@{n}" for w, n in worst[:8]))


if __name__ == "__main__":
    main()
