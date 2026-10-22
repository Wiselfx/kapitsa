#!/usr/bin/env python3
"""Collects concurrency.py streams results from round directories: per (build, n, temperature) every run's total tok/s
(and decode-window tok/s), with the round's foreign GPU load (gpu_hogs monitor) during each build's window.

  collect.py ROUND_DIR [ROUND_DIR ...]
"""
import json
import os
import sys
from collections import defaultdict


def foreign(rd, b):
    marks, start, end = {}, None, None
    try:
        for line in open(os.path.join(rd, "hogs_marks.txt")):
            w = line.split()
            if w[1] == b:
                if w[0] == "start":
                    start = float(w[2])
                elif w[0] == "end":
                    end = float(w[2])
    except OSError:
        return None
    vals = []
    try:
        for line in open(os.path.join(rd, "hogs.txt")):
            w = line.split()
            if len(w) >= 2 and start and end and start <= float(w[0]) <= end:
                vals.append((float(w[1]), " ".join(w[2:])))
    except OSError:
        return None
    if not vals:
        return None
    mx = max(vals, key=lambda v: v[0])
    return sum(v[0] for v in vals) / len(vals), mx[0], mx[1]


def main():
    res = defaultdict(list)
    for rd in sys.argv[1:]:
        for b in ("a", "b"):
            p = os.path.join(rd, f"client_{b}.txt")
            if not os.path.exists(p):
                continue
            fg = foreign(rd, b)
            for line in open(p):
                line = line.strip()
                if not line.startswith("{"):
                    continue
                j = json.loads(line)
                if "total_tps" not in j or "n" not in j:
                    continue
                res[(j["n"], j["temperature"], b)].append(
                    (os.path.basename(rd.rstrip("/")), j["total_tps"], j["decode_window_tps"], sum(j["tokens"]),
                     len(j["errors"]), fg))
    keys = sorted({(n, t) for (n, t, _) in res})
    for n, t in keys:
        line = f"n={n} T={t}:"
        means = {}
        for b in ("a", "b"):
            runs = res.get((n, t, b), [])
            if not runs:
                continue
            v = [r[1] for r in runs]
            means[b] = sum(v) / len(v)
            line += f"  {b.upper()} " + " ".join(f"{x:.1f}" for x in v) + f" (mean {means[b]:.1f}"
            line += f", range {min(v):.1f}-{max(v):.1f}, {len(v)} runs)"
            errs = sum(r[4] for r in runs)
            if errs:
                line += f" ERRORS {errs}"
        if "a" in means and "b" in means:
            line += f"  B/A {100 * means['b'] / means['a'] - 100:+.1f}%"
        print(line)
    print()
    for (n, t, b), runs in sorted(res.items()):
        for r in runs:
            fg = r[5]
            print(f"  {r[0]:22s} n={n} T={t} {b}: total {r[1]:.2f} tok/s, decode window {r[2]:.2f}, tokens {r[3]}, "
                  f"errors {r[4]}, foreign GPU " + (f"mean {fg[0]:.1f}% max {fg[1]:.1f}% ({fg[2]})" if fg else "-"))


if __name__ == "__main__":
    main()
