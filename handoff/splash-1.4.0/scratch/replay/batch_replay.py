#!/usr/bin/env python3
"""Batched-decode row allocation replayed on ni_run lattice recordings (NI_DFLASH_LAT + NI_DFLASH_STEP1, greedy).

Each member of a batch is one recording (one prompt's plain greedy output with the drafter's lattice at every
position). A batched cycle drafts the walk's chain for every member at its anchor; the members' verify rows are
allocated by a policy; a member that verifies k drafts emits min(a7, k) + 1 tokens, a7 the chain's matched length at
full depth (exact at T = 0: the emitted tokens are the recorded output whatever is verified).

Policies: the fixed table (rows per member), the KFlash budget (sequential: rows in decreasing calibrated survival,
each kept while its survival pays rate x its cost increment; total <= 8), the same with a floor of one draft per
member, and the oracle (the best allocation of the same total rows given the true a7).

  batch_replay.py --model 35b|27b rec0.lat rec1.lat ...
"""
import argparse
import itertools
import math
import sys

import numpy as np

sys.path.insert(0, "/Users/flaminis/Desktop/kapitsa/.claude/worktrees/splash-1-4-0-analysis-3bbeaf/tools")
from tree_sim import Run  # noqa: E402


def chain(rec):
    """walk's argmax chain at this anchor: tokens and log q (softmax over the 16 candidates of the walk's scores)"""
    toks, lq = [], []
    for p in range(7):
        s = rec["ws"][p].astype(np.float64)
        k = int(rec["wk"][p])
        m = s.max()
        lse = m + math.log(np.exp(s - m).sum())
        toks.append(int(rec["ci"][p][k]))
        lq.append(float(s[k] - lse))
    return toks, lq


def logit_q(lq):
    q = min(math.exp(lq), 0.999999)
    return lq - math.log1p(-q)


def fit(x, y, iters=30):
    """KfBudget::fit: L2-regularized logistic regression toward (0, 1)"""
    lam, a, b = 1.0, 0.0, 1.0
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    for _ in range(iters):
        z = a + b * x
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p)
        r = p - y
        ga, gb = lam * a + r.sum(), lam * (b - 1) + (r * x).sum()
        haa, hab, hbb = lam + w.sum(), (w * x).sum(), lam + (w * x * x).sum()
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        a = min(max(a - da, -8), 8)
        b = min(max(b - db, 0.1), 4)
        if abs(da) + abs(db) < 1e-6:
            break
    return a, b


def a7_of(run, P, toks):
    cont = run.cont(P, 7)
    a = 0
    while a < len(cont) and a < 7 and toks[a] == cont[a]:
        a += 1
    return a


def calib_records(run):
    """chain records over every recorded anchor: (depth d = p + 1, logit q, hit) for p <= a7"""
    out = []
    for r in run.recs:
        P = int(r["P"])
        if P + 8 > run.end:
            break
        toks, lq = chain(r)
        a = a7_of(run, P, toks)
        for p in range(min(a + 1, 7)):
            out.append((p + 1, logit_q(lq[p]), 1.0 if p < a else 0.0))
    return out


def calibration(runs):
    recs = [x for r in runs for x in calib_records(r)]
    cal = {}
    for d in range(1, 8):
        xs = [x for (dd, x, y) in recs if dd == d]
        ys = [y for (dd, x, y) in recs if dd == d]
        cal[d] = fit(xs, ys) if len(xs) >= 32 else (0.0, 1.0)
    return cal


def survivals(lq, cal):
    s, out = 1.0, []
    for p in range(7):
        a, b = cal[p + 1]
        f = 1.0 / (1.0 + math.exp(-(a + b * logit_q(lq[p]))))
        s *= f
        out.append(s)
    return out


def allocate_seq(surv, cost, rate, floor=0, nmax=8):
    """the sequential rule: every member its anchor (+ floor drafts), then rows by decreasing survival while each pays
    rate x its cost increment (rate 0: while the ratio E / cost grows); returns drafts per member"""
    M = len(surv)
    k = [min(floor, 7) for _ in range(M)]
    n = M + sum(k)
    E = M + sum(sum(surv[g][:k[g]]) for g in range(M))
    order = sorted(((surv[g][p], p, g) for g in range(M) for p in range(k[g], 7)), key=lambda t: (-t[0], t[1], t[2]))
    for m, p, g in order:
        if n >= nmax:
            break
        if p != k[g]:  # (not reached: survivals fall along a chain)
            continue
        c0, c1 = cost[n], cost[n + 1]
        pays = m >= rate * (c1 - c0) if rate > 0 else (E + m) / c1 > E / c0
        if not pays:
            break
        k[g] += 1
        n += 1
        E += m
    return k


def oracle(a7s, N):
    """best allocation of N rows given the true matched lengths: each draft row of a member is worth 1 while within a7"""
    M = len(a7s)
    extra = N - M
    best = None
    for ks in itertools.product(range(8), repeat=M):
        if sum(ks) != extra:
            continue
        t = sum(min(a, k) + 1 for a, k in zip(a7s, ks))
        if best is None or t > best[0]:
            best = (t, ks)
    return list(best[1])


def cost_model(model, M):
    """GPU ms of a batched cycle (M drafting passes + one verify of N rows), N = 0 .. 8 (index N)"""
    if model == "35b-measured":  # KfBatch's table on the M4 Pro (B's warm-up and round 1): cycle GPU ms by M, N
        t = {2: [0, 0, 20.66, 21.85, 23.53, 25.19, 25.70, 27.00, 28.06],
             3: [0, 0, 0, 26.25, 27.75, 29.31, 29.18, 30.42, 32.00],
             4: [0, 0, 0, 0, 31.22, 32.32, 33.49, 34.76, 36.02]}[M]
        return [t[n] if t[n] else 1e9 for n in range(9)]
    if model.startswith("35b"):
        draft = 3.4
        base = {2: 14.3, 3: 15.0, 4: 15.8}[M]
        return [M * draft + base + 1.2 * n for n in range(9)]
    draft = 8.7
    base = 74.0 + 1.5 * (M - 2)
    return [M * draft + base for n in range(9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="35b")
    ap.add_argument("--rate", type=float, default=-1, help="tok/ms (default: the fixed table's replayed rate)")
    ap.add_argument("recs", nargs="+")
    a = ap.parse_args()
    runs = [Run(p) for p in a.recs]
    table = {"35b": {2: 4, 3: 2, 4: 2}, "35b-measured": {2: 4, 3: 2, 4: 2}, "27b": {2: 4, 3: 2, 4: 2}}[a.model]
    print(f"{a.model}: {len(runs)} recordings, {sum(len(r.recs) for r in runs)} anchors")
    for M in (2, 3, 4):
        groups = [list(c) for c in itertools.combinations(range(len(runs)), M)]
        cost = cost_model(a.model, M)
        res = {}
        for name in ("table", "budget", "budget_floor1", "oracle_same_N", "oracle_8"):
            res[name] = [0, 0, 0.0, 0]  # cycles, tokens, ms, rows
        cals = {}
        rate = a.rate
        for name in res:
            if name != "table" and rate < 0:  # the table's replayed tokens per ms: the long-run rate a row must beat
                rate = res["table"][1] / res["table"][2]
            for gi in groups:
                members = [runs[i] for i in gi]
                if tuple(gi) not in cals:
                    cals[tuple(gi)] = calibration([runs[i] for i in range(len(runs)) if i not in gi])
                cal = cals[tuple(gi)]
                P = [m.P0 for m in members]
                while all(P[g] + 9 <= members[g].end and P[g] in members[g].by_P for g in range(M)):
                    recs = [members[g].recs[members[g].by_P[P[g]]] for g in range(M)]
                    ch = [chain(r) for r in recs]
                    a7s = [a7_of(members[g], P[g], ch[g][0]) for g in range(M)]
                    if name == "table":
                        k = [table[M] - 1] * M
                    elif name.startswith("budget"):
                        surv = [survivals(c[1], cal) for c in ch]
                        k = allocate_seq(surv, cost, rate, floor=1 if name.endswith("floor1") else 0)
                    elif name == "oracle_same_N":
                        k = oracle(a7s, M * table[M])
                    else:
                        k = oracle(a7s, 8)
                    toks = [min(a7s[g], k[g]) + 1 for g in range(M)]
                    N = M + sum(k)
                    st = res[name]
                    st[0] += 1
                    st[1] += sum(toks)
                    st[2] += cost[N]
                    st[3] += N
                    for g in range(M):
                        P[g] += toks[g]
        t = res["table"]
        print(f"  M={M} ({len(groups)} groups; rate {rate:.4f} tok/ms): tokens per cycle / rows / est. tok/s (cost model)")
        for name, st in res.items():
            print(f"    {name:14s} {st[1] / st[0]:.3f} tok/cycle  {st[3] / st[0]:.2f} rows  "
                  f"{1000 * st[1] / st[2]:.1f} tok/s  ({100 * (st[1] / st[2]) / (t[1] / t[2]) - 100:+.1f}% vs table)")


if __name__ == "__main__":
    main()
