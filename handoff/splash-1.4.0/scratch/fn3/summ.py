#!/usr/bin/env python3
"""summ.py RUN_DIR ...: each round's numbers from its harness results (result.json or inv*-s*.json): decode tok/s per
run and median (T=0.6 and greedy), prefill tok/s with the follow-ups' time to first token, peak memory; the least free
memory (kern.memorystatus_level); foreign GPU load from gpu_hogs.py monitor (mean, max, longest run of samples above
5%), restricted to the timed phases; and the server log's lines about the expert store / arena / plan."""
import glob
import json
import os
import re
import sys


def f(x, d=1):
    return "NA" if x is None else f"{x:.{d}f}"


def contention(run, phases):
    p = os.path.join(run, "gpu.txt")
    if not os.path.exists(p):
        return "no gpu.txt"
    rows = []
    for line in open(p):
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1]), " ".join(parts[2:])))
            except ValueError:
                pass
    if phases:
        rows = [r for r in rows if any(a - 1 <= r[0] <= b + 1 for a, b in phases)]
    if not rows:
        return "no samples in the timed phases"
    longest = cur = 0
    tops = {}
    amb = {"Claude Helper", "Claude", "Claude Helper (Renderer)", "WindowServer", "mediaanalysisd", "photoanalysisd"}
    for t, pct, top in rows:
        if pct > 5.0 and top not in amb:
            cur += 1
            longest = max(longest, cur)
            tops[top] = tops.get(top, 0) + 1
        else:
            cur = 0
    mean = sum(r[1] for r in rows) / len(rows)
    mx = max(rows, key=lambda r: r[1])
    return (f"foreign GPU {len(rows)} samples: mean {mean:.1f}%, max {mx[1]:.1f}% ({mx[2]}), longest run >5% by a non-ambient process: "
            f"{longest} samples" + (f" by {tops}" if tops else ""))


def show(path, phases):
    r = json.load(open(path))
    print(f"  [{os.path.basename(path)}] {r.get('when')} .. {r.get('when_end')}")
    for key, label in (("decode", "T=0.6"), ("decode_greedy", "greedy")):
        runs = r.get(key) or []
        if runs:
            tps = [d.get("decode_tps") for d in runs]
            acc = [d.get("draft_acceptance") for d in runs]
            s = r.get(key + "_summary") or {}
            print(f"    decode {label}: runs {[f(x) for x in tps]} tok/s, median {s.get('median_tps')}; e2e "
                  f"{[f(d.get('decode_tps_e2e')) for d in runs]}; TTFT {[f(d.get('ttft_s'), 2) for d in runs]} s; tokens "
                  f"{[d.get('completion_tokens') for d in runs]}; d4 {[d.get('distinct4') for d in runs]}"
                  + (f"; acceptance {acc}" if any(a is not None for a in acc) else "")
                  + (f" INVALID {s.get('invalid')}" if s.get("invalid") else "")
                  + (f" flags {s.get('flags')}" if s.get("flags") else ""))
    for p in r.get("prefill") or []:
        c = p.get("cached") or {}
        print(f"    prefill {p.get('target')}: {p.get('prompt_tokens')} tokens, TTFT {f(p.get('ttft_s'), 2)} s = "
              f"{f(p.get('prefill_tps'), 0)} tok/s (cold cached {p.get('cold_cached_tokens')}); follow-ups TTFT "
              f"{[f(s.get('ttft_s'), 3) for s in c.get('samples', [])]} s (prompt {[s.get('prompt_tokens') for s in c.get('samples', [])]}, "
              f"cached {[s.get('cached_tokens') for s in c.get('samples', [])]}), median {c.get('median_ttft_s')}"
              + (f" INVALID {p.get('invalid') + c.get('invalid', [])}" if p.get("invalid") or c.get("invalid") else "")
              + (f" flags {p.get('flags') + c.get('flags', [])}" if p.get("flags") or c.get("flags") else ""))
    if r.get("warmup"):
        w = r["warmup"]
        print(f"    warm-up: {[(s['name'], s.get('ok'), s.get('s')) for s in w.get('steps', [])]}")
    print(f"    peak phys_footprint {f((r.get('peak_memory_mb') or 0) / 1024, 2)} GB (sampler "
          f"{f((r.get('peak_footprint_mb') or 0) / 1024, 2)} GB, RSS {f((r.get('peak_rss_mb') or 0) / 1024, 2)} GB), "
          f"mapped files {f((r.get('mapped_file_clean_mb') or 0) / 1024, 2)} GB, swap {r.get('vm_end', {}).get('swap_used_mb')} MB"
          + (f"; INVALID {r.get('invalid')}" if r.get("invalid") else "") + (f"; flags {r.get('flags')}" if r.get("flags") else ""))
    for name, ph in (r.get("phases") or {}).items():
        if name != "warmup":
            phases.append(tuple(ph))


for run in sys.argv[1:]:
    print(f"== {os.path.basename(run.rstrip('/'))}")
    hl = os.path.join(run, "hold.log")
    if os.path.exists(hl):
        for line in open(hl):
            if re.search(r"ready after|not ready|store|cmd:|KILL|exit [1-9]", line):
                print("  hold: " + line.rstrip()[:300])
    phases = []
    files = sorted(glob.glob(os.path.join(run, "result.json")) + glob.glob(os.path.join(run, "inv*-s*.json")))
    for p in files:
        try:
            show(p, phases)
        except Exception as e:
            print(f"  {p}: {e!r}")
    print("  " + contention(run, phases))
    ml = os.path.join(run, "memlevel.txt")
    if os.path.exists(ml):
        lv = [int(x.split()[1]) for x in open(ml) if len(x.split()) >= 2 and x.split()[1].isdigit()]
        if lv:
            print(f"  free memory level: min {min(lv)}% (= {min(lv) * 48 / 100:.1f} GB of 48), max {max(lv)}% over {len(lv)} samples")
    pc = os.path.join(run, "precheck.txt")
    if os.path.exists(pc):
        txt = open(pc).read()
        m = re.search(r"load averages: ([\d.]+)", txt)
        g = re.search(r"gpu users \(4 s\):\n(.*?)(?=Pages free)", txt, re.S)
        mp = re.search(r"memory free percentage: (\d+)%", txt)
        print(f"  precheck: load {m.group(1) if m else '?'}, memory free {mp.group(1) if mp else '?'}%, GPU users: "
              f"{(g.group(1).strip() or 'none') if g else '?'}")
    sl = os.path.join(run, "server.log")
    if os.path.exists(sl):
        pat = re.compile(r"expert store:|memory:|loaded /|[Ee]xpert|arena|PLE|ple|plan|budget|Memory|verify rows|"
                         r"[Ee]rror|fatal|Traceback|Refused|warn", re.I)
        n = 0
        for line in open(sl, errors="replace"):
            if pat.search(line) and "chat: prompt" not in line and n < int(os.environ.get("SUMM_LOG_LINES", "14")):
                print("  | " + line.rstrip()[:330])
                n += 1
