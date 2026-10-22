#!/usr/bin/env python3
"""summary.py RUN_DIR ...: a hold's numbers: decode tok/s (each run, median), prefill tok/s, cached TTFT, peak memory,
the lowest free-memory level, and the expert store's lines from the server log (per request)."""
import json
import os
import re
import sys


def f(x, d=1):
    return "NA" if x is None else f"{x:.{d}f}"


for run in sys.argv[1:]:
    print(f"== {os.path.basename(run.rstrip('/'))}")
    rp = os.path.join(run, "result.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        for key, label in (("decode", "T=0.6"), ("decode_greedy", "greedy")):
            runs = r.get(key) or []
            if runs:
                tps = [d.get("decode_tps") for d in runs]
                acc = [d.get("draft_acceptance") for d in runs]
                s = r.get(key + "_summary") or {}
                print(f"  decode {label}: runs {[f(x) for x in tps]} tok/s, median {s.get('median_tps')}, "
                      f"TTFT {[f(d.get('ttft_s'), 2) for d in runs]} s, tokens {[d.get('completion_tokens') for d in runs]}, "
                      f"d4 {[d.get('distinct4') for d in runs]}" + (f", acceptance {acc}" if any(acc) else "")
                      + (f" INVALID {s.get('invalid')}" if s.get("invalid") else "") + (f" flags {s.get('flags')}" if s.get("flags") else ""))
        for p in r.get("prefill") or []:
            c = p.get("cached") or {}
            print(f"  prefill {p.get('target')}: {p.get('prompt_tokens')} tokens, TTFT {f(p.get('ttft_s'), 2)} s = "
                  f"{f(p.get('prefill_tps'), 0)} tok/s; follow-ups TTFT {[f(s.get('ttft_s'), 3) for s in c.get('samples', [])]} "
                  f"(cached {[s.get('cached_tokens') for s in c.get('samples', [])]}), median {c.get('median_ttft_s')}"
                  + (f" INVALID {p.get('invalid') + c.get('invalid', [])}" if p.get("invalid") or c.get("invalid") else ""))
        print(f"  peak footprint {f((r.get('peak_memory_mb') or 0) / 1024, 2)} GB, peak (sampler) "
              f"{f((r.get('peak_footprint_mb') or 0) / 1024, 2)} GB, mapped files {f((r.get('mapped_file_clean_mb') or 0) / 1024, 2)} GB, "
              f"swap {r.get('vm_end', {}).get('swap_used_mb')} MB; flags {r.get('flags')}")
    ml = os.path.join(run, "memlevel.txt")
    if os.path.exists(ml):
        lv = [int(x.split()[1]) for x in open(ml) if len(x.split()) >= 2 and x.split()[1].isdigit()]
        if lv:
            print(f"  free memory level (kern.memorystatus_level, %): min {min(lv)}, max {max(lv)} over {len(lv)} samples "
                  f"= min {min(lv) * 48 / 100:.1f} GB of 48")
    sl = os.path.join(run, "server.log")
    if os.path.exists(sl):
        for line in open(sl, errors="replace"):
            if re.search(r"expert store|chat: prompt|experts in memory|memory:|error|fatal", line, re.I):
                print("  | " + line.rstrip()[:400])
