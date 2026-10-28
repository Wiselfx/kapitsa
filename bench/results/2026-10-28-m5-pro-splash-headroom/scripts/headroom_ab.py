#!/usr/bin/env python3
"""Issue #753: does `kapitsa serve --engine splash` keep too much memory back? A/B of Splash's `--expert-memory` at
A - 14 GiB (what src/splash/plan.cpp does today) against A - 10 GiB, where A is the free+reclaimable memory at launch.

Plain Splash 1.4.0 (`python -m server.server`) is the engine under test, as in the issue: Kapitsa's plan only computes this one
flag, so plain Splash with the same flag is the same engine without a second variable. Everything else is the engines bench's harness
(`../../2026-10-28-m5-pro-flash-next-engines/scripts/flash_next_bench.py`, imported, not copied): the same frozen prompts, the same
requests, the same memory guard. This file adds the two arms, the pairing, /status read counters per request and the verdict.

  headroom_ab.py preflight                       checks inputs and prints the plan; no GPU, no lock, loads nothing
  headroom_ab.py plan [--pairs 4]                the leg order and the exact command of every leg
  headroom_ab.py leg --arm A14|A10 --pair P --idx I --out DIR [--reps 3]
                                                 one server start, one arm; run it under the GPU lock (run_ab.sh does)
  headroom_ab.py summarize DIR                   table, summary.json and the verdict of the pre-registered rule
  headroom_ab.py selftest                        CPU checks of the arithmetic, the label parser and the verdict

Exit codes of `leg`: 0 done, 75 not started (memory not as planned, retry), 3 stopped by the memory guard, 4 Splash refused or
exited, 5 an A10 leg finished but broke a safety bound (the verdict is then KEEP 14 GiB; more legs would add nothing), 6 the Mac
was not on AC power at the start or the end of the leg (timing on battery is not valid; a leg that ended on battery is kept as
legN-ARM-battery.json and does not count), 2 usage. The driver (run_ab.sh) stops on 3, 4, 5 and 6 and does not go on."""
import argparse, glob, importlib.util, json, math, os, re, signal, statistics, subprocess, sys, tempfile, threading, time, types, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
FNB_PATH = os.path.join(os.path.dirname(RES), "2026-10-28-m5-pro-flash-next-engines", "scripts", "flash_next_bench.py")

GIB = 1 << 30
HEADROOM = {"A14": 14, "A10": 10}   # GiB kept back from A; A14 is kHeadroom in src/splash/plan.cpp today
EM_MIN, EM_MAX = 8, 28              # kMinCache, and the cap of the proposed clamp(floor(A - 10), 8, 28)
MIN_START_A = 31.0                  # issue #753: start a leg only when A >= 31 GiB
PAIR_DRIFT = 1.5                    # a pair's second leg starts only when A >= the pair's A - 1.5 GiB
PAIRS = 4                           # legs per arm; order A14 A10 | A10 A14 | A14 A10 | A10 A14
REPS = 3
EX_NOT_STARTED, EX_GUARD, EX_REFUSED, EX_A10_UNSAFE, EX_BATTERY = 75, 3, 4, 5, 6

# The pre-registered rule (README.md, "Decision rule"). Constants here are what summarize applies.
PRIMARY = {"turn2_ttft_s": "lower", "ttft_4K_s": "lower"}
SECONDARY = {"ttft_1K_s": "lower", "ttft_16K_s": "lower", "turn1_ttft_s": "lower", "decode_T0_tps": "higher", "decode_T06_tps": "higher"}
MIN_PAIRS = 3          # complete pairs needed for any verdict
FAVOR_FRACTION = 0.75  # of pairs, A-10 must be better in at least this fraction (3 of 4)
MIN_GAIN_PCT = 5.0     # median paired gain, % of the A-14 median
SAFE_MIN_FREE_GIB = 3.0
SAFE_MAX_SWAP_GROWTH_GB = 1.0


def on_ac():
    """True when `pmset -g batt` says the Mac draws from AC power (a desktop without a battery says AC too)."""
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return "'AC Power'" in out


def load_fnb():
    spec = importlib.util.spec_from_file_location("flash_next_bench", FNB_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def em_gib(a_gib, arm):
    """Whole GiB handed to --expert-memory: clamp(floor(A) - headroom, 8, 28)."""
    return max(EM_MIN, min(EM_MAX, int(math.floor(a_gib)) - HEADROOM[arm]))


def order(pairs):
    return [(2 * p + k, "A14" if (p + k) % 2 == 0 else "A10", p) for p in range(pairs) for k in (0, 1)]


# ---- /status read counters ---------------------------------------------------------------------------------------------

def flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(flatten(v, prefix + str(k) + "."))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix[:-1]] = d
    return out


def delta(before, after):
    """Per-key difference of the numeric leaves under streamed_experts; keys present in both snapshots."""
    if before is None or after is None: return None
    b, a = flatten(before.get("streamed_experts") or {}), flatten(after.get("streamed_experts") or {})
    return {"streamed_experts." + k: a[k] - b[k] for k in a if k in b}


def status(port):
    try: return json.load(urllib.request.urlopen("http://127.0.0.1:%d/status" % port, timeout=5))
    except Exception: return None


LABEL = re.compile(r"\[session (?:b\d+-\d+(T06|T0|1K|4K|16K|tt)|warm\d+|smoke)\]")


def label(msgs, tools):
    m = LABEL.match(msgs[0]["content"])
    if not m: return "other"
    k = m.group(1)
    if k is None: return "warmup"
    if k == "tt": return "turn2" if len(msgs) > 2 else "turn1"
    return {"T0": "decode_T0", "T06": "decode_T06"}.get(k, "prefill_" + k)


def patch_chat(fnb, reqlog, snap):
    orig = fnb.Leg.chat

    def chat(self, msgs, max_tokens, temperature=0.0, tools=None, cache=True):
        before = status(self.port)
        if "first" not in snap: snap["first"] = before   # the whole /status once, so the counters' names are on record
        r = orig(self, msgs, max_tokens, temperature, tools, cache)
        after = status(self.port)
        snap["last"] = after
        reqlog.append({"label": label(msgs, tools), "prompt_tokens": r["prompt_tokens"], "cached_tokens": r["cached_tokens"],
                       "completion_tokens": r["completion_tokens"], "ttft_s": r["ttft_s"], "decode_tps": r["decode_tps"],
                       "reads": delta(before, after), "status_before": before is not None, "status_after": after is not None,
                       "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return r
    fnb.Leg.chat = chat


# ---- leg ---------------------------------------------------------------------------------------------------------------

def run_leg(a):
    fnb = load_fnb()
    if fnb.running_model_processes():
        print("a model process is already running, not starting:\n" + "\n".join(fnb.running_model_processes())); return EX_NOT_STARTED
    if not on_ac():
        print("ON BATTERY: not starting; timing on battery is not valid (pmset -g batt)"); return EX_BATTERY
    os.makedirs(a.out, exist_ok=True)
    a_now = fnb.vm()["reclaimable_gib"]
    pf = os.path.join(a.out, "pair%d.json" % a.pair)
    pair = json.load(open(pf)) if os.path.exists(pf) else None
    if a_now < a.min_start:
        print("WAIT: A = %.1f GiB at launch, need >= %.1f" % (a_now, a.min_start)); return EX_NOT_STARTED
    if pair and a_now < pair["a_launch_gib"] - PAIR_DRIFT:
        print("WAIT: A = %.1f GiB, pair %d was planned on %.1f (drift limit %.1f)" % (a_now, a.pair, pair["a_launch_gib"], PAIR_DRIFT)); return EX_NOT_STARTED
    if pair is None:
        pair = {"pair": a.pair, "a_launch_gib": a_now, "a_floor_gib": int(math.floor(a_now)), "first_leg_idx": a.idx,
                "em_gib": {k: em_gib(a_now, k) for k in HEADROOM}}
        if a.em_gib is not None: pair["em_gib"][a.arm] = a.em_gib
        json.dump(pair, open(pf, "w"), indent=1)
    em = pair["em_gib"][a.arm]
    fnb.SPLASH_FLAGS = ["--expert-memory", str(em * GIB), "--idle-release", "off"]
    reqlog, snap, fine = [], {}, {"min": a_now, "stop": False}
    patch_chat(fnb, reqlog, snap)

    def watch_free():  # free+reclaimable every 0.5 s (the harness's own sampler reads it every 2 s)
        while not fine["stop"]:
            try: fine["min"] = min(fine["min"], fnb.vm()["reclaimable_gib"])
            except Exception: pass
            time.sleep(0.5)
    threading.Thread(target=watch_free, daemon=True).start()
    ns = types.SimpleNamespace(arm="S", idx=a.idx, out=a.out, reps=a.reps, smoke=False, group=a.arm, kbin=fnb.BIN["K"], nbin=fnb.BIN["N"],
                               llama=fnb.BIN["L"], nenv="none")
    print("leg %d %s pair %d: A = %.2f GiB, --expert-memory %d GiB (headroom %.2f GiB)" % (a.idx, a.arm, a.pair, a_now, em, a_now - em), flush=True)
    stem = os.path.join(a.out, "leg%d-" % a.idx)
    try:
        res = fnb.run_leg(ns)
        fine["stop"] = True
    except SystemExit as e:
        fine["stop"] = True
        for l in fnb.running_model_processes():
            try: os.kill(int(l.split()[0]), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError): pass
        log = stem + "S.log"
        tail = open(log, errors="replace").read()[-3000:] if os.path.exists(log) else ""
        json.dump({"arm": a.arm, "idx": a.idx, "pair": a.pair, "em_gib": em, "a_launch_gib": a_now, "reason": str(e), "log_tail": tail,
                   "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open("%srefused-%s.json" % (stem, a.arm), "w"), indent=1)
        if os.path.exists(log): os.replace(log, "%s%s.log" % (stem, a.arm))
        guard = str(e).startswith("aborted while loading")
        print("%s: %s" % ("STOPPED BY THE MEMORY GUARD" if guard else "REFUSED/EXITED", e)); return EX_GUARD if guard else EX_REFUSED
    res.update({"arm": a.arm, "pair": a.pair, "headroom_gib_nominal": HEADROOM[a.arm], "em_gib": em, "a_launch_gib": a_now,
                "a_pair_gib": pair["a_launch_gib"], "expert_memory_flags": fnb.SPLASH_FLAGS, "requests": reqlog,
                "status_first": snap.get("first"), "status_last": snap.get("last")})
    res["power_at_end"] = "AC" if on_ac() else "battery"
    res["memory"]["free_reclaimable_gib_min_fine"] = fine["min"]
    res["memory"]["free_reclaimable_gib_min"] = min(res["memory"]["free_reclaimable_gib_min"], fine["min"])
    if not any(r.get("reads") for r in reqlog): print("WARNING: /status gave no streamed_experts counters; read columns will be empty")
    json.dump(res, open("%s%s.json" % (stem, a.arm), "w"), indent=1)
    if res["power_at_end"] != "AC":
        os.replace("%s%s.json" % (stem, a.arm), "%s%s-battery.json" % (stem, a.arm))
        print("ON BATTERY at the end of the leg: kept as leg%d-%s-battery.json, does not count" % (a.idx, a.arm)); res_rc = EX_BATTERY
    else: res_rc = None
    os.remove("%sS.json" % stem)
    if os.path.exists("%sS.log" % stem): os.replace("%sS.log" % stem, "%s%s.log" % (stem, a.arm))
    if res_rc: return res_rc
    if res["memory"]["aborted"]: return EX_GUARD
    return EX_A10_UNSAFE if a.arm == "A10" and safety([res]) else 0


# ---- summary and verdict -------------------------------------------------------------------------------------------------

def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def leg_cells(l):
    c = {}
    c["decode_T0_tps"] = med([r["decode_tps"] for r in l["decode"]["T0"] if (r["completion_tokens"] or 0) >= 250])
    c["decode_T06_tps"] = med([r["decode_tps"] for r in l["decode"]["T06"] if (r["completion_tokens"] or 0) >= 250])
    for s in ("1K", "4K", "16K"): c["ttft_%s_s" % s] = med([r["ttft_s"] for r in l["prefill"][s]])
    tt = [t for t in l["twoturn"] if "turn2" in t]
    c["turn2_ttft_s"] = med([t["turn2"]["ttft_s"] for t in tt])
    c["turn1_ttft_s"] = med([t["turn1"]["ttft_s"] for t in tt])
    return c


def paired(cells14, cells10, direction, pairs):
    """Per pair, gain = how much better A-10 is than A-14 (positive = A-10 better); noise = larger leg-median range of the two arms."""
    d14 = [cells14[p] for p in pairs]; d10 = [cells10[p] for p in pairs]
    if any(x is None for x in d14 + d10): return None
    sign = 1 if direction == "lower" else -1
    gains = [sign * (x - y) for x, y in zip(d14, d10)]
    base = statistics.median(d14)
    noise = max(max(d14) - min(d14), max(d10) - min(d10))
    mg = statistics.median(gains)
    need = math.ceil(FAVOR_FRACTION * len(pairs))
    fav = sum(g > 0 for g in gains); unfav = sum(g < 0 for g in gains)
    pct = 100 * mg / base if base else 0.0
    return {"n_pairs": len(pairs), "median_a14": base, "median_a10": statistics.median(d10), "median_gain": mg, "gain_pct": pct, "noise_range": noise,
            "pairs_a10_better": fav, "pairs_a10_worse": unfav, "gains": gains,
            "a10_wins": fav >= need and mg > noise and pct >= MIN_GAIN_PCT,
            "a10_loses": unfav >= need and -mg > noise and -pct >= MIN_GAIN_PCT}


def safety(legs):
    bad = []
    for l in legs:
        m = l["memory"]
        why = []
        if m["aborted"]: why.append("guard stopped it: %s" % m["aborted"])
        if m["free_reclaimable_gib_min"] < SAFE_MIN_FREE_GIB: why.append("free+reclaimable fell to %.1f GiB (< %.0f)" % (m["free_reclaimable_gib_min"], SAFE_MIN_FREE_GIB))
        if m["swap_gb_max"] - m["swap_gb_start"] > SAFE_MAX_SWAP_GROWTH_GB: why.append("swap grew %.2f GB (> %.0f)" % (m["swap_gb_max"] - m["swap_gb_start"], SAFE_MAX_SWAP_GROWTH_GB))
        if m["pressure_level_max"] >= 4: why.append("pressure level 4")
        if l["errors"]: why.append("request errors: %s" % "; ".join(l["errors"])[:200])
        if why: bad.append({"leg": l["idx"], "arm": l["arm"], "why": why})
    return bad


def read_medians(legs):
    out = {}
    for l in legs:
        for r in l.get("requests", []):
            if not r.get("reads"): continue
            for k, v in r["reads"].items(): out.setdefault(r["label"], {}).setdefault(k, []).append(v)
    return {lab: {k: statistics.median(v) for k, v in sorted(d.items())} for lab, d in out.items()}


def summarize(d):
    legs, refused = [], []
    for f in sorted(glob.glob(os.path.join(d, "leg*-A1[04].json"))): legs.append(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(d, "leg*-refused-A1[04].json"))): refused.append(json.load(open(f)))
    by = {k: sorted([l for l in legs if l["arm"] == k], key=lambda l: l["idx"]) for k in HEADROOM}
    done = {}
    for l in legs: done.setdefault(l["pair"], {})[l["arm"]] = l
    full = sorted(p for p, v in done.items() if set(v) == set(HEADROOM) and not any(x["memory"]["aborted"] for x in v.values()))
    cells = {l["idx"]: leg_cells(l) for l in legs}
    out = {"rule": {"primary": PRIMARY, "secondary": SECONDARY, "min_pairs": MIN_PAIRS, "favor_fraction": FAVOR_FRACTION, "min_gain_pct": MIN_GAIN_PCT,
                    "safe_min_free_gib": SAFE_MIN_FREE_GIB, "safe_max_swap_growth_gb": SAFE_MAX_SWAP_GROWTH_GB},
           "legs": [{"idx": l["idx"], "arm": l["arm"], "pair": l["pair"], "a_launch_gib": l["a_launch_gib"], "a_pair_gib": l["a_pair_gib"], "em_gib": l["em_gib"],
                     "free_min_gib": l["memory"]["free_reclaimable_gib_min"], "swap_growth_gb": l["memory"]["swap_gb_max"] - l["memory"]["swap_gb_start"],
                     "pressure_max": l["memory"]["pressure_level_max"], "peak_footprint_mib": l["memory"]["peak_footprint_mib"], "cells": cells[l["idx"]]} for l in sorted(legs, key=lambda l: l["idx"])],
           "refused": refused, "complete_pairs": full, "cells": {}, "reads_median_per_label": {k: read_medians(by[k]) for k in HEADROOM}}
    for cell, direction in {**PRIMARY, **SECONDARY}.items():
        c14 = {p: cells[done[p]["A14"]["idx"]][cell] for p in full}
        c10 = {p: cells[done[p]["A10"]["idx"]][cell] for p in full}
        out["cells"][cell] = (paired(c14, c10, direction, full) if len(full) >= 1 else None) or {"n_pairs": 0}
        out["cells"][cell]["direction"] = direction
        out["cells"][cell]["primary"] = cell in PRIMARY
    unsafe = {k: safety(by[k]) for k in HEADROOM}
    reasons = []
    if refused: reasons.append("Splash refused or exited in %d leg(s): %s" % (len(refused), ", ".join("%s leg %d" % (r["arm"], r["idx"]) for r in refused)))
    reasons += ["%s leg %d: %s" % (b["arm"], b["leg"], "; ".join(b["why"])) for k in HEADROOM for b in unsafe[k] if k == "A10"]
    c = out["cells"]
    if len(full) < MIN_PAIRS: verdict, why = "INCONCLUSIVE", "%d complete pairs, need %d" % (len(full), MIN_PAIRS)
    elif any(b for b in unsafe["A10"]) or any(r["arm"] == "A10" for r in refused): verdict, why = "KEEP 14 GiB (A-10 unsafe)", "; ".join(reasons)
    else:
        wins = [k for k in PRIMARY if c[k].get("a10_wins")]
        loses = [k for k in {**PRIMARY, **SECONDARY} if c[k].get("a10_loses")]
        if wins and not loses: verdict, why = "ADOPT A-10 CANDIDATE", "A-10 wins on %s, loses on nothing; open a PR with this A/B" % ", ".join(wins)
        elif loses: verdict, why = "KEEP 14 GiB", "A-10 is slower on %s" % ", ".join(loses)
        else: verdict, why = "KEEP 14 GiB", "no primary cell shows a gain beyond the rule's noise bound"
    out["verdict"] = {"verdict": verdict, "why": why, "safety_a14": unsafe["A14"], "safety_a10": unsafe["A10"]}
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    rows = ["| cell | A-14 median | A-10 median | median gain (A-10 better +) | % | noise (leg-median range) | pairs A-10 better / worse of n | A-10 wins | A-10 loses |", "|---|---|---|---|---|---|---|---|---|"]
    for cell, v in c.items():
        if v.get("n_pairs"): rows.append("| %s%s | %.3f | %.3f | %+.3f | %+.1f | %.3f | %d / %d of %d | %s | %s |" % (
            cell, " (primary)" if v["primary"] else "", v["median_a14"], v["median_a10"], v["median_gain"], v["gain_pct"], v["noise_range"],
            v["pairs_a10_better"], v["pairs_a10_worse"], v["n_pairs"], "yes" if v["a10_wins"] else "no", "yes" if v["a10_loses"] else "no"))
    rows += ["", "| idx | arm | pair | A at launch GiB | --expert-memory GiB | min free+reclaimable GiB | swap growth GB | pressure max | peak footprint MiB |", "|---|---|---|---|---|---|---|---|---|"]
    rows += ["| %d | %s | %d | %.1f | %d | %.1f | %.2f | %d | %.0f |" % (l["idx"], l["arm"], l["pair"], l["a_launch_gib"], l["em_gib"], l["free_min_gib"], l["swap_growth_gb"], l["pressure_max"], l["peak_footprint_mib"]) for l in out["legs"]]
    rows += ["", "**Verdict: %s.** %s" % (verdict, why)]
    open(os.path.join(d, "table.md"), "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))
    return out


# ---- preflight and plan -------------------------------------------------------------------------------------------------

def leg_cmd(idx, arm, pair, out, reps):
    return "python3 scripts/headroom_ab.py leg --arm %s --pair %d --idx %d --out %s --reps %d" % (arm, pair, idx, out, reps)


def preflight(a):
    fnb = load_fnb(); ok = True
    for what, path in (("Splash engine", fnb.SPLASH_BIN), ("Splash python", fnb.SPLASH_PY), ("model folder", fnb.SPLASH_MODEL), ("GGUF", fnb.GGUF)) + tuple(
            ("prompt " + s, "%s/prompts/ctx_%s.txt" % (os.path.dirname(RES) + "/2026-10-28-m5-pro-flash-next-engines", s)) for s in fnb.SIZES):
        e = os.path.exists(path); ok &= e; print("%-14s %s %s" % (what, "ok " if e else "MISSING", path))
    ac = on_ac(); ok &= ac
    print("power          %s" % ("AC Power" if ac else "BATTERY  <- timing on battery is not valid; plug in"))
    v = fnb.vm(); busy = fnb.running_model_processes()
    print("A now          %.1f GiB (need >= %.0f to start)%s; wired %.1f GiB; swap used %.2f GB; pressure level %s; memorystatus %s%%" % (
        v["reclaimable_gib"], MIN_START_A, "" if v["reclaimable_gib"] >= MIN_START_A else "  <- too low now", v["wired_gib"], fnb.swap_gb(),
        fnb.sysctl_int("kern.memorystatus_vm_pressure_level"), fnb.sysctl_int("kern.memorystatus_level")))
    print("model servers  %s" % ("none running" if not busy else "RUNNING: " + "; ".join(busy)))
    print("--expert-memory if launched now: A14 = %d GiB, A10 = %d GiB" % (em_gib(v["reclaimable_gib"], "A14"), em_gib(v["reclaimable_gib"], "A10")))
    return 0 if ok else 1


def plan(a):
    print("%d legs, %d reps each (n = %d legs and %d requests per cell per arm)" % (2 * a.pairs, REPS, a.pairs, a.pairs * REPS))
    for idx, arm, p in order(a.pairs): print("leg %2d  pair %d  %-3s  %s" % (idx, p, arm, leg_cmd(idx, arm, p, "legs", REPS)))
    return 0


# ---- selftest ------------------------------------------------------------------------------------------------------------

def selftest():
    fails = []
    def eq(name, got, want):
        if got != want: fails.append("%s: got %r, want %r" % (name, got, want))
    eq("em 31.9 A14", em_gib(31.9, "A14"), 17); eq("em 31.9 A10", em_gib(31.9, "A10"), 21)
    eq("em 29.3 A10", em_gib(29.3, "A10"), 19); eq("em 45 A10 cap", em_gib(45, "A10"), 28); eq("em 20 A14 floor", em_gib(20, "A14"), 8)
    eq("order", order(4), [(0, "A14", 0), (1, "A10", 0), (2, "A10", 1), (3, "A14", 1), (4, "A14", 2), (5, "A10", 2), (6, "A10", 3), (7, "A14", 3)])
    eq("flatten", flatten({"a": {"b": 1, "c": {"d": 2.5}}, "e": "x", "f": True}), {"a.b": 1, "a.c.d": 2.5})
    eq("delta", delta({"streamed_experts": {"prefill": {"reads": 10, "x": 1}}}, {"streamed_experts": {"prefill": {"reads": 25, "x": 1}, "new": 3}}),
       {"streamed_experts.prefill.reads": 15, "streamed_experts.prefill.x": 0})
    eq("delta none", delta(None, {}), None)
    m = lambda s, n=2: [{"role": "system", "content": "[session %s] Answer." % s}] * n
    eq("label decode", label(m("b3-0T0"), None), "decode_T0"); eq("label decode06", label(m("b3-1T06"), None), "decode_T06")
    eq("label 4K", label(m("b3-24K"), None), "prefill_4K"); eq("label 16K", label(m("b3-016K"), None), "prefill_16K")
    eq("label t1", label(m("b3-0tt", 2), None), "turn1"); eq("label t2", label(m("b3-0tt", 4), None), "turn2"); eq("label warm", label(m("warm3"), None), "warmup")
    # verdict on synthetic legs
    def mk(d, idx, arm, pair, turn2, ttft4k, free=8.0, swap=0.0, level=1, dec=55.0):
        d = str(d)
        pr = lambda v, n: [{"ttft_s": v, "prompt_tokens": n, "cached_tokens": 0, "completion_tokens": 16, "decode_tps": None, "prefill_tps": n / v}] * 2
        l = {"arm": arm, "idx": idx, "pair": pair, "em_gib": 17, "a_launch_gib": 31.5, "a_pair_gib": 31.5, "errors": [],
             "decode": {"T0": [{"decode_tps": dec, "completion_tokens": 256}] * 2, "T06": [{"decode_tps": dec, "completion_tokens": 256}] * 2},
             "prefill": {"1K": pr(1.4, 1035), "4K": pr(ttft4k, 4074), "16K": pr(16.0, 19286)},
             "twoturn": [{"turn1": {"ttft_s": 3.7}, "turn2": {"ttft_s": turn2}}] * 2,
             "memory": {"aborted": None, "free_reclaimable_gib_min": free, "swap_gb_start": 1.0, "swap_gb_max": 1.0 + swap, "pressure_level_max": level, "peak_footprint_mib": 25000.0},
             "requests": [{"label": "turn2", "reads": {"streamed_experts.prefill.reads": 100 if arm == "A14" else 60}}]}
        json.dump(l, open("%s/leg%d-%s.json" % (d, idx, arm), "w"))
    def scenario(name, want, fn):
        with tempfile.TemporaryDirectory() as t:
            for p in range(4):
                for k, arm in ((0, "A14" if p % 2 == 0 else "A10"), (1, "A10" if p % 2 == 0 else "A14")): fn(t, 2 * p + k, arm, p)
            sys.stdout = open(os.devnull, "w")
            try: got = summarize(t)["verdict"]["verdict"]
            finally: sys.stdout = sys.__stdout__
            eq(name, got, want)
    noise = lambda p: 0.01 * (p % 2)  # small leg-to-leg wobble
    scenario("faster A10", "ADOPT A-10 CANDIDATE", lambda t, i, arm, p: mk(t, i, arm, p, (0.56 if arm == "A14" else 0.45) + noise(p), 3.5 + noise(p)))
    scenario("same speed", "KEEP 14 GiB", lambda t, i, arm, p: mk(t, i, arm, p, 0.56 + noise(i), 3.5 + noise(i)))
    scenario("gain inside noise", "KEEP 14 GiB", lambda t, i, arm, p: mk(t, i, arm, p, 0.56 - (0.02 if arm == "A10" else 0) + 0.06 * (i % 2), 3.5))
    scenario("faster but low free", "KEEP 14 GiB (A-10 unsafe)", lambda t, i, arm, p: mk(t, i, arm, p, 0.56 if arm == "A14" else 0.45, 3.5, free=2.0 if arm == "A10" else 8.0))
    scenario("faster but swaps", "KEEP 14 GiB (A-10 unsafe)", lambda t, i, arm, p: mk(t, i, arm, p, 0.56 if arm == "A14" else 0.45, 3.5, swap=1.5 if arm == "A10" else 0.0))
    scenario("faster turn 2, slower decode", "KEEP 14 GiB", lambda t, i, arm, p: mk(t, i, arm, p, 0.56 if arm == "A14" else 0.45, 3.5, dec=55.0 if arm == "A14" else 45.0))
    with tempfile.TemporaryDirectory() as t:
        for p in range(2):
            for k, arm in ((0, "A14"), (1, "A10")): mk(t, 2 * p + k, arm, p, 0.5, 3.5)
        sys.stdout = open(os.devnull, "w")
        try: got = summarize(t)["verdict"]["verdict"]
        finally: sys.stdout = sys.__stdout__
        eq("two pairs", got, "INCONCLUSIVE")
    glue_test(eq)
    for f in fails: print("FAIL", f)
    print("selftest: %s" % ("FAILED (%d)" % len(fails) if fails else "ok"))
    return 1 if fails else 0


def glue_test(eq):
    """A whole leg against a fake Leg (no server, no GPU): the harness's real run_leg drives the real request sequence, our wrapper
    labels each request, reads /status deltas, writes pair and leg files, and summarize reads them back."""
    global load_fnb, status, on_ac
    real_load, real_status, real_ac = load_fnb, status, on_ac
    on_ac = lambda: True
    fnb = real_load()
    calls = {"n": 0}

    def fake_init(self, arm, log, a):
        self.arm, self.port, self.a, self.cmd, self.up_s, self.free_start, self.free_min, self.wired_max = arm, 1, a, ["fake"], 0.1, 32.0, 31.0, 20.0
        self.rss_peak, self.fp_peak, self.fp_samples, self.fp_pids, self.sys_used_at_fp_peak = 1, 2 ** 30, 1, 1, 30.0
        self.swap0, self.swap_max, self.level_max, self.pct_min, self.aborted, self.model = 1.0, 1.0, 1, 50, None, None
    def fake_chat(self, msgs, max_tokens, temperature=0.0, tools=None, cache=True):
        calls["n"] += 1
        tc = [{"id": "c", "name": "read_file", "arguments": "{}"}] if tools and len(msgs) == 2 else []
        return {"reasoning": "", "content": "x", "finish": "stop", "tool_calls": tc, "prompt_tokens": 1000, "completion_tokens": 256, "cached_tokens": 0,
                "ttft_s": 0.5, "total_s": 5.0, "decode_tps": 50.0, "text_md5": "0", "engine_timings": None}
    saved = (fnb.Leg.__init__, fnb.Leg.chat, fnb.Leg.close, fnb.running_model_processes, fnb.vm, fnb.binary_rev)
    fnb.Leg.__init__, fnb.Leg.chat, fnb.Leg.close = fake_init, fake_chat, lambda self: None
    fnb.running_model_processes, fnb.vm, fnb.binary_rev = lambda: [], lambda: {"reclaimable_gib": 32.4, "wired_gib": 20.0}, lambda a: {"path": "x"}
    load_fnb = lambda: fnb
    status = lambda port: {"streamed_experts": {"prefill": {"reads": 10 * calls["n"], "bytes": 100 * calls["n"]}}}
    try:
        with tempfile.TemporaryDirectory() as t:
            rc = [run_leg(types.SimpleNamespace(arm=arm, pair=0, idx=i, out=t, reps=2, min_start=31.0, em_gib=None)) for i, arm in enumerate(("A14", "A10"))]
            eq("glue rc", rc, [0, 0])
            l = json.load(open(t + "/leg1-A10.json")); p = json.load(open(t + "/pair0.json"))
            eq("glue em", (p["em_gib"], l["em_gib"], l["expert_memory_flags"]), ({"A14": 18, "A10": 22}, 22, ["--expert-memory", str(22 * GIB), "--idle-release", "off"]))
            labels = sorted({r["label"] for r in l["requests"]})
            eq("glue labels", labels, ["decode_T0", "decode_T06", "prefill_16K", "prefill_1K", "prefill_4K", "turn1", "turn2", "warmup"])
            eq("glue reads", all(r["reads"] and r["reads"]["streamed_experts.prefill.reads"] == 10 for r in l["requests"]), True)
            eq("glue status kept", bool(l["status_first"] and l["status_last"]), True)
            eq("glue files", sorted(f for f in os.listdir(t)), ["leg0-A14.json", "leg1-A10.json", "pair0.json"])
            sys.stdout = open(os.devnull, "w")
            try: out = summarize(t)
            finally: sys.stdout = sys.__stdout__
            eq("glue verdict", out["verdict"]["verdict"], "INCONCLUSIVE")
            eq("glue read medians", out["reads_median_per_label"]["A10"]["turn2"], {"streamed_experts.prefill.bytes": 100, "streamed_experts.prefill.reads": 10})
            on_ac = lambda: False
            eq("glue battery at start", run_leg(types.SimpleNamespace(arm="A14", pair=1, idx=2, out=t, reps=1, min_start=31.0, em_gib=None)), EX_BATTERY)
            eq("glue battery no files", os.path.exists(t + "/pair1.json"), False)
            on_ac = lambda: True
            fnb.vm = lambda: {"reclaimable_gib": 20.0, "wired_gib": 20.0}  # too little free memory: must not start
            eq("glue wait", run_leg(types.SimpleNamespace(arm="A14", pair=1, idx=2, out=t, reps=1, min_start=31.0, em_gib=None)), EX_NOT_STARTED)
            eq("glue no files on wait", os.path.exists(t + "/pair1.json"), False)
    finally:
        fnb.Leg.__init__, fnb.Leg.chat, fnb.Leg.close, fnb.running_model_processes, fnb.vm, fnb.binary_rev = saved
        load_fnb, status, on_ac = real_load, real_status, real_ac


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight"); sub.add_parser("selftest")
    p = sub.add_parser("plan"); p.add_argument("--pairs", type=int, default=PAIRS)
    l = sub.add_parser("leg")
    l.add_argument("--arm", required=True, choices=sorted(HEADROOM)); l.add_argument("--pair", type=int, required=True); l.add_argument("--idx", type=int, required=True)
    l.add_argument("--out", required=True); l.add_argument("--reps", type=int, default=REPS)
    l.add_argument("--min-start", type=float, default=MIN_START_A); l.add_argument("--em-gib", type=int, default=None, help="test only: override this arm's --expert-memory")
    s = sub.add_parser("summarize"); s.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "selftest": return selftest()
    if a.cmd == "preflight": return preflight(a)
    if a.cmd == "plan": return plan(a)
    if a.cmd == "summarize": summarize(a.dir); return 0
    return run_leg(a)


if __name__ == "__main__":
    sys.exit(main())
