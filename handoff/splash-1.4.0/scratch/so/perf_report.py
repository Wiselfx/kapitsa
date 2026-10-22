"""Session 7's tables from the servers' logs (one "chat: prompt" line per request, in the client's order; a structured
output line comes before its request's line) and the clients' replies.
usage: perf_report.py RUN_DIR free|pairs"""
import glob, json, os, re, sys

RUN, MODE = sys.argv[1], sys.argv[2]
P = re.compile(r"chat: prompt (\d+) \(cached (\d+)\) in ([\d.]+) ms \(\+([\d.]+) ms template.*?\), (\d+) tokens in ([\d.]+) ms "
               r"\(([\d.]+) tok/s\), (\w+)(?:, ([\d.]+) tok/cycle)?")
M = re.compile(r"chat: structured output .*?: (\d+) masks.*?, ([\d.]+) us each")


def parse(log):
    rows, pend = [], None
    for line in open(log, errors="replace"):
        m = M.search(line)
        if m:
            pend = (int(m.group(1)), float(m.group(2)))
            continue
        m = P.search(line)
        if m:
            rows.append({"prompt": int(m.group(1)), "cached": int(m.group(2)), "pf_ms": float(m.group(3)),
                         "tmpl_ms": float(m.group(4)), "tok": int(m.group(5)), "dec_ms": float(m.group(6)),
                         "tps": float(m.group(7)), "finish": m.group(8),
                         "cyc": float(m.group(9)) if m.group(9) else None, "masks": pend})
            pend = None
    return rows


runs = {}
for f in sorted(glob.glob(os.path.join(RUN, "out-*.json"))):
    name, rnd = re.match(r"out-(.+)-(\d+)\.json", os.path.basename(f)).groups()
    reqs = json.load(open(f))
    rows = parse(os.path.join(RUN, "server-%s-%s.log" % (name, rnd)))
    if len(rows) != len(reqs):
        print("MISMATCH %s round %s: %d log lines, %d requests" % (name, rnd, len(rows), len(reqs)))
        continue
    for q, r in zip(reqs, rows):
        q.update(r)
    runs[(name, int(rnd))] = reqs


def agg(rs):
    t = sum(r["tok"] for r in rs)
    ms = sum(r["dec_ms"] for r in rs)
    return t / ms * 1000 if ms else 0


def text(q):  # the reply without the tool calls' random ids
    calls = [(c["function"]["name"], c["function"]["arguments"]) for c in q["tool_calls"] or []]
    return json.dumps([q["content"], calls], sort_keys=True)


if MODE == "free":
    names = sorted({n for n, _ in runs}, key=lambda n: (n.split("-")[1], n.split("-")[0] != "main"))
    print("decode tok/s per request (server: tokens / decode ms), prefill tok/s ((prompt - cached) / prefill ms)")
    for n in names:
        for rnd in (1, 2):
            rs = runs.get((n, rnd))
            if not rs:
                continue
            d = [r for r in rs if r["kind"] == "decode"]
            p = [r for r in rs if r["kind"] == "prefill"]
            print("%-11s r%d  decode %s | all %.1f   prefill %s   (cached %s; tmpl %s ms)" % (
                n, rnd, " ".join("%.1f" % r["tps"] for r in d), agg(d),
                " ".join("%s %d tok %.0f t/s" % (r["label"], r["prompt"], (r["prompt"] - r["cached"]) / r["pf_ms"] * 1000)
                         for r in p), ",".join(str(r["cached"]) for r in p), ",".join("%.0f" % r["tmpl_ms"] for r in p)))
    for mode in ("on", "off"):
        ref = runs.get(("main-%s" % mode, 1))
        for n in ("main-%s" % mode, "branch-%s" % mode):
            for rnd in (1, 2):
                rs = runs.get((n, rnd))
                if rs and ref:
                    print("replies %s r%d vs main-%s r1: %s" % (n, rnd, mode, " ".join(
                        "same" if text(a) == text(b) else "DIFF" for a, b in zip(rs, ref) if a["kind"] == "decode")))
    for mode in ("on", "off"):
        for kind in ("decode",):
            m = [agg([r for r in runs[(n, k)] if r["kind"] == kind]) for n in ["main-" + mode] for k in (1, 2) if (n, k) in runs]
            b = [agg([r for r in runs[(n, k)] if r["kind"] == kind]) for n in ["branch-" + mode] for k in (1, 2) if (n, k) in runs]
            print("spec %-3s decode all: main %s  branch %s" % (mode, " / ".join("%.1f" % x for x in m),
                                                                 " / ".join("%.1f" % x for x in b)))
        for lab in ("4K", "16K"):
            def pf(n):
                return " / ".join("%.0f" % ((r["prompt"] - r["cached"]) / r["pf_ms"] * 1000) for k in (1, 2) if (n, k) in runs
                                  for r in runs[(n, k)] if r["kind"] == "prefill" and r["label"] == lab)
            print("spec %-3s prefill %-3s: main %s  branch %s" % (mode, lab, pf("main-" + mode), pf("branch-" + mode)))
else:
    for n in ("on", "off"):
        for rnd in (1, 2):
            rs = runs.get((n, rnd))
            if not rs:
                continue
            print("== speculation %s, round %d" % (n, rnd))
            for r in rs:
                if r["kind"] == "warm":
                    continue
                mk = ""
                if r["masks"]:
                    cnt, us = r["masks"]
                    mk = "  masks %d x %.1f us = %.1f us per token" % (cnt, us, cnt * us / max(r["tok"], 1))
                tc = r["tool_calls"]
                what = ("tool call %s" % tc[0]["function"]["name"]) if tc else "text"
                print("  %-13s %-9s %4d tok %6.1f tok/s %-6s %s%s" % (r["kind"], r["label"], r["tok"], r["tps"], r["finish"],
                                                                      what, mk))
            pairs = {}
            for r in rs:
                pairs.setdefault(r["label"], {})[r["kind"]] = r
            same = []
            for lab, d in pairs.items():
                ks = [k for k in d if k != "warm"]
                if len(ks) == 2:
                    same.append("%s: %s" % (lab, "same" if text(d[ks[0]]) == text(d[ks[1]]) else "differs"))
            print("  unconstrained vs constrained reply: " + ", ".join(same))
    print()
    for n in ("on", "off"):
        kinds = []
        for k in (1, 2):
            for r in runs.get((n, k), []):
                if r["kind"] != "warm" and r["kind"] not in kinds:
                    kinds.append(r["kind"])
        for kind in kinds:
            vals = [agg([r for r in runs[(n, k)] if r["kind"] == kind]) for k in (1, 2) if (n, k) in runs]
            toks = [sum(r["tok"] for r in runs[(n, k)] if r["kind"] == kind) for k in (1, 2) if (n, k) in runs]
            ms = [(sum(r["masks"][0] * r["masks"][1] for r in runs[(n, k)] if r["kind"] == kind and r["masks"]),
                   sum(r["masks"][0] for r in runs[(n, k)] if r["kind"] == kind and r["masks"]),
                   sum(r["tok"] for r in runs[(n, k)] if r["kind"] == kind and r["masks"])) for k in (1, 2) if (n, k) in runs]
            mtxt = " ".join("%.1f us/mask %.1f us/token" % (a / b, a / c) for a, b, c in ms if b and c)
            print("spec %-3s %-13s all %s tok/s (%s tokens)  %s" % (n, kind, " / ".join("%.1f" % v for v in vals),
                                                                   " / ".join(str(t) for t in toks), mtxt))
