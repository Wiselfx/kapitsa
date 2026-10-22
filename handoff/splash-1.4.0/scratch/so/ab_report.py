"""The decode rates in an ab_inner.sh run's server logs, per config and round, and whether the replies match.
usage: ab_report.py RUN_DIR"""
import glob, json, os, re, sys

D = sys.argv[1]
line = re.compile(r"chat: prompt (\d+) \(cached (\d+)\).*?, (\d+) tokens in ([\d.]+) ms \(([\d.]+) tok/s\), (\w+)(.*)")
mask = re.compile(r"structured output \((.*?)\), (.*?): (\d+) masks, ([\d.]+) us each")
rows = {}
for log in sorted(glob.glob(os.path.join(D, "server-*.log"))):
    name, rnd = re.match(r"server-(.*)-(\d+)\.log", os.path.basename(log)).groups()
    rates, masks = [], []
    for l in open(log, errors="replace"):
        m = line.search(l)
        if m:
            cyc = re.search(r"([\d.]+) tok/cycle", m.group(7))
            rates.append((int(m.group(3)), float(m.group(5)), cyc.group(1) if cyc else "-"))
        mm = mask.search(l)
        if mm:
            masks.append((int(mm.group(3)), float(mm.group(4))))
    rows.setdefault(name, []).append((int(rnd), rates, masks))
for name, rs in rows.items():
    for rnd, rates, masks in sorted(rs):
        tot_t = sum(t for t, _, _ in rates)
        tot_s = sum(t / r for t, r, _ in rates if r > 0)
        print("%-10s round %d: %s | all %.1f tok/s%s" % (
            name, rnd, "  ".join("%d tok %.1f t/s (%s/cyc)" % x for x in rates), tot_t / tot_s if tot_s else 0,
            ("  masks: " + ", ".join("%d x %.1f us" % x for x in masks)) if masks else ""))
# replies: every config's, every round's, against the first
outs = {}
for f in sorted(glob.glob(os.path.join(D, "out-*.json"))):
    name, rnd = re.match(r"out-(.*)-(\d+)\.json", os.path.basename(f)).groups()
    outs[(name, int(rnd))] = json.load(open(f))
if outs:
    ref_key = sorted(outs)[0]
    ref = outs[ref_key]
    for key, o in sorted(outs.items()):
        same = [a["content"] == b["content"] for a, b in zip(o, ref)]
        print("replies %s round %d vs %s round %d: %s" % (key[0], key[1], ref_key[0], ref_key[1],
                                                          " ".join("same" if s else "DIFF" for s in same)))
