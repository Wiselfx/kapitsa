"""Session 10's comparison: the branch's replies against main's, per budget setting (kf, kf0).
usage: gpu_check_report.py RUN_DIR"""
import json, os, sys

RUN = sys.argv[1]


def load(name):
    p = os.path.join(RUN, "out-%s.json" % name)
    return json.load(open(p)) if os.path.exists(p) else None


def same(a, b):
    return a["content"] == b["content"] and a["calls"] == b["calls"]


def first_diff(a, b):
    x, y = a["content"] or "", b["content"] or ""
    i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
    return "at char %d: main %r / branch %r" % (i, x[max(0, i - 30):i + 30], y[max(0, i - 30):i + 30])


for mode in ("kf", "kf0"):
    m, b = load("main-" + mode), load("branch-" + mode)
    if not m or not b:
        print("%s: missing (%s)" % (mode, "main" if not m else "branch"))
        continue
    print("== %s" % mode)
    for k, (x, y) in enumerate(zip(m["alone"], b["alone"])):
        print("  alone %s: %s (%d / %d tokens)" % (("greedy", "sampled")[k], "same" if same(x, y) else "DIFFERS " + first_diff(x, y),
                                                 x["tokens"], y["tokens"]))
    for r, (rm, rb) in enumerate(zip(m["rounds"], b["rounds"])):
        parts = []
        for k, (x, y) in enumerate(zip(rm, rb)):
            if y["constrained"]:
                parts.append("%d schema %s" % (k, "valid" if y["valid"] else "INVALID"))
            else:
                parts.append("%d free %s" % (k, "same" if same(x, y) else "DIFFERS"))
        print("  round %d: %s" % (r, ", ".join(parts)))
        for k, (x, y) in enumerate(zip(rm, rb)):
            if not y["constrained"] and not same(x, y):
                print("    request %d: %s" % (k, first_diff(x, y)))
