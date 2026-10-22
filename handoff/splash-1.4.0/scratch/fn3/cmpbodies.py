import json, sys
def firsts(p):
    out = []
    for line in open(p):
        if line.strip():
            b = json.loads(line)["body"]
            if len(b["messages"]) <= 2:
                out.append(b)
    return out
a, b = firsts(sys.argv[1]), firsts(sys.argv[2])
print(f"{sys.argv[1].split('/')[-2:]} {len(a)} first turns vs {len(b)}", end=": ")
same = len(a) == len(b) and all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(a, b))
print("IDENTICAL" if same else "DIFFERENT")
if not same:
    for i, (x, y) in enumerate(zip(a, b)):
        if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True):
            ks = set(x) | set(y)
            print(" ", i, [k for k in ks if x.get(k) != y.get(k)])
