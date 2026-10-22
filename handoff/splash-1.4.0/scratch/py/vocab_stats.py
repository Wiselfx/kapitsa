import json, sys
j = json.load(open(sys.argv[1]))
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256+n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
u2b = {v: k for k, v in bytes_to_unicode().items()}
vocab = j["model"]["vocab"]
toks = {}
for t, i in vocab.items():
    toks[i] = bytes(u2b[c] for c in t)
added = {a["id"]: (a["content"], a.get("special")) for a in j["added_tokens"]}
print("vocab", len(vocab), "max id", max(vocab.values()), "added", len(added), "max added", max(added))
for i in sorted(added)[:40]:
    print(i, added[i])
nb = sum(len(b) for b in toks.values())
print("total bytes", nb, "avg", nb / len(toks))
start_q = [b for b in toks.values() if b.startswith(b'"')]
has_q = [b for b in toks.values() if b'"' in b]
has_bs = [b for b in toks.values() if b'\\' in b]
digit = [b for b in toks.values() if b[:1].isdigit()]
print("start quote", len(start_q), "has quote", len(has_q), "has backslash", len(has_bs), "start digit", len(digit))
print("digit examples", sorted(digit, key=len)[-10:])
sp = [b for b in toks.values() if b.startswith(b' ')]
nl = [b for b in toks.values() if b.startswith(b'\n')]
print("start space", len(sp), "start nl", len(nl))
nodes = set()
for b in toks.values():
    for k in range(1, len(b)+1):
        nodes.add(b[:k])
print("trie nodes", len(nodes))
qn = set(x for x in nodes if x.startswith(b'"'))
print("trie nodes under quote", len(qn))
suf = set()
for b in toks.values():
    k = b.find(b'"')
    if k >= 0: suf.add(b[k+1:])
print("distinct suffixes after first quote", len(suf))
ws = [b for b in toks.values() if b and all(c in b' \t\n\r' for c in b)]
print("pure ws tokens", len(ws))
lt = [b for b in toks.values() if b'<' in b]
print("has <", len(lt), "has nl", len([b for b in toks.values() if b'\n' in b]))
dup = {}
for i, b in toks.items():
    dup.setdefault(b, []).append(i)
print("duplicate byte strings", sum(1 for v in dup.values() if len(v) > 1))
