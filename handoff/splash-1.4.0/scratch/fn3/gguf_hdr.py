#!/usr/bin/env python3
"""gguf_hdr.py FILE: tensor types and byte totals of a GGUF from its header (the first MBs are enough)."""
import struct, sys
from collections import defaultdict
f = open(sys.argv[1], "rb")
def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s(): n = u64(); return f.read(n).decode("utf-8", "replace")
SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def val(t):
    if t == 8: return s()
    if t == 9:
        at = u32(); n = u64()
        if at == 8:
            return [s() for _ in range(n)] if n < 64 else (lambda: ([s() for _ in range(n)], f"<{n} strings>")[1])()
        f.read(SZ[at] * n); return f"<{n} x type {at}>"
    fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
    return struct.unpack(fmt, f.read(SZ[t]))[0]
assert f.read(4) == b"GGUF"
ver = u32(); nt = u64(); nkv = u64()
kv = {}
for _ in range(nkv):
    k = s(); t = u32(); kv[k] = val(t)
# type sizes: (block elements, block bytes)
TS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 20: (32, 18),
      10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292), 16: (256, 66),
      17: (256, 74), 18: (256, 98), 19: (256, 50), 21: (256, 110), 22: (256, 82), 23: (256, 136), 29: (256, 56),
      39: (32, 17), 40: (64, 36), 42: (64, 18)}
NAMES = {0: "F32", 1: "F16", 30: "BF16", 8: "Q8_0", 2: "Q4_0", 20: "IQ4_NL", 42: "Q2_0", 12: "Q4_K", 14: "Q6_K", 13: "Q5_K"}
tot = defaultdict(int); cnt = defaultdict(int); ex = {}
for _ in range(nt):
    name = s(); nd = u32(); dims = [u64() for _ in range(nd)]; t = u32(); off = u64()
    n = 1
    for d in dims: n *= d
    be, bb = TS.get(t, (1, 0))
    b = n // be * bb
    key = NAMES.get(t, f"type{t}")
    kind = "experts" if "_exps" in name else ("ple_table" if name == "per_layer_token_embd.weight" else "other")
    tot[(key, kind)] += b; cnt[(key, kind)] += 1
    ex.setdefault((key, kind), name + str(dims))
print(f"GGUF v{ver}: {nt} tensors, {nkv} keys; split {kv.get('split.count')} / {kv.get('split.tensors.count')}; arch {kv.get('general.architecture')}")
for k in sorted(tot, key=lambda k: -tot[k]):
    print(f"  {k[0]:7} {k[1]:9} {cnt[k]:5} tensors {tot[k] / 1e9:8.3f} GB  e.g. {ex[k]}")
