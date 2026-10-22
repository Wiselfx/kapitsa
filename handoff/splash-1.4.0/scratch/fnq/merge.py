#!/usr/bin/env python3
"""After the two converter instances: files 28-50 from the second (work/q20b) moved into the pack, the pack's done
list completed, so a rerun of the first converter only adds the MTP head, the table, the small files, the index and
the conversion record."""
import json
import os

OUT = os.path.expanduser("~/.cache/kapitsa/models/qwen3.8-flash-next-q20")
B = os.path.expanduser("~/.cache/kapitsa/work/q20b")
done_b = set(json.load(open(os.path.join(B, ".q20.done.json"))))
moved = []
for i in range(28, 51):
    f = "model-%05d-of-00050.safetensors" % i
    if f not in done_b:
        raise SystemExit(f"{f} is not finished in {B}")
    os.replace(os.path.join(B, f), os.path.join(OUT, f))
    moved.append(f)
done_a = set(json.load(open(os.path.join(OUT, ".q20.done.json"))))
missing = [f"model-{i:05d}-of-00050.safetensors" for i in range(1, 28)
           if f"model-{i:05d}-of-00050.safetensors" not in done_a]
if missing:
    raise SystemExit(f"not finished in {OUT}: {missing}")
done = sorted(done_a | set(moved))
json.dump(done, open(os.path.join(OUT, ".q20.done.json"), "w"))
print(f"moved {len(moved)} files; {OUT} done list: {len(done)} entries")
