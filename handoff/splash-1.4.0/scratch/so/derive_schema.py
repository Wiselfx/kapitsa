"""From the greedy unconstrained replies of session 8 (JSON cut at 512 tokens), a closed JSON Schema per prompt that the
reply follows: every object's keys declared (additionalProperties false, the keys present required), arrays' items
merged, numbers as numbers. Under it the constrained greedy reply should be the same tokens.
usage: derive_schema.py OUT_JSON_OF_SESSION8 SCHEMAS.json"""
import json, sys


def close(text):
    """Completes a JSON text cut anywhere: closes an open string, drops a dangling key or separator, closes brackets."""
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            stack.pop()
    t = text
    if in_str:
        if esc:
            t = t[:-1]
        t += '"'
    for _ in range(4):
        s = t.rstrip()
        if s.endswith(",") or s.endswith(":"):
            t = s[:-1]
        try:
            json.loads(t + "".join("}" if c == "{" else "]" for c in reversed(stack)))
            break
        except ValueError:
            # a key without its value: drop the key
            s = t.rstrip()
            if s.endswith('"'):
                k = s[:-1].rfind('"')
                t = s[:k].rstrip()
    return json.loads(t + "".join("}" if c == "{" else "]" for c in reversed(stack)))


def schema(v):
    if isinstance(v, dict):
        return {"type": "object", "properties": {k: schema(x) for k, x in v.items()}, "required": list(v),
                "additionalProperties": False}
    if isinstance(v, list):
        if not v:
            return {"type": "array"}
        merged = schema(v[0])
        for x in v[1:]:
            merged = merge(merged, schema(x))
        return {"type": "array", "items": merged}
    if isinstance(v, bool):
        return {"type": "boolean"}
    if isinstance(v, (int, float)):
        return {"type": "number"}
    if v is None:
        return {"type": "null"}
    return {"type": "string"}


def merge(a, b):
    if a.get("type") == "object" and b.get("type") == "object":
        props = dict(a["properties"])
        for k, s in b["properties"].items():
            props[k] = merge(props[k], s) if k in props else s
        req = [k for k in a["required"] if k in b["required"]]
        return {"type": "object", "properties": props, "required": req, "additionalProperties": False}
    if a.get("type") == "array" and b.get("type") == "array":
        if "items" in a and "items" in b:
            return {"type": "array", "items": merge(a["items"], b["items"])}
        return a if "items" in a else b
    if a == b:
        return a
    return {"anyOf": [a, b]}


reqs = json.load(open(sys.argv[1]))
out = {}
for r in reqs:
    if r["kind"] == "json-free":
        out[r["label"]] = schema(close(r["content"]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, s in out.items():
    print(k, json.dumps(s)[:300])
