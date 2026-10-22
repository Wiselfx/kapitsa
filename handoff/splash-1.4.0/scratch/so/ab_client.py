"""Sends a fixed set of requests one at a time to a running server; writes each reply (content, finish, tokens) as JSON.
usage: ab_client.py URL OUT.json MODE   MODE: free (no format) | schema (response_format json_schema)"""
import json, sys, time, urllib.request

URL, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]

FREE = [
    "Write a Python function that parses an ISO 8601 date string without using datetime, with docstring and tests.",
    "Explain how a hash map handles collisions, with a small C implementation using open addressing.",
    "Write a bash script that backs up a directory to a timestamped tarball and keeps the last 7 backups.",
]
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150}, "email": {"type": "string"},
    "address": {"type": "object", "properties": {"street": {"type": "string"}, "city": {"type": "string"},
                                                  "country": {"type": "string"}},
                "required": ["street", "city", "country"], "additionalProperties": False},
    "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
    "bio": {"type": "string"}},
    "required": ["name", "age", "email", "address", "hobbies", "bio"], "additionalProperties": False}
RECIPE = {"type": "object", "properties": {
    "title": {"type": "string"}, "servings": {"type": "integer", "minimum": 1, "maximum": 20},
    "ingredients": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {
        "item": {"type": "string"}, "quantity": {"type": "number", "minimum": 0},
        "unit": {"enum": ["g", "kg", "ml", "l", "cup", "tbsp", "tsp", "piece"]}},
        "required": ["item", "quantity", "unit"], "additionalProperties": False}},
    "steps": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["title", "servings", "ingredients", "steps"], "additionalProperties": False}
SCHEMA = [("Make up a detailed fictional person living in Lisbon, with a long bio.", PERSON),
          ("A detailed recipe for a vegetable lasagna for six people.", RECIPE),
          ("Make up a detailed fictional software engineer living in Oslo, with a long bio.", PERSON),
          ("A detailed recipe for a Thai green curry.", RECIPE)]

reqs = []
if MODE == "free":
    for p in FREE:
        reqs.append({"messages": [{"role": "user", "content": p}], "temperature": 0})
    reqs.append({"messages": [{"role": "user", "content": FREE[0]}], "temperature": 0.7, "seed": 5})
else:
    for p, sch in SCHEMA:
        reqs.append({"messages": [{"role": "system", "content": "You answer in JSON."}, {"role": "user", "content": p}],
                     "temperature": 0, "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": sch}}})
    reqs.append(dict(reqs[0], temperature=0.7, seed=5))
out = []
for k, r in enumerate(reqs):
    body = dict(r, model="m", max_tokens=512, chat_template_kwargs={"enable_thinking": False})
    t0 = time.time()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    j = json.loads(urllib.request.urlopen(req, timeout=600).read())
    c = j["choices"][0]
    out.append({"k": k, "content": c["message"].get("content"), "finish": c["finish_reason"],
                "tokens": j["usage"]["completion_tokens"], "wall": time.time() - t0})
json.dump(out, open(OUT, "w"), indent=1)
