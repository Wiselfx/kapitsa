"""Sends a fixed set of requests one at a time to a running server and writes each reply as JSON (the server's log has
the timings: one "chat: prompt" line per request, in this order).
usage: perf_client.py URL OUT.json MODE CORPUS
  free:  a warm-up, then 4 unconstrained replies of 512 tokens (3 greedy, 1 sampled) and 2 cold prompts (~4K, ~16K
         tokens of the harness corpus, a nonce first) for prefill
  pairs: a warm-up, then the same request unconstrained and constrained: 4 prompts with and without a JSON schema,
         2 tool prompts with tool_choice auto and required (orders alternate)"""
import json, random, sys, time, urllib.request

URL, OUT, MODE, CORPUS = sys.argv[1:5]
corpus = open(CORPUS, encoding="utf-8", newline="").read()

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
WRITE_FILE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a text file in the workspace.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                   "required": ["path", "content"]}}}
ADD_CONTACTS = {"type": "function", "function": {
    "name": "add_contacts", "description": "Add people to the address book.",
    "parameters": {"type": "object", "properties": {"contacts": {"type": "array", "items": {
        "type": "object", "properties": {"name": {"type": "string"}, "phone": {"type": "string"},
                                         "email": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "phone", "email", "age"]}}}, "required": ["contacts"]}}}
TOOLS = [("Write a Python module implementing a binary min-heap class with push, pop, peek and heapify, with "
          "docstrings, and save it as heap.py.", [WRITE_FILE]),
         ("Make up eight people with names, phone numbers, emails and ages and add them to my address book.",
          [ADD_CONTACTS])]


def post(body):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


out = []


def send(kind, label, body):
    body = dict(body, model="m", chat_template_kwargs={"enable_thinking": False})
    t0 = time.time()
    j = post(body)
    c = j["choices"][0]
    m = c["message"]
    out.append({"kind": kind, "label": label, "content": m.get("content"), "tool_calls": m.get("tool_calls"),
                "finish": c["finish_reason"], "tokens": j["usage"]["completion_tokens"],
                "prompt_tokens": j["usage"]["prompt_tokens"], "wall": time.time() - t0})


def nonce_messages(chars, offset, question):
    n = "session %016x" % random.getrandbits(64)
    return [{"role": "system", "content": "[%s] You are a helpful assistant.\n\n# Context\n%s" % (n, corpus[offset:offset + chars])},
            {"role": "user", "content": question}]


# warm-up: a ~2.7K-token prompt from a part of the corpus no timed prompt uses (more than one prefill chunk)
send("warm", "warm", {"messages": nonce_messages(11475, 100000, "Say OK."), "max_tokens": 8, "temperature": 0})
if MODE == "free":
    for k, p in enumerate(FREE):
        send("decode", "greedy %d" % k, {"messages": [{"role": "user", "content": p}], "max_tokens": 512, "temperature": 0})
    send("decode", "sampled", {"messages": [{"role": "user", "content": FREE[0]}], "max_tokens": 512,
                               "temperature": 0.7, "seed": 5})
    for size in (4000, 16000):
        send("prefill", "%dK" % (size // 1000), {"messages": nonce_messages(int(size * 4.25), 0,
                                                 "In one sentence, what is the first file in the context about?"),
                                                 "max_tokens": 16, "temperature": 0})
elif MODE == "derived":
    # the JSON prompts under a closed schema made from their unconstrained greedy reply (derive_schema.py)
    import os
    derived = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "derived_schemas.json")))
    for k, (p, sch) in enumerate(SCHEMA):
        base = {"messages": [{"role": "system", "content": "You answer in JSON."}, {"role": "user", "content": p}],
                "max_tokens": 512, "temperature": 0}
        con = dict(base, response_format={"type": "json_schema", "json_schema": {
            "name": "x", "schema": derived["prompt %d" % k], "strict": True}})
        pair = [("json-free", base), ("json-schema", con)]
        for kind, b in (pair if k % 2 == 0 else pair[::-1]):
            send(kind, "prompt %d" % k, b)
elif MODE == "same":
    # requests whose greedy reply is already valid unconstrained, so that the constrained reply is the same tokens:
    # the JSON prompts under json_object, a tool call the model makes anyway under tool_choice required
    for k, (p, sch) in enumerate(SCHEMA):
        base = {"messages": [{"role": "system", "content": "You answer in JSON."}, {"role": "user", "content": p}],
                "max_tokens": 512, "temperature": 0}
        pair = [("json-free", base), ("json-object", dict(base, response_format={"type": "json_object"}))]
        for kind, b in (pair if k % 2 == 0 else pair[::-1]):
            send(kind, "prompt %d" % k, b)
    for k, n in enumerate(("eight", "twenty")):
        base = {"messages": [{"role": "user", "content": "Make up %s people with names, phone numbers, emails and ages and "
                                                          "add them to my address book." % n}],
                "tools": [ADD_CONTACTS], "max_tokens": 1024, "temperature": 0}
        pair = [("tool-auto", dict(base, tool_choice="auto")), ("tool-required", dict(base, tool_choice="required"))]
        for kind, b in (pair if k % 2 == 0 else pair[::-1]):
            send(kind, "tool %d" % k, b)
else:
    for k, (p, sch) in enumerate(SCHEMA):
        base = {"messages": [{"role": "system", "content": "You answer in JSON."}, {"role": "user", "content": p}],
                "max_tokens": 512, "temperature": 0}
        con = dict(base, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": sch}})
        pair = [("schema-free", base), ("schema", con)]
        for kind, b in (pair if k % 2 == 0 else pair[::-1]):
            send(kind, "prompt %d" % k, b)
    for k, (p, tools) in enumerate(TOOLS):
        base = {"messages": [{"role": "user", "content": p}], "tools": tools, "max_tokens": 1024, "temperature": 0}
        pair = [("tool-auto", dict(base, tool_choice="auto")), ("tool-required", dict(base, tool_choice="required"))]
        for kind, b in (pair if k % 2 == 0 else pair[::-1]):
            send(kind, "tool %d" % k, b)
json.dump(out, open(OUT, "w"), indent=1)
print("%s: %d requests" % (MODE, len(out)))
