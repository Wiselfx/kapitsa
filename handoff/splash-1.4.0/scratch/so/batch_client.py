"""Two requests under a JSON schema and two free ones at once, three times (greedy, then sampled); each constrained reply
must be valid. usage: batch_client.py URL"""
import json, sys, threading, time, urllib.request
import jsonschema

URL = sys.argv[1]
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "city": {"type": "string"}, "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "bio": {"type": "string"}},
    "required": ["name", "age", "city", "hobbies", "bio"], "additionalProperties": False}
v = jsonschema.Draft202012Validator(PERSON)
fails = 0
for rnd, temp in enumerate([0, 0.7, 0.7]):
    out = {}
    def one(k):
        body = {"model": "m", "max_tokens": 384, "temperature": temp, "seed": rnd * 10 + k,
                "chat_template_kwargs": {"enable_thinking": False}}
        if k < 2:
            body["messages"] = [{"role": "user", "content": ["Make up a person from Rome.", "Make up a person from Seoul."][k]}]
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "p", "schema": PERSON}}
        else:
            body["messages"] = [{"role": "user", "content": ["Write a poem about rain.", "Explain TCP slow start."][k - 2]}]
        req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        out[k] = (json.loads(urllib.request.urlopen(req, timeout=600).read()), time.time() - t0)
    th = [threading.Thread(target=one, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    for k in range(4):
        r, wall = out[k]
        c = r["choices"][0]
        ok = True
        if k < 2:
            try:
                ok = not list(v.iter_errors(json.loads(c["message"]["content"])))
            except ValueError:
                ok = False
        fails += not ok
        print("round %d T=%.1f request %d (%s): %s, %d tokens in %.1f s, finish %s" % (
            rnd, temp, k, "schema" if k < 2 else "free", "valid" if ok else "INVALID " + repr(c["message"]["content"])[:200],
            r["usage"]["completion_tokens"], wall, c["finish_reason"]))
print("ALL VALID" if not fails else "INVALID: %d" % fails)
sys.exit(1 if fails else 0)
