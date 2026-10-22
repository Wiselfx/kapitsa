"""The merge's GPU check, one server: a request alone (unconstrained, greedy and sampled), then requests at once:
round 0 two under a JSON schema and two free, all greedy; round 1 the same with the constrained ones sampled; round 2
four free, greedy. Writes every reply; checks each constrained reply against its schema (on main the format is ignored:
not checked). usage: gpu_check_client.py URL OUT.json CHECK(0|1)"""
import json, sys, threading, time, urllib.request
import jsonschema

URL, OUT, CHECK = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
PERSON = {"type": "object", "properties": {
    "name": {"type": "string"}, "age": {"type": "integer", "minimum": 0, "maximum": 150},
    "city": {"type": "string"}, "hobbies": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
    "bio": {"type": "string"}},
    "required": ["name", "age", "city", "hobbies", "bio"], "additionalProperties": False}
v = jsonschema.Draft202012Validator(PERSON)
out = {"alone": [], "rounds": []}
fails = 0


def post(body, timeout=150):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def reply(j):
    c = j["choices"][0]
    m = c["message"]
    calls = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
    return {"content": m.get("content"), "calls": calls, "finish": c["finish_reason"],
            "tokens": j["usage"]["completion_tokens"]}


base = {"model": "m", "chat_template_kwargs": {"enable_thinking": False}}
for k, sp in enumerate([{"temperature": 0}, {"temperature": 0.7, "seed": 3}]):
    j = post(dict(base, max_tokens=512, messages=[{"role": "user", "content": "Write a Python function that merges "
                  "overlapping intervals, with a docstring and three doctests."}], **sp))
    out["alone"].append(reply(j))
    print("alone %d: %d tokens, %s" % (k, out["alone"][-1]["tokens"], out["alone"][-1]["finish"]))

FREE = ["Write a short poem about rain on a tin roof.", "Explain TCP slow start in a few paragraphs."]
CON = ["Make up a person from Rome.", "Make up a person from Seoul."]
for rnd in range(3):
    res = {}

    def one(k):
        body = dict(base, max_tokens=384)
        if rnd < 2 and k < 2:
            body["messages"] = [{"role": "user", "content": CON[k]}]
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "p", "schema": PERSON}}
            body.update({"temperature": 0.7, "seed": 10 + k} if rnd == 1 else {"temperature": 0})
        else:
            body["messages"] = [{"role": "user", "content": (FREE + FREE)[k] if rnd < 2 else (FREE + CON)[k]}]
            body["temperature"] = 0
        t0 = time.time()
        res[k] = (reply(post(body)), time.time() - t0, "response_format" in body)

    th = [threading.Thread(target=one, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
        time.sleep(0.05)
    for t in th:
        t.join()
    rows = []
    for k in range(4):
        r, wall, con = res[k]
        ok = None
        if con and CHECK:
            try:
                doc = json.loads(r["content"])
                ok = not list(v.iter_errors(doc))
            except (ValueError, TypeError):
                ok = False
            if not ok:
                fails += 1
        rows.append(dict(r, constrained=con, valid=ok, wall=wall))
        print("round %d request %d (%s): %d tokens in %.1f s, %s%s" % (
            rnd, k, "schema" if con else "free", r["tokens"], wall, r["finish"],
            "" if ok is None else (", valid" if ok else ", INVALID: " + repr((r["content"] or "")[:200]))))
    out["rounds"].append(rows)
json.dump(out, open(OUT, "w"), indent=1)
print("ALL VALID" if fails == 0 else "%d INVALID" % fails)
sys.exit(1 if fails else 0)
