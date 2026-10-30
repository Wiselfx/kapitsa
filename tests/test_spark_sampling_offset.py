#!/usr/bin/env python3
"""Two conventions no CPU test here would otherwise catch: where a sampled token goes, and what down_proj reads.

This is the bug that made Spark reply with one token repeated forever. Spark::encode_step called
`enc_sample(..., pos, ...)`, which is `tokens[out_index] = chosen` (sample.metal) -- so each step overwrote the token it
had just embedded. The engine reads `s.tokens()[pos]` afterwards to get the model's output, so it read back the prompt's
own token, and the next step re-embedded a slot nothing had written. After the first step the input never changed again:
deterministic, plausible-looking, completely wrong, and invisible to every CPU test.

Qwen35::encode_step writes to `pos + 1` (qwen35.cpp:2439) and this is the single line where the two differed. Checked
statically because no test here runs a GPU.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)



def check_mlp_inputs():
    """gate/up read x_ (and normalize it); down_proj must read the gate*up activation, not the residual.

    Spark::enc_mlp's inner matvec lambda hard-coded buffer 3 to x_ for all three projections, so down_proj multiplied
    the residual and the GELU output was never used -- every layer's MLP degenerated to a linear map with no activation.
    The result was fluent-looking nonsense (garbled tokens in several scripts), not a crash, so nothing caught it.
    """
    spark = (ROOT / "src/model/spark.cpp").read_text()
    body = re.search(r"void Spark::enc_mlp\(.*?\n\}", spark, re.S)
    check(body is not None, "Spark::enc_mlp found")
    if not body:
        return
    b = body.group(0)
    calls = re.findall(r"matvec\((L\.m[gud]),\s*([A-Za-z0-9_]+),", b)
    check(len(calls) == 3, "enc_mlp issues three matvecs (found %d: %s)" % (len(calls), calls))
    inputs = dict(calls)
    check(inputs.get("L.mg") == "x_", "gate_proj reads x_ (got %s)" % inputs.get("L.mg"))
    check(inputs.get("L.mu") == "x_", "up_proj reads x_ (got %s)" % inputs.get("L.mu"))
    check(inputs.get("L.md") == "mh_", "down_proj reads mh_, the gate*up activation (got %s)" % inputs.get("L.md"))
    check("mh_" in b, "the GELU's output buffer (mh_) is written")


def main():
    spark = (ROOT / "src/model/spark.cpp").read_text()
    qwen = (ROOT / "src/model/qwen35.cpp").read_text()
    sample = (ROOT / "src/kernels/sample.metal").read_text()

    # the kernel's contract, which the call sites must match
    check("tokens[p.out_index] = int(chosen)" in sample,
          "sample_stage2 writes tokens[out_index] (so out_index is the slot the NEXT step reads)")

    # the reference: qwen35 adds one
    m = re.search(r"void Qwen35::encode_step\(.*?\n\}", qwen, re.S)
    check(m is not None, "Qwen35::encode_step found")
    if m:
        check("enc_sample(e, s, x_, pos + 1," in m.group(0),
              "Qwen35::encode_step samples into pos + 1 (the convention Spark must follow)")

    # spark: every sampling call site, and none may pass a bare `pos`
    body = re.search(r"void Spark::encode_step\(.*?\n\}", spark, re.S)
    check(body is not None, "Spark::encode_step found")
    if body:
        calls = re.findall(r"enc_sample\([^;]*\);", body.group(0))
        check(len(calls) == 1, "Spark::encode_step has exactly one enc_sample call (found %d)" % len(calls))
        for c in calls:
            check("pos + 1" in c, "Spark::encode_step samples into pos + 1, not pos: %s" % c.strip())

    pre = re.search(r"void Spark::encode_prefill\(.*?\n\}", spark, re.S)
    check(pre is not None, "Spark::encode_prefill found")
    if pre:
        calls = re.findall(r"enc_sample\([^;]*\);", pre.group(0))
        for c in calls:
            check("pos + 1" in c, "Spark::encode_prefill's last row samples into pos + 1: %s" % c.strip())

    # the capacity guard must allow that extra slot. Matched on the guard statement itself: a bare "pos + 2" search
    # passes on the comment above it, which is how this check was vacuous the first time it was written.
    if body:
        guard = re.search(r"if \(pos \+ (\d) > s\.cap", body.group(0))
        check(guard is not None and guard.group(1) == "2",
              "Spark::encode_step's capacity guard allows the pos + 1 slot it writes (if (pos + 2 > s.cap))")

    check_mlp_inputs()

    print()
    if fails:
        print(f"test_spark_sampling_offset: {len(fails)} check(s) failed")
        return 1
    print("test_spark_sampling_offset: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
