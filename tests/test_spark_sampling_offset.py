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



def check_kv_regrow():
    """A grown full-attention cache must copy each KV head to the NEW row stride.

    Spark passed kv_regrow `{old_words, old_words, target_words}`, the layout kv_copy_prefix takes. kv_regrow reads
    (source stride, destination stride), so head 1 landed at the old offset and every later head read zeros after the
    first reallocation -- i.e. any conversation longer than the initial 1024-token allocation.
    """
    spark = (ROOT / "src/model/spark.cpp").read_text()
    m = re.search(r"const uint32_t p\[4\] = \{([^}]*)\}", spark)
    check(m is not None, "Spark::reserve builds the kv_regrow params")
    if not m:
        return
    args = [a.strip() for a in m.group(1).split(",")]
    check(len(args) >= 2 and "target" in args[1],
          "kv_regrow's destination stride is the new capacity, not the old one: p[1] = %r" % (args[1] if len(args) > 1 else ""))


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
    check("multi_proj(e, gu, 2, x_, L.post_norm)" in b,
          "gate/up share one multi-segment dispatch (falls back to two matvecs on mixed formats)")
    calls = re.findall(r"matvec\((L\.m[gud]),\s*([A-Za-z0-9_]+),", b)
    check(len(calls) == 3, "the fallback still reads all three matrices (found %d: %s)" % (len(calls), calls))
    inputs = dict(calls)
    check(inputs.get("L.mg") == "x_", "gate_proj reads x_ (got %s)" % inputs.get("L.mg"))
    check(inputs.get("L.mu") == "x_", "up_proj reads x_ (got %s)" % inputs.get("L.mu"))
    check(inputs.get("L.md") == "mh_", "down_proj reads mh_, the gate*up activation (got %s)" % inputs.get("L.md"))
    check("e.buf(2, mh_)" in b, "the GELU writes mh_")
    # Decode matvecs read fp32 x, so the decode epilogue must write fp32. Prefill's down_proj is a qrf GEMM and reads
    # half; that path uses spark_gelu_gu_half and no longer needs a separate rows_to_half pass. Checked statically because
    # no test here runs a GPU.
    metal = (ROOT / "src/kernels/spark/spark.metal").read_text()
    check("device float* out" in metal, "spark_gelu_gu writes fp32 for the decode matvec")
    check("spark_gelu_gu_half" in metal, "prefill has a half epilogue for qrf")
    check("spark_gelu_gu_half" in spark and "enc_rows_half(e, gate" not in spark,
          "enc_mlp_prefill writes hh_ directly instead of converting gate")


def check_qkv_segments():
    """q/k/v are three row ranges of one fused matrix, so each multi-segment view starts at its own row.

    The first version bound q_k_v_proj's w/s/b unchanged for all three segments. The multi kernel then multiplied the
    q rows three times, labelled the results k and v, and produced wrong attention with no crash. Only the offset views
    at 0, qd, qd+kd make the segments the matrices they claim to be.
    """
    spark = (ROOT / "src/model/spark.cpp").read_text()
    m = re.search(r"ProjSeg proj\[4\] = \{(.*?)\};", spark, re.S)
    check(m is not None, "enc_attn builds the q/k/v/gate segment table")
    if not m:
        return
    segs = m.group(1)
    for what, rows in (("{&L.qkv, stg_, 0, uint32_t(qd)}", "q rows"),
                       ("{&L.qkv, kr_, uint32_t(qd), uint32_t(kd)}", "k rows"),
                       ("{&L.qkv, vr_, uint32_t(qd + kd), uint32_t(kd)}", "v rows")):
        check(what in segs, "qkv segment at its own row and length (%s): %s" % (rows, what))
    check("{&L.gate, gate_, 0, uint32_t(L.gate.N)}" in segs, "g_proj is a whole matrix sharing the input")
    check("e.buf(i * 4 + 0, W.w->buf, size_t(segs[i].row) * w_row)" in spark,
          "multi_proj binds each segment at its row offset, not the matrix start")
    check("const uint32_t nrows = segs[i].n ? segs[i].n : W.N" in spark,
          "multi_proj dispatches the segment's own row count, not W.N for every slice")
    check("if (!multi_proj(e, proj, 4" in spark and "if (!multi_proj(e, proj, 3" in spark,
          "four segments when gate shares a format, three otherwise")


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
        check(len(calls) == 1, "Spark::encode_prefill has exactly one enc_sample call (found %d)" % len(calls))
        # The invariant is NOT the literal "pos + 1" -- that was encode_prefill's spelling while it ran one row at a
        # time. Batched, the pass starts at p0 and covers n rows, so the slot after the last row is p0 + n, which is the
        # same position (pos0 + T) and the same requirement: the sample must NOT overwrite the row it just embedded, and
        # must land one past it. Test the property, not the arithmetic's spelling -- matching the string made this check
        # fail on a correct rewrite, and would have passed on an incorrect one that kept the words.
        for c in calls:
            m = re.search(r"enc_sample\(e, s,\s*(.+?),\s*sp,", c, re.S)
            arg = m.group(1).strip() if m else ""
            check(arg not in ("pos", "p0", "pos0"),
                  "Spark::encode_prefill samples past its last row, not into it: enc_sample arg is %r" % arg)
            check(arg.endswith("+ n") or arg.endswith("+ 1"),
                  "Spark::encode_prefill's sample slot is the row after the pass's last: enc_sample arg is %r" % arg)
        # and the pass must embed exactly the rows it claims (n = the pass length), so `p0 + n` is the row after it
        check("enc_embed(e, s.tok, p0, n," in pre.group(0),
              "Spark::encode_prefill embeds n rows from p0, so p0 + n is the slot after the last embedded row")

    # the capacity guard must allow that extra slot. Matched on the guard statement itself: a bare "pos + 2" search
    # passes on the comment above it, which is how this check was vacuous the first time it was written.
    if body:
        guard = re.search(r"if \(pos \+ (\d) > s\.cap", body.group(0))
        check(guard is not None and guard.group(1) == "2",
              "Spark::encode_step's capacity guard allows the pos + 1 slot it writes (if (pos + 2 > s.cap))")

    check_kv_regrow()
    check_qkv_segments()
    check_mlp_inputs()

    print()
    if fails:
        print(f"test_spark_sampling_offset: {len(fails)} check(s) failed")
        return 1
    print("test_spark_sampling_offset: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
