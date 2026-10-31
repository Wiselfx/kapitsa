# Spark-X2.6 in Kapitsa — the working plan

**READ THIS FIRST AFTER ANY CONTEXT COMPACTION.** If you are an agent resuming this task, or you think you are missing
information, read this file top to bottom before touching a file. It is the single source of truth for the Spark port.
Section 9 is the state log: it says exactly what is done, what is next, and what must not be claimed.

---

## 1. What we are doing

Port **Spark-X2.6-1.7B** and **Spark-X2.6-4B** (XHToken) into Kapitsa, the native Apple-silicon engine that runs Qwen,
Nemotron and Flint models. Both are dense transformers in the `spark2_5` family; they are architecturally simpler than
anything Kapitsa already runs (no GDN, no MoE, no Mamba, no sparse indexer), which is why this is worth doing.

The goal is a new `Spark` model backend in `src/model/`, reusing Kapitsa's existing generic kernels wherever their math
matches, plus a small number of new Metal kernels where Spark's math genuinely differs. Speed is the product; a change
that is faster but produces different tokens is a different change and must say so (`AGENTS.md`).

## 2. Hard constraints (do not forget these)

- **No Apple hardware on the dev box.** Development happens on Linux aarch64: no Metal, no Xcode, no Apple SDK. A local
  `cmake --build` of `kapitsa` **cannot** succeed and must not be attempted as proof of anything.
- **The author does not run a GPU.** Per repo precedent (AGENTS.md, PR #170), the PR is marked **"Not verified on a GPU"**
  in both title and description. The user runs the Mac verification pass.
- **CI does the compile checking.** GitHub Actions `macos-15` runners build the C++ and, via `metal-compile`, run
  `xcrun metal` over the embedded kernel libraries. That is the only automated signal about the Metal sources here.
  Note `metal-compile` cannot see wrong arithmetic, races, or the 32 KB threadgroup budget.
- **One change per PR**, ~1000 changed lines max (`pr-size.yml`). This port is deliberately kept in one coherent PR to
  the fork; see §7 for the upstream-split plan.
- **We push to the fork `Wiselfx/kapitsa`**, never upstream `nace-ai/kapitsa`. Never merge, never use admin bypasses.
- **Never write a speedup number, "bit for bit", or "same outputs"** for anything not run on a GPU.
- **Do not commit** `build/`, models, `*.bin`, agent plans or personal settings. `docs/spark/` is a deliberate exception:
  it is project documentation required by the user, and is described in §8.

## 3. The two models, as facts

Source of truth: `docs/spark/configs/*.json` (verbatim copies of the HF `config.json`) and `docs/spark/configs/NOTES.md`.

| | 1.7B | 4B |
|---|---|---|
| HF repo | `XHToken/Spark-X2.6-1.7B` | `XHToken/Spark-X2.6-4B` |
| `model_type` | `spark2_5` | `spark2_5` |
| `architectures` | `Spark2_5ForCausalLM` | `Spark2_5ForCausalLM` |
| params | 1,707,657,216 | 4,112,079,360 |
| hidden_size | 2048 | 2560 |
| intermediate_size | 6656 | 10240 |
| layers | 28 | 36 |
| q heads / kv heads | 8 / 2 (**G=4**) | 16 / 4 (**G=4**) |
| head_dim | 256 | 256 |
| layer pattern | (S,S,S,F)×7, 21 S + 7 F | (S,S,S,F)×9, 27 S + 9 F |
| sliding_window | 512 | 512 |
| MLP act | `gelu` (exact, erf) | `gelu` (exact, erf) |
| attn output gate | **per head**, sigmoid | **per head**, sigmoid |
| RoPE full layers | θ=5e6, `partial_rotary_factor` 0.25 → 64 rot dims | same |
| RoPE sliding layers | θ=1e4, `partial_rotary_factor` 1.0 → 256 rot dims | same |
| vocab / tied | 131072 / yes | 131072 / yes |
| q_norm / k_norm | **absent** | **absent** |
| MLP/attn bias | none | none |
| max_position_embeddings | 1048576 | 1048576 |

**Weight keys (identical set for both, 290 keys for the 4B):**
`model.embedding.weight`, `model.norm.weight`, and per layer `model.layers.N.input_layernorm.weight`,
`model.layers.N.post_attention_layernorm.weight`, `model.layers.N.self_attn.q_k_v_proj.weight`,
`model.layers.N.self_attn.g_proj.weight`, `model.layers.N.self_attn.out_proj.weight`,
`model.layers.N.mlp.{gate,up,down}_proj.weight`. Note the **fused `q_k_v_proj`** (one matrix, `[q_dim+2*kv_dim, hidden]`)
and the **`g_proj` of shape `[num_heads]`** (8 for the 1.7B, 16 for the 4B).

Per-head gate: `attn_out *= sigmoid(g_proj(x))` where the scalar is broadcast over all 256 head dims.
This is `sigmoid`, not silu, and it is **per head**, not per channel. Kapitsa's existing kernels apply a *per-channel*
gate vector, so this needs a broadcast mode.

### Available quantized packs (no conversion needed)

`mlx-community/Spark-X2.6-4B-OptiQ-4bit` is MLX affine, group 64, 3.04 GB, with **per-tensor mixed bits**
(8-bit on `q_k_v_proj`/`g_proj`/`out_proj`/`embedding`, mostly 4-bit MLP). Kapitsa's `Qwen35::quant_of()` already reads
exactly that per-tensor `{bits, group_size}` map, so pointing Kapitsa at the repo should load without repacking.
Others: `abenzerps/...-MLX-{4,6,8}bit`, `0x36/...-mlx-{4bit,8bit,bf16}`, `hermitdave/...-SLQ-4.50bpw`,
`code-and-canvas/...-oQ{4,6,8}e-fp16`, `Jakevin/...-gptq4-mlx`, and the official `XHToken/...-{INT4,INT8,FP8,GGUF}`.
Kapitsa's Spark formats (`format_supported`, `src/model/spark.cpp`): bf16; group 64 at 3/4/5/6/8 bit; group 32 at
4 and 8 bit (Spark-MLX-LLM's default profile). Group-32 decode goes through the format-generic `qmvg_f4g32`/`f8g32`
kernels because the `qmv2_b*_g64`/`qmv3_b*` kernels hard-code a 64-value scale group.

## 4. Architecture decisions (settled — do not re-litigate without evidence)

**D1. A new model class `Spark`, not a config of `Qwen35`.**
`Qwen35` hard-requires the GDN fields (`linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`,
`linear_value_head_dim` — read unconditionally at the top of `read_config`) and unconditionally loads
`q_norm.weight`/`k_norm.weight` per attention layer. Spark has neither. Bending `Qwen35` would put `if (spark)` branches
through a 3610-line file that is the engine's hottest path. A separate class matches how this repo already handles
Divergent architectures (`Dory`, `Flint`, `NemotronH`, `Clover`, `Qwen4Exp` are all separate classes).

**D2. Reuse Dory's skeleton shape, not its kernels.**
`Dory` is the closest existing dense-attention-only model (also 256 head_dim, also per-layer
`sliding_attention`/`full_attention` with per-type RoPE). Its *config parsing and cache layout* are the right template.
But its `dory_*` kernels bake in NVIDIA's scaling constants, NVFP4 formats, slerp residuals and looped-recurrent layout,
so its **kernels** are the wrong template. Reuse the structure, write Spark's own encode path.

**D3. Reuse these existing generic kernels where the math matches exactly:**
- ~~`attn_prep`~~ — **this part of D3 was wrong and is superseded (see §13, defect 5).** `attn_prep` applies an RMSNorm
  to q and k *unconditionally*; an all-ones weight means "normalize to unit RMS", which is not the identity, and it
  changes `softmax(q·k/√D)` because q and k get different per-row factors. Spark has no q/k norm at all, so the prologue
  had to be Spark's own: **`spark_attn_prep`** (RoPE + q|gate packing + KV store in one pass). The rest of D3 stands.
- `attn_prefill11r` / `attn_prefill11` (simdgroup prefill attention, head dim 256, G = Hq/Hkv).
- `qmv` / `qmvf` / `qmms` / `qmm3` quantized matvec and tile GEMMs — but **only after** the GELU work in D4.
- The sampler, tokenizer, KV caches, `Seq`, prefix cache, server, API adapters: untouched.

**D4. New kernel work is limited to three things.**
1. **Exact (erf) GELU MLP.** Kapitsa is SiLU throughout. Spark's config says `gelu`, and ACT2FN maps that to
   `nn.functional.gelu` with its default `approximate='none'` — the exact erf form, not the tanh approximation this
   section first assumed (that assumption was wrong and is corrected in §13). Add it as a compile-time template flag
   (`GELU`) so SiLU instantiation stays bit-identical for every existing model — this is the key numerics-safety
   property: **existing models must compile to exactly the same kernel code as before.**
2. **Per-head broadcast gate.** A variant of the attention epilogue that broadcasts one scalar per head over 256 dims,
   instead of a per-channel vector. Cheapest correct option is to keep the existing per-channel gate layout and have the
   host *write the same scalar to all 256 slots* — zero kernel change. **Decided: do the host-side broadcast first**
   (it costs 256 writes per head per token on the host, negligible), and only add a kernel variant if measurement shows
   it matters. This keeps the numerics surface at zero.
3. **Sliding-window masking** for the prefill attention path. `qattn_t_*` and `attn_prefill11*` have no window parameter;
   `dory_attn` does. For correctness-first, either (a) cap the key range per query in the encode path, or (b) add a
   `window` field to `AttnPrefParams` and use it exactly as `dory_attn.metal` does (`kfirst`/`klo`). **(b) is the plan:**
   it mirrors proven logic and is a small, reviewable diff. Note: adding a field to `AttnPrefParams` changes a struct
   that `qattn_t.metal` also declares as `QaParams` — keep both in sync, and prefer appending the field so existing
   dispatches keep their offsets.

**D5. Read `rot` per layer type, not globally.** `Qwen35` computes one global `rot_dims = head_dim *
partial_rotary_factor`. Spark needs 64 on full layers and 256 on sliding layers. `AttnPrepParams.rot` is already a
per-dispatch argument, so this is a host-side change only: store `rot_full` and `rot_swa`, pick per layer.

**D6. No drafter in scope.** Spark ships no MTP head and no DFlash drafter. `VitreousCut/Spark-X2.6-4B-EAGLE3` exists
but Kapitsa has no EAGLE support and adding one is a separate project. Ship prompt-lookup (`--spec N`) only. The user
confirmed a drafter is not critical.

**D7. No Neural Engine split.** Kapitsa's ANE FFN split is gated on measured shapes (`mma_share_measured` wants
hidden 2560 / I 9216 / 32 layers). Spark-4B is hidden 2560 but I=10240, 36 layers, so it will not auto-enable; leave
`NI_ANE` off. Revisit only with measurements.

## 5. Work breakdown

See `docs/spark/TODO.md` for the live checklist. The phases are:

- **P0 — Scaffolding.** `Spark` class skeleton, `is_model()`, engine dispatch, CMake wiring, config struct + parser,
   dense/quant format validation. No kernels. Must compile.
- **P1 — Load.** Map Spark tensor names onto `QLinear`/`Tensor`, handle the fused `q_k_v_proj` (slice into q/k/v views,
   which the existing multi-matvec dispatch already supports as 3 segments) and the `g_proj` vector, tied embeddings.
- **P2 — GELU kernels.** Template flag through the MLP kernels; assert existing instantiations are unchanged.
- **P3 — Attention path.** Prefill + decode with per-layer `rot`, window masking, no q/k norm, per-head gate
   broadcast, correct KV cache sizing (a sliding layer's ring holds `window + prefill_chunk`, rounded to a power of
   two; a bare 512-slot ring is overwritten by a batched pass, see §14).
- **P4 — Config tests.** Extend the CPU-test surface so `Spark` config parsing is covered without a GPU, in the style of
   `qwen4_config_test` (which is a *pure CPU* test in the CI list — this is our main automated proof).
- **P5 — Docs + catalog.** `docs/spark/` finalized, README/usage notes, model detection documented.
- **P6 — PR + push to fork.**

## 6. Verification plan (what is provable here, and what is not)

| Claim | Provable on this box? | How |
|---|---|---|
| C++ compiles | No — needs macOS | CI `build-and-test` |
| Config parsing / shape validation correct | **Yes** | new `spark_config_test`, run locally if a host compiler is present |
| Tensor-name mapping / quant-format acceptance | **Yes** | same CPU test with synthetic metadata |
| Metal kernels compile | No | CI `metal-compile` |
| Kernels compute the right answer | No | **user's Mac** |
| Outputs match the reference | No | **user's Mac**, `logit_dump` + `tools/logit_compare.py` |
| Speed | No | **user's Mac**, `kapitsa bench` |

Local C++ checks that *are* possible on Linux: compile individual translation units that have no Apple framework
dependency (`src/model/dory.cpp`-style code pulls in Metal, so mostly not possible), and run the pure-logic tests if
their dependencies allow. Treat every local build claim as best-effort and state it precisely.

## 7. Delivery

- Work on a branch in the fork, push at the end, open a PR **against the fork's own main** (not upstream) unless the
  user says otherwise.
- PR body: what changed; a **Not verified** section naming the GPU/Metal gap explicitly; the exact Mac commands the user
  should run to verify.
- Workflows, as the user asked: the fork keeps **`ci.yml`** (trimmed to `lint`, `pr-size-script`, `docs-links`,
  `build-and-test`, `metal-compile`) and **`pr-size.yml`**. Deleted: `release.yml` (signing/tarball/Homebrew, irrelevant
  here) and `community-bench.yml` (community benchmark ingestion, and it uses `pull_request_target`). Also dropped from
  `ci.yml`: the `community-table` and `bench-readme` jobs (they police `bench/community` tables and the harness README,
  which this project does not touch) and `agentic-metrics-script` (tests `bench/harness/agentic.py`). `build-and-test`
  and `metal-compile` are the two that matter: they are the only automated proof that the C++ compiles and that the
  Metal libraries translate.
  - Deleting `release.yml` broke a relative link in `CONTRIBUTING.md`; found by the kept `docs-links` job and fixed.
- Keep one logical change per PR at the upstream-contribution level; if the user later wants to send this upstream, the
  split is: (PR1) `Spark` class + config + tests, (PR2) GELU kernel flag, (PR3) window masking, (PR4) docs/catalog.

## 8. What lives in `docs/spark/`

- `SPARK.md` — this file. The plan and the state log.
- `TODO.md` — the live checklist.
- `configs/*.json` — verbatim HF `config.json` for both models (facts, not edited).
- `configs/NOTES.md` — derived architecture notes, per-layer tables, weight-key inventory.
- `configs/NOTES.md` §"Reference file pointers" — the upstream URLs and the line numbers that matter. The reference
  sources are deliberately **not** vendored: they are someone else's code, and carrying them would trip the repo's own
  lint and PR-size checks for no benefit.

## 9. State log (append a dated entry at every major turn)

**2026-10-29 — P0/P1/P3-code landed; nothing run on a GPU.**
- Forked to `Wiselfx/kapitsa`, cloned to `/home/paseo/workspace/kapitsa-spark`.
- Dev box is Linux aarch64: no Metal, no Xcode. The **user tests on a Mac**; CI compiles on GitHub.
- Read `AGENTS.md` in full; constraints captured in §2.
- **Decisions taken, with the evidence:**
  - `Qwen35` hard-requires GDN config fields and `q_norm`/`k_norm` weights (`src/model/qwen35.cpp:988-989`,
    `read_config` at `:604-610`) ⇒ **D1**, a separate `Spark` class.
  - `AttnPrepParams.rot` and `.theta` are per-dispatch ⇒ **D5** is host-side only.
  - **D4 was changed from the original plan:** instead of adding a `GELU` template flag to the shipping MLP kernels
    (which would recompile the hot path of *every existing model* and put their numerics at risk), Spark gets its own
    tiny kernel library `src/kernels/spark/`. The change is purely additive. See §10.
- **Landed:**
  - `src/model/spark_config.{h,cpp}` — pure-CPU config parse, validation, derived tables (rot/window/kv/rings).
  - `tools/spark_config_test.cpp` — CPU test over both published configs + 12 refusal cases, plus the untied-head and
    group-32 kernel-name sweeps. **Runs and passes on this Linux box.** Verified it has teeth (deliberately broke
    `rot_dims` → 2 failures).
  - `src/model/spark.{h,cpp}` — the `Spark` model: load (per-tensor mixed quant, fused `q_k_v_proj` row-slicing,
    tied or untied output, HF vs MLX norms decided from values), KV caches (a window + chunk ring for sliding layers,
    append-only for the 9/7 full layers), the decode path, and batched prefill (`enc_attn_prefill`/`enc_mlp_prefill`).
  - `src/kernels/spark/spark.metal` + `cmake/embed_kernels_spark.cmake` + CMake/CI wiring.
  - Engine dispatch (`engine.cpp`), the "model Kapitsa cannot run" refusal, the memory plan, and the no-drafter gate.
  - `tools/lx_syntax_check.sh` — local Linux syntax check (see §11).
- **A real bug caught by reading the kernel contracts, not by compiling:** `attn_prep` reads the query out of an
  **interleaved `[row][Hq][2][D]`** buffer (Qwen's `q_proj` emits query+gate together, `attn.metal:35`), while my first
  version handed it the dense `q_k_v_proj` output. Fixed with `spark_pack_qg`, which assembles the interleaved layout
  from the dense q and the broadcast per-head scalar. This is exactly the class of mistake only a GPU run confirms —
  see §6.
- **Closed since the first version** (each has a section below and a `TODO.md` entry):
  - A **wrapped sliding window** now works: `attn_decode_partial` reads the ring cyclically (`wrapped`/`slot_mask`, the
    tile loads start at the oldest live key and wrap -- the same weighted sum over the same keys), so a conversation
    is no longer capped at 512 tokens (§12d).
  - **Snapshots** (`save`/`restore`, `can_snapshot`) now work: only the rings are copied, so the engine can checkpoint
    at a turn boundary and a conversation can have a second turn (§12e).
- **Known gaps (P3 remainder), all stated in code comments and `TODO.md`:**
  - Batched prefill is bound (`encode_prefill`); `encode_logits_rows` still runs a row at a time and is slow.
  - The reference's W8A8 activation quantization is not reproduced: its INT8 out_proj/lm_head weights are
    dequantized to bf16 and multiplied by bf16 activations, so results are close but not bit-identical to that backend.
  - No `copy_prefix` override: a second conversation sharing a system prompt re-reads it rather than copying another
    slot's caches. Correct, slower, off the failing path.
  - Gate and up projections are two dispatches, not one (D4; see `TODO.md` for why it was rejected).
  - No drafter (by design, D6); prompt-lookup only.
- **CI, first green run (PR #1, commit 4a14d22 → 4c8e9f1):**
  - `lint` success, `docs-links` success, `pr-size-script` success.
  - `metal-compile` success — **all six embedded libraries translate**, `spark` among them at metal3.2.
  - `build-and-test` success on `macos-15` — the whole CLI plus every CPU test target builds, and
    `build/spark_config_test` **ran and printed `spark_config_test: ok`** on a macOS runner. So the C++ does compile
    for real (not just under the Linux shim) and the config logic is verified on the shipped configs.
  - `metal-compile` earned its keep immediately: it caught a **Metal vector-width mismatch** (`uint2` grid position
    with a scalar `uint` thread position) in both Spark kernels that no Linux check could see. Fixed in 4c8e9f1.
- **What "green CI" does NOT mean**, and the PR says so: the kernels' arithmetic, the ring cache, the interleaved
  q|gate layout, and every speed number are still unverified. `metal-compile` explicitly cannot see wrong arithmetic,
  races or the 32 KB threadgroup budget; `build-and-test` never loads a model.
- Next, in order: batched prefill + GELU fold in the GEMM tile,
  then the Mac verification pass (a short generation and, if it works, the `logit_dump` output check).

**2026-10-29 — delivery.**
- Branch `spark-x2.6` pushed to `Wiselfx/kapitsa`; PR #1 opened against the fork's `main` (not upstream).
- `pr-size` is red by design: ~2,100 counted lines against a 1,000 cap. That cap exists to keep *upstream* PRs one
  change each; a new model backend is one indivisible change (model + config + kernels + tests). The fork has no
  reviewer to grant `size/override`, and the author must not label their own PR through it, so it stays red and is
  explained in the PR body rather than bypassed.

**2026-10-29 (later) — QA pass against primary sources: four defects fixed.**

I reviewed my own port against primary sources rather than against my memory of them: the pinned
`transformers==4.57.1` wheel, the official PyTorch docs, and this repo's own kernel and encoder code. It found four
real defects. Three would have shipped silently or died on the first Mac run. Full detail in §13; the summary:

1. **The MLP used the wrong GELU** (correctness, silent). I had written the tanh approximation and claimed in a comment
   that `ACT2FN["gelu"]` meant it. `transformers==4.57.1 activations.py` maps `"gelu" -> GELUActivation`, whose forward
   is `nn.functional.gelu` with **no** `approximate` argument, and torch's default is `'none'` — the exact erf form.
   `"gelu_pytorch_tanh"` is a *separate* key that Spark's config does not request. Up to 4.7e-4 per element wrong,
   through 28-36 layers. Now exact erf via the same A&S 7.1.26 approximation `vit.metal` already carries (Metal has no
   `erf` — the repo knew this and I initially wrote `metal::precise::erf`, which does not exist). The CPU test now
   asserts exact-not-tanh and was verified to fail if it is reverted.
2. **`SparkLinear::fmt()` returned the wrong spelling** (`"bf16"` / `"4-bit group 64"`) for a string that is
   concatenated into kernel names, where the suffixes are `"f16"` / `"f4g64"`. Every non-4-bit weight would have failed
   its pipeline lookup at load.
3. **A kernel name that does not exist** — `qmv2_b8_g64_r2`; `qmv.metal` only instantiates `_r{1,2,4}_norm` for b8.
4. **An unstated alignment assumption** in the fused-qkv byte-offset slicing, now asserted rather than assumed.

Defects 2 and 3 had one root cause: **the matvec kernel-selection rule was copy-pasted into four call sites** with
slightly different conditions. It is now a single function, `Spark::matvec_pipe(W, norm, add)`. That is the structural
fix; the naming fixes are just its symptoms.

The pass also *confirmed* several things with citations, so they should not be re-litigated without new evidence: the
RoPE convention (`rotate_half` half-split + `theta^(-2i/rot)`, matching `attn.metal`), the per-layer rotary dims
(64 / 256 with the tail passed through), the attention math and scale, the gate's position after attention and before
`out_proj`, tied embeddings, the absence of q/k norm, and the `(i+1) % 4 == 0` full-attention pattern.

What it does **not** establish: anything about the kernels' arithmetic on a GPU. The PR remains "Not verified on a GPU".

**2026-10-29 (later still) — QA pass 2, three more defects; CI green again on 5c16e64.**

Continuing the review found the worst defect of the batch, plus two more:

- **`attn_prep` cannot be reused** (this overturns decision **D3** in §4). It RMS-normalizes q and k unconditionally;
  handing it an all-ones weight means "normalize to unit RMS", not "skip", and because q and k get different factors it
  changes `softmax(q·k/√D)`. Replaced by `spark_attn_prep`. The `ones_` buffer that existed only to feed it is gone,
  which is the tell: it had no reason to exist.
- **`spark_pack_qg` read a buffer nothing had written** (dense q came from `qg_`, the pack read `qo_`), and the obvious
  in-place fixes for both the packing and the rotation are races. q now has three distinct buffers.
- **`hf_norms` was dead** — detected, logged, never applied, so every RMSNorm would have used a weight centred on 0.

CI on `5c16e64`: `lint`, `docs-links`, `pr-size-script`, `metal-compile` (both Spark kernels translate at metal3.2) and
`build-and-test` on `macos-15` all green, with `spark_config_test` **executing** — it now covers the exact-GELU
decision, the HF-vs-MLX norm rule, and the 1 + w conversion.

The honest status is unchanged in one respect and better in another. Better: seven defects are fixed and the host-side
logic is now either tested or asserted. Unchanged: **no GPU has run any of this**, so the kernel arithmetic, the ring
cache, the RoPE end-to-end and every speed number remain unverified, and the PR still says so.

**2026-10-29 (later) — a note on how to review this branch.**
The QA pass is the reason to distrust any claim in this branch that is not backed either by a CPU test that runs in CI
or by a quoted primary source. Where those two disagree with a code comment, the comment is wrong. Every remaining
GPU-dependent assertion is listed in §6 and marked as such.

## 10. Why Spark's kernels are a separate library (the numerics argument)

Kapitsa's rule is that a change which is faster but produces different tokens is a different change. Adding a `GELU`
flag to `qmms.metal` / `qmm3.metal` / `qmv.metal` / `qmvf.metal` would instantiate new templates in the same translation
unit that every existing model's kernels are built from. Even with the flag defaulted off, that is a recompile of the
hottest code in the engine for a new family's convenience, and any accidental change to the SiLU path would move
Qwen/Nemotron/Dory/Flint outputs — requiring the full `logit_dump` gate on models nobody meant to touch.

`src/kernels/spark/spark.metal` is therefore its own library, embedded by `cmake/embed_kernels_spark.cmake` and compiled
by `Gpu::compile_extra("spark", kSparkKernelSource)` on first use (the same mechanism Dory and the tensor library use).
Existing libraries are byte-identical to before this change. The cost is two extra dispatches per MLP (the GELU fold)
rather than a fused epilogue; that is a speed trade to revisit *after* correctness, with measurements.

## 11. The local verification loop (Linux, no Apple SDK)

`tools/lx_syntax_check.sh` compiles the C++ side with `g++ -fsyntax-only`, shimming the two macOS-only things in the
scratch directory (never committed): `__builtin_available(...)` and `<sys/sysctl.h>`. It proves: no typos, correct
signatures, correct struct/field use, no missing includes. It does **not** prove: that the code links, that Metal
compiles, or that anything computes the right answer.

Verified working on this box for `spark.cpp`, `spark_config.cpp`, `engine.cpp` (and, as a control, `dory.cpp` /
`qwen35.cpp` / `nemotron_h.cpp` all check clean too). `tools/spark_config_test.cpp` is a genuine executable test and is
run for real. Both are wrapped by the same commands recorded in the PR body.

## 12. If you are resuming after a compaction

**Read §13 (the QA pass) and §9's last two entries before touching anything.** §13 lists four defects that were already
fixed and six properties that were verified against primary sources; re-deriving those from scratch burns context and
risks "fixing" something that is already correct. The current state of the branch is:

- **Host-side logic: reviewed, four defects found and fixed.** Config parsing, kernel-name construction, the
  fused-qkv byte-offset slicing and the GELU choice are covered by `tools/spark_config_test.cpp` (runs in CI).
- **The model now loads, templates, and produces tokens on a Mac** — that is the user's report, not a measurement:
  `v0.2.25` loaded and `/v1/chat/completions` returned a 400 from the chat template (§12c); after that the model ran
  and reached 513 tokens (§12d); then a second turn needed checkpoints (§12e).
- **Kernel arithmetic: only partly verified.** Nothing here has measured logits against `transformers`, so "it runs"
  is not "it computes the right thing". Do not describe it as correct. The ring read and the checkpoint copies in
  particular have never produced a token.
- **Functional state:** the wrapped sliding window works (§12d) and snapshots work (§12e), so a multi-turn
  conversation is no longer structurally impossible. Batched prefill is implemented (§14); the remaining speed work is
  measurement on the Mac, not another structural rewrite. No drafter.


1. Read §2 (constraints), §4 (decisions D1–D7), then the last §9 entry.
2. `docs/spark/TODO.md` is the live checklist — trust it over your memory of this file.
3. Run `bash tools/lx_syntax_check.sh` and the config test before changing anything, so you know the baseline is green.
4. Do not claim anything about Metal, speed or output equality. The PR says "Not verified on a GPU" because it is.

## 12b. Handing the build to a Mac (the release)

The user cannot verify anything until there is a binary, and this box cannot build one. So the fork publishes one:
`.github/workflows/release.yml` builds on `macos-15` via `packaging/release/build.sh` and attaches the tarball to a GitHub
release. Push a tag whose name matches `KAPITSA_VERSION` in `CMakeLists.txt` (currently `0.2.29`, so `v0.2.29`);
`build.sh` refuses a mismatch.

**The workflow is deliberately not upstream's.** Upstream's release job pushes the formula to `nace-ai/homebrew-tap` and
*fails* when `TAP_GITHUB_TOKEN` is missing. A fork must not push upstream's tap, and this fork has no tap, so that step is
gone: **Homebrew is not an install path for these builds**, `install.sh --from-file` and the tarball are. Signing and
notarization are optional and absent (no secrets), so the binaries are ad hoc signed — the notes tell the user to clear
the quarantine attribute, and explain why Gatekeeper complains.

Two things that will bite the next person, both found by checking rather than assuming:

- **There is no `kapitsa run`.** Kapitsa has no one-shot generation command; `kapitsa serve` plus a
  `/v1/chat/completions` request is the way, so that is what the release notes give.
- **`kapitsa doctor` takes `--model M` or `--models-dir DIR`, not a positional folder.** The earlier TODO and PR text
  said `kapitsa doctor ~/models/spark-4b`, which would have been a usage error on the user's first command.

## 12c. The first real request: a chat-template bug outside Spark's own code

The model loaded on a Mac (the load path works, including the fused-qkv slicing and the norm conversion), and then the
request failed before any inference:

```
400: chat template: Unknown argument separators for function tojson at row 51, column 66
```

Spark's published template renders tool schemas with `tojson(separators=(',', ':'))`. Kapitsa's template engine is
`third_party/minja`, whose `tojson` builtin accepted only `(value, indent)`, so it raised on the keyword. Nothing to do
with the model code — but it meant **no request carrying tools could reach the model at all**, and the template calls
that filter unconditionally whenever `tools` is non-empty.

Fixed in `third_party/minja/minja.hpp`: `tojson` now takes Jinja's `separators` (a 2-tuple, or one value for both),
matching Python's `json.dumps`, and serializes rather than post-processing text. That last part matters — my first
attempt rendered with `Value::dump` and then rewrote `", "`/`": "` into the requested separators, which corrupted the
JSON (`{"name",:"get_weather""description",:...}`) because it also rewrote the separators *inside* string values. The
fix is to emit directly, which is what the code now does.

Two things this changed about the repo's own assumptions, both corrected:

- `src/engine/engine.cpp` said "minja (third_party, **not edited**)". It is edited now; the comment says what and why.
  `ensure_ascii` is still not accepted, so Flint's source rewrite stays.
- minja had no test. `tools/minja_template_test.cpp` pins the filter (compact, default spacing, single-argument,
  indent, string contents untouched, JSON `true`/`null`) and renders **Spark's published template** both with and
  without tools. Verified it fails against the unpatched header — with exactly the error from the Mac — and passes
  after. Wired into CI and `ctest`.

This is the general lesson from the whole port, again: the failure was in an assumption I never checked (that the
template engine could run the template), not in the code I wrote.

## 12d. Past 512 tokens: the wrapped window

The next request failed at 513 tokens (prompt + reply), on the deferred blocker:

```
Spark: context reached 513 tokens, past the 512-token sliding window, whose wrapped key range needs the
ring-aware attention read (docs/spark/SPARK.md section 6); positions below the window are fine
```

That was honest but useless: a 512-token window is smaller than most real prompts, so the model could not hold a
conversation at all. `docs/spark/TODO.md` had three designs pencilled in (a kernel that can start at an arbitrary slot,
a per-token compaction pass, a double buffer). The first is what landed, and it is much smaller than I expected because
I had been thinking about it wrong.

**The key realisation.** I had been treating the wrapped window as a *reordering* problem -- the ring holds the right
keys in the "wrong" order, so something must reorder them. But attention is a weighted sum over a *set* of keys:

    out = Σ_t softmax(q·k_t) v_t

The order comes from the softmax weights, not from the slots. Reading the ring in ring order gives bit-for-bit the same
sum as reading it linearly; it is a change of rotation in the iteration, not a permutation of the result. So the kernel
needs no sort, no compaction, no second buffer -- only to start its tile loads at the oldest live key and wrap the
index. That is `AttnDecParams::wrapped` plus `slot_mask` (`(wrapped + t) & slot_mask`), ~15 lines in
`src/kernels/attn.metal`, and it applies to the existing kernel rather than replacing it. Every other model passes
`{0, 0}` and takes the same linear path as before.

**What it depends on**, now checked by `tools/ring_slot_test.cpp` instead of asserted in a comment:
- the ring length (`window + prefill_chunk`, rounded to 4096) is a power of two, so `& (ring_rows_-1)` is the modulus;
- window is a multiple of the kernel's `kTile` (16), so a tile crosses the seam at most once and can be loaded as two
  contiguous runs (the whole tail first, then the head);
- the host's `wrapped` equals the write side's slot for the same position -- an off-by-one here would attend over the
  wrong keys and produce *plausible nonsense*, which no smoke test would catch. The test compares the two formulas over
  four windows of positions and checks each live key is read exactly once.

The later batched-prefill bug (§14, R1) showed that a ring of exactly `window` slots is not enough: the pass writes its
whole chunk before attention reads, so the first query's oldest keys are overwritten. `spark_ring_rows` is now
`next_pow2(window + prefill_chunk)`; the slot arithmetic is unchanged, only the capacity grew.

One thing I got wrong on the way, in my own test: `read_slot(512, 511) == 512 & slot_mask` parses as
`(a == 512) & slot_mask`, so it compared a bool against an int and "failed" against correct code. Parenthesised.

`L` also had to become the *live* span rather than `pos+1`: passing 513 with a 520-slot ring would have read a phantom
key. For a full layer the two are equal, so the expression stays `span`.

**Not verified on a GPU**: CI compiles the Metal and runs the CPU slot test; nothing here has produced a token past 512.

## 12e. A second message: the missing snapshot

With the window fixed, the next thing a real conversation does failed:

```
Spark has no snapshot (can_snapshot is false)
```

`Spark::save`/`restore` were fatal stubs and `can_snapshot()` returned false, so the engine could not take a checkpoint
at a turn boundary. The first message got through (nothing needed a checkpoint); the second needed the state after the
first turn, and there wasn't one. The stubs were honest -- they never silently did nothing -- but they made the model
single-turn, which is not a model that can be tested.

The implementation is Dory's, and it is small because of one observation: **only the rings need saving.** A
full-attention layer's cache only appends, so every position up to the checkpoint is still in it and truncating to a
shorter sequence reads the same bytes. A ring, by contrast, has been overwritten by the time you want to go back. So
`save` copies exactly the sliding layers' caches (`spark_copy4`, in Spark's own kernel library rather than borrowing
`dory_copy4` -- D4), `restore` copies them back, and `can_snapshot()` is now true.

That last part is not cosmetic: the engine's `in_pass` flag reads it, and with `can_snapshot()` true the engine stops
cutting prefill chunks short at a boundary and instead asks `encode_prefill` to snapshot mid-chunk. Having returned
true, `encode_prefill` must honour `snap`/`snap_row` or the checkpoint is silently never taken -- so it does, and
because prefill is still one row at a time that is one `save` queued behind the row's layers in the same encoder. This
is the one place where the slow row-at-a-time prefill makes something *easier* than the batched path.

`copy_prefix` is left at the base class's `false`: a second conversation sharing a system prompt re-reads it rather
than copying another slot's caches. Correct, slower, and not on the path that was failing.

**Not verified on a GPU**: the checks are that `save`/`restore` touch only ring layers, that a declined save leaves
`pos = -1` (so it is not mistaken for one), and that the existing CPU tests still pass. No multi-turn conversation has
actually been run here.

## 12f. The SSD cache that claimed to work

Found by asking "where is this checked?" rather than by a failing request -- the user asked how snapshots ought to be
implemented and where that contract lives, and reading `src/model/model.h` for the answer turned up a second, worse bug
in the same area.

`Model` describes a state to the SSD prefix cache through three virtuals: `disk_format()` (non-empty enables it),
`disk_rows` (the linear attention caches) and `disk_state` (the state kept whole -- its comment names "Dory: its
sliding-window caches"). Both hooks default to doing nothing. Spark implemented `disk_format` from the first commit and
neither hook, and the consequence is not a degraded cache but a **silently wrong one**:

- `Engine::disk_cache_supported()` is literally `!disk_format().empty()` (`engine.cpp:630`), and the server defaults
  `--disk-cache` to `"auto"` (`main.cpp:948`), so Spark *enabled* the cache.
- `disk_rows` returned nothing, so `dc::kv_layout` hashed an empty list and no KV block was ever written; `disk_state`
  likewise, so `StMeta` hashed 0 and no state was written.
- `disk_restore` compared those zero layouts, matched, restored nothing, and **returned true** -- then logged "read N
  tokens of a stored conversation into slot N" and set `start = N`.

So a second conversation would have continued from an uninitialised cache and produced plausible nonsense, with a
success message. Worse than the ring fatal, which at least stopped.

**The fix is the hooks, not a disabled feature**, because Spark has exactly the split Dory already solves: its
full-attention caches are linear (position p is row p) and go in `disk_rows`; its sliding layers are rings, whose
positions do not index them, so they go whole in `disk_state` (k then v, the same order and sizes for a checkpoint as
for the live state). Verified line-by-line against `Dory::disk_rows`/`disk_state`, which have the same shape.

**And a guard, because the class of bug is bigger than Spark.** A model that advertises a format it cannot describe
should not be able to reach the engine at all:

- `Model::disk_hooks_live(seq)` asks the hooks what they describe; `Engine::disk_start` refuses to enable the cache when
  the answer is nothing, and says which model and why. Defence in depth for a case the static check cannot see (a
  plugin, a split model).
- `tests/test_disk_hooks.py` resolves `disk_format` *and* both hooks **through the class hierarchy** and fails if any
  model ends up with a format but the base class's no-ops. Its first version reported five false positives (Clover,
  DSpark and LiliCorr inherit their hooks from Qwen35/NhDrafter; NemotronH overrides none of the three) -- inheritance
  was the thing I had to get right, and getting it wrong would have made the test useless rather than wrong. Verified
  it fails when Spark's two hook declarations are removed and passes when they are back.

The lesson worth keeping: `disk_format` returning a string is a *promise*, and I made it three commits before doing
anything that could keep it. Nothing in the build, the tests or a smoke run distinguishes "cache works" from "cache
silently stores nothing", which is why the guard is a check and not a comment.

**Not verified on a GPU**: the hooks compile and the static tests pass; no state has been written to and read from an
SSD cache for Spark. The user runs with `--disk-cache 0`, which is why this never surfaced on their machine -- the
default is `auto`, so anyone else would have hit it.

## 12g. Audit: the load path, against the other models (four more defects)

A full trace of Spark's load path against Qwen35/Dory/Flint found four more, and three of them were in things I had
already declared verified.

**1. `owned_` was a `std::vector`, and its element addresses are handed out.** `Spark::norm_copy` converts HF's
zero-centred norms to `1 + w` and returns `&owned_.back()`. On the 4B that is 73 calls; every `push_back` may
reallocate, so every pointer the loader still holds (`L.in_norm`, `L.post_norm`, `final_norm_`) dangles and the model
reads freed storage. Qwen35's identical function keeps `std::deque<Tensor> owned_` -- pointer-stable -- and this file
copied the pattern and changed the container. That difference reads as a style choice and is the whole bug. The tell was
in the sibling implementation the whole time; I had compared the *logic* of `bf16_copy` and not the storage.

**2. `MemPlan::activations` was wrong twice.** It multiplied one row's activations by `prefill_chunk`, reserving
~310 MB on the 4B that nothing allocates (prefill runs a row at a time), and the per-row formula did not match
`alloc_state` even then (ffn counted twice, `part_o_`/`part_ml_` missing). The number feeds the engine's fit checks and
`doctor`, so Spark looked much larger than it is. Now `spark_activation_bytes` in the CPU-only config module, called by
both `plan()` and the test.

**3. Two buffers were allocated and never read**: `xh_` (a bf16 copy of the residual, which the `_norm` matvec forms
make unnecessary -- they normalize in-kernel) and `qo_` ("spare q scratch (prefill rows)", a leftover of a design that
no longer exists). 30 KB, but two comments describing behaviour that does not happen.

**4. My own matvec-name test was checking the wrong names.** It hand-typed the kernel set and got it wrong twice:
it used `f16g64` for bf16 (wfmt.metal's suffix is `_f16`; `SparkLinear::fmt` returns `"f16"`), so the format a *bf16
checkpoint* actually uses was never checked at all, and it claimed `qmvg_f8g64_*` decode kernels that are never
instantiated (decode is the `NI_WF_NO8` group; 8-bit takes the `qmv2_b8`/`qmv3_b8` path). The real kernel set is now
generated from the Metal sources (`tools/spark_kernel_names.py`, wired into CMake) and the sweep checks
`matvec_pipe` against it. Verified it fails when a kernel is removed from the generated list.

**What the audit found sound**, having actually traced it rather than assumed it: `matvec_pipe` emits only names that
exist (44 decode kernels, swept over every (bits, K, N, mode) combination the two models can produce); the fused
q|k|v path's 16-byte alignment assert; `check_shapes`; the `zero_centred` decision and its ≤64K guard; `reserve`
copying via `kv_regrow` and never growing a ring; `reserve` being called once per chunk rather than per row; the
`gpu_cores()` sizing of both the split buffers and the chunk count.

The pattern in all four: every one was a place where I had written something *similar to* an existing model's code
and the difference -- a container type, a chunk multiplier, a kernel-name list -- was where the bug lived. Reading the
sibling implementation for agreement, not just for the idea, is what found them.

## 12h. "This is all I see": the sampled token went to the wrong slot

With the window and the snapshots fixed, the model finally replied -- the same token, over and over:

```
<｜start▁of▁sentence｜><｜start▁of▁sentence｜><｜start▁of▁sentence｜>...
```

A perfectly repetitive output is a very specific symptom: it means the *input* stopped changing. The cause was one
line. `Spark::encode_step` called `enc_sample(..., pos, ...)`, and `sample_stage2` does `tokens[p.out_index] = chosen`,
so each step wrote its token over the token it had just embedded. The engine then reads `s.tokens()[pos]` to get the
model's output and got back the *prompt's own* token, and the next step embedded a slot nothing had written. After
step one, the input was constant, so the output was constant.

Qwen35 does `enc_sample(e, s, x_, pos + 1, sp, step)` (qwen35.cpp:2439). This is the one character where the two
implementations differed, and it is why the port looked like it worked: the prefill produced a plausible first token,
and everything after it was frozen.

The capacity guard had to widen with it -- `pos + 2` rather than `pos + 1`, or the sampled slot may not exist.

`tests/test_spark_sampling_offset.py` pins all of it statically, because no test here runs a GPU: it checks that
`sample_stage2` writes `tokens[out_index]`, that Qwen35 uses `pos + 1`, that both of Spark's sampling call sites do
too, and that the guard matches. Verified the test fails on the pre-fix code. Its guard check was **vacuous when first
written** ("pos + 2" also appears in the comment above the line), so it now matches the `if (...)` statement itself;
a test that passes on prose is not a test.

**What this says about the rest of the audit.** I traced the load path and the kernels twice and never looked at the
one line that connects them to the engine's expectations, because "the model runs and emits tokens" read as working. The
lesson from §12c–§12g repeated: the bug is in the seam between my code and somebody else's assumption, and the seam is
exactly where a diff against the sibling implementation is cheapest. Both of the last two defects -- this and the
`std::vector` in §12g -- were a single token of difference from Qwen35.

## 12i. Garbled output: down_proj multiplied the residual

After the sampling fix the model produced varied but meaningless text -- fluent-looking token soup in a mix of scripts,
reproducible, the same for a given prompt. Varied output ruled out the previous bug and pointed at the arithmetic.

The cause was in `enc_mlp`. Its inner `matvec` lambda took `(W, y, norm, resid)` and bound buffer 3 to `x_`
unconditionally; buffer 3 is the matvec's *input row*. That is right for gate_proj and up_proj, which read the residual
and normalize it on the fly, and wrong for down_proj, which must read `mh_` -- the `gelu(gate) * up` output. So every
layer computed `down_proj(x)` instead of `down_proj(act * up)`: **the entire MLP had no activation in it and did not
depend on gate or up at all.** A linear map per layer, applied to the residual.

The lambda's signature is what hid it: three call sites, four parameters, and the buffer that differed was the one not
in the parameter list. `matvec` now takes the input explicitly (`matvec(L.mg, x_, g, ...)`, `matvec(L.md, mh_, x_, ...)`),
so the call site states what it reads.

`tests/test_spark_sampling_offset.py` (renamed in spirit: it now guards conventions rather than just the offset) checks
that gate and up read `x_` and down reads `mh_`, and fails if down goes back to `x_`.

**The pattern, one more time.** §12g, §12h and §12i were all the same failure: a line that resembles the sibling
implementation closely enough to pass review, differing in exactly the thing that matters -- a container type, `pos`
versus `pos + 1`, and here a buffer that was implicit. Nothing in a CPU test can see any of them; all three were found
only because the model's actual output was nonsense on a Mac.

## 13. QA pass (2026-10-29) — what it found, and what it cost

**Ten defects, in four batches.** They are recorded in the order found, because the order is the interesting part: each
batch was found by checking the thing the previous batch had taught me to distrust. Batch 1 was the activation and the
kernel names; batch 2 was the attention prologue and the norm conversion; batch 3 was the matvec rule, exhaustively;
batch 4 was the KV cache. Nothing was found by re-running a check — every one came from reading a kernel's actual
contract against what the host code assumed it was.

Six of the ten share one root cause: **a rule expressed in more than one place**. The matvec kernel choice was
copy-pasted four times; the buffer that should have been named `stg_` was called `qo_` and described as scratch. Where
that pattern appeared, a defect followed. The fixes are structural (one function, one test that checks it against the
kernel sources, buffer names that say what they hold), not just corrections.


A review of the port against **primary sources** (the pinned `transformers==4.57.1` wheel, the official PyTorch docs,
and this repo's own kernels), not against my recollection. Four real defects, three of which would have produced wrong
numbers or a crash on the first Mac run. Recorded here because the pattern matters more than the individual bugs.

### Defect 1 (correctness): the MLP activation was the wrong GELU

I wrote tanh-approximation GELU and asserted in a code comment that `ACT2FN["gelu"]` meant it. It does not.

Evidence chain, all primary:
- `transformers==4.57.1`, `src/transformers/activations.py`, `ACT2CLS` l.315: `"gelu": GELUActivation`, and l.320:
  `"gelu_pytorch_tanh": GELUTanh` — **a separate key**, which Spark's config does not ask for.
- Same file, `GELUActivation.__init__` l.79: `self.act = nn.functional.gelu` — called with no `approximate` argument.
- `docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu`: signature `gelu(input, approximate='none')`, and
  `'none'` is `x * Phi(x)`, the exact erf form. (`'tanh'` is the other option.)
- So Spark needs `0.5 x (1 + erf(x/sqrt(2)))`. Measured difference from the tanh form: **up to 4.7e-4** near |x|~3,
  1.5e-4 near |x|~1 — per element, through 28-36 layers. That is a silent-corruption bug, not a rounding nit.
- Metal has no `erf`; this repo already knew that (`src/kernels/vit.metal:86`, "erf to 1.5e-7 … Metal has no erf").
  Spark now carries the same A&S 7.1.26 rational approximation, measured at max 1.4e-7 (erf) / 2.1e-7 (gelu) against
  `math.erf`.
- `tools/spark_config_test.cpp` now asserts exact-not-tanh and fails loudly if anyone "simplifies" it back (verified:
  swapping the formula in produces 3 failures).

**Lesson:** "gelu" in a config is ambiguous across five HF registrations. The mapping had to be read, not assumed.

### Defect 2 (would not have loaded): kernel-name suffix

`SparkLinear::fmt()` returned `"bf16"` / `"4-bit group 64"`. It is concatenated into kernel names
(`pipe("qmvg_" + fmt() + "_r2")`), and `wfmt.metal`'s suffixes are `f16` / `f4g64` / `f8g64` (see `QLinear::fmt` in
`qwen35.cpp:100`). Every non-4-bit weight would have died on a pipeline lookup at load.

### Defect 3 (same class): a kernel that does not exist

`qmv2_b8_g64_r2` (8-bit, no folded norm) has no instantiation — `qmv.metal` only defines `qmv2_b8_g64_r{1,2,4}_norm`.
The 8-bit path had to route to `qmv3_b8_r2`, which does have plain/`_add`/`_norm`.

Defects 2 and 3 had the same root cause: **the matvec kernel-selection rule was copy-pasted into four call sites** with
subtly different conditions. Fixed by extracting `Spark::matvec_pipe(W, norm, add) -> {pipeline, rows, nsg}`; the rule
now exists once. This is the kind of thing that only bites at run time, on a Mac, one format at a time.

### Defect 4 (latent): an unstated alignment assumption

The fused `q_k_v_proj` is sliced into q/k/v by binding the weights at a byte offset (`row0 * K * bits / 8`). That keeps
16-byte alignment only if a row is a whole number of 16-byte words. Verified true for both models at every supported
format (hidden 2048/2560, ffn 6656/10240, 3/4/5/6/8/16 bits, group 32/64) — and now **asserted in the encode path**
rather than left implicit, so a future variant with an awkward `ffn` fails loudly instead of reading misaligned data.

### Verified as correct (so do not re-litigate without new evidence)

- **RoPE convention.** The reference's `rotate_half` is GPT-NeoX style (split the `rot` dims in half, `cat(-x2, x1)`),
  and `compute_rope_cos_sin` builds `inv_freq = theta^(-2i/rot)` for `i` in `[0, rot/2)` doubled by `cat(freqs, freqs)`.
  `attn_prep` (`src/kernels/attn.metal:40-51`) does exactly this: `half_rot = rot/2`, `inv_freq = theta^(-2i/rot)`, with
  the `-x[+half]/+x[-half]` swap. **Match.**
- **Rotary dims.** `partial_rotary_factor` 0.25 → 64 on full layers, 1.0 → 256 on sliding: the reference applies RoPE to
  `x[:rot]` and passes the rest through, which is what `attn_prep` does with `tid < p.rot`.
- **Attention math.** `eager_attention_forward` is plain dense causal (optionally sliding-masked) with
  `scaling = 1/sqrt(head_dim)`; `attn_decode_partial`/`_reduce` use the same scale and sigmoid the gate. **Match.**
- **Gate ordering.** The reference applies `attn * sigmoid(g)` **after** attention and **before** `out_proj`; so does
  the Kapitsa path (the epilogue multiplies, then `o_proj` runs). **Match.**
- **Tied embeddings** (`_tied_weights_keys`), **no biases**, **no q/k norm** in this family, and the config-level layer
  pattern (`(i+1) % 4 == 0` → full attention) all confirmed against the reference and covered by the CPU test.
- **Embedding gather** exists for every format (`embed_f16`, `embed_f4g64`, `embed_f8g64` via `NI_WF_ALL`), so the
  OptiQ pack's 8-bit embedding is fine.

### Defect 5 (correctness, structural): the attention prologue reused a kernel that normalizes

The worst of the batch. `enc_attn` reused `attn_prep`, passing an all-ones vector as the q/k norm weight on the theory
that this disables the normalization. It does not — `attn_prep` computes
`x * rsqrt(sum(x^2)/D + eps) * w` unconditionally (`src/kernels/attn.metal`), so "all ones" means *normalize to unit
RMS*, which is a different function from not normalizing. Spark's reference applies RoPE straight to the raw
`q_k_v_proj` output and contains zero occurrences of `q_norm`/`k_norm`.

This matters more than it looks: rescaling q and k by a common per-row factor leaves `softmax(q·k/√D)` unchanged, but
this is a *per-row* normalization to a *different* norm (RMS ≈ 1) than the reference's, and q and k get different
factors — so the attention distribution changes. Replaced by `spark_attn_prep`, Spark's own prologue, which does RoPE
(per-layer `rot`/`theta`, NeoX half-split), assembles the interleaved `[Hq][2][D]` q|gate buffer the reduce reads, and
appends k/v to the cache in one pass. The all-ones buffers (and the `ones_` allocation that existed only to feed them)
are gone.

### Defect 6 (correctness): the q|gate packing read a buffer nothing had written

`spark_pack_qg` read the dense q from `qo_`, but the fused matvec wrote q into `qg_`, so the pack kernel would have
gathered uninitialized memory. Chasing it exposed two aliasing hazards in the same few lines, both of which the new
kernel has to avoid and now does: packing into `qg_` in place is a race (group `h` writes `out[h*2D+d]` and
`out[h*2D+D+d]`, which are later groups' *inputs* — with `Hq ≥ 2` every group's input is another group's output), and
rotating q in place is a race (each thread reads its partner lane `tid ± rot/2`). So: q is written to its own buffer,
roped into a second, and the interleaved buffer is a third.

### Defect 7 (correctness, silent and total): `hf_norms` was detected, logged, and never applied

The loader detected Hugging Face's zero-centred RMSNorm layout (`sum/n < 0.5` with negatives → apply as `1 + w`), set
`cfg_.hf_norms`, printed "RMSNorm weights read as 1 + w (Hugging Face layout)" — and then nothing used the flag. Every
norm would have been applied as the raw zero-centred `w`, scaling every layer by a number near 0. This is the class of
bug that reads as working code: a flag, a log line, and no consumer. Fixed with `Spark::norm_copy` (the same
`1 + w`, rounded once to bf16, as `Qwen35::bf16_copy`), applied at load, plus the `n > 64Ki` guard the Qwen35
precedent has so a mis-keyed wide tensor cannot flip the whole checkpoint's norm layout. Both the decision rule and
the conversion now have a CPU test.

### Defect 8 (would not have loaded): `_add_norm` combinations that no kernel defines

`matvec_pipe` appended `_add` and `_norm` independently, so it could ask for `qmv3_b4_r2_add_norm` or
`qmvg_f16g64_r2_add_norm`. **No kernel anywhere has both**: `qmv.metal` instantiates no `_add_norm` form, and
`wfmt.metal`'s `NI_WF_DECODE` lists `r2`, `r4`, `r2_add`, `r2_norm`, `r4_norm` — note `r4` has no `_add` either. The
rule also picked `qmv2_b8_g64_r2` for an 8-bit matrix with no folded norm, which does not exist (only `_norm` does).

Rewritten against the instantiation lists read out of the two kernel sources, with `add`/`norm` made mutually exclusive
by construction (the one place a residual and a norm are both involved, `out_proj`, passes `add` only because
attention's output is already the normalized value). Note the bf16/3/5/6-bit path: only `r2` has an `_add`, so a wide
matrix takes `r2` — my earlier `qmvg_f16g64_r2_add` guess was right by luck, not by rule.

`tools/spark_config_test.cpp` now transcribes both the instantiation set **and** the selection rule and sweeps
bits × K × N × mode (~570 combinations), asserting every name exists. Verified it fails when the rule is changed to
emit the nonexistent `qmv2_b8_g64_r2`.

### Defect 9 (latent, would have crashed): a staging buffer sized for q and used for q|k|v

`qo_` (q_dim floats) was made the target of all three fused-matvec row slices, which need `qkv_dim`. Once the buffer
roles were named (the code had one buffer serving three purposes under a comment that said "also used as scratch"),
this was a one-line fix; it is listed because it is the same root cause as defects 2/3 — a rule expressed in several
places at once.

### Defect 10 (silent, loses the prompt): `reserve` threw the KV cache away when it grew

`reserve` allocated fresh cache buffers when the context outgrew them and **copied nothing**. Every position before the
reallocation became uninitialised memory, and nothing errors — the model simply "forgets". It would trigger on the first
prompt longer than the initial capacity, and again on every growth. Rewritten to allocate, copy, commit, release (and to
respect `mem_budget_`, which it also ignored), reusing the base library's `kv_regrow` kernel — the same one
`Qwen35::grow_kv` uses. Its `uint3` is `(source stride, destination stride, Hkv)` in 16-byte words, as
`Qwen35::enc_kv_regrow` passes it. The landed Spark call passed `(old, old, target)` — the shape `kv_copy_prefix`
takes — so after the first growth every KV head after the first was copied at the old offset, and attention read zeros
from the new rows. That made the long-context path wrong even though the cache now preserved its bytes; fixed to
`(old, target, hkv)` and pinned by `check_kv_regrow` in `tests/test_spark_sampling_offset.py`.

### Defect 11 (would have corrupted memory): the ring caches were written at absolute positions

`spark_attn_prep` stored k and v at `pos` — the absolute token position — into a cache `cache_rows()` sized
`window + 8` = 520 slots for a sliding layer. Past position 519 that writes outside the buffer. The wrapped-window
*fatal* I had written only fires on the **read** side and only `if (ring && lo > 0)`, so at positions 512..519 the model
would have written past the end while reading the plain range — a corruption bug that the "we refuse what we do not
implement" message did not actually cover.

Fixed: `SparkPrepParams.slot_mask` (0 for a full layer, else the ring's `ring_rows_ - 1`) and the prologue stores at
`pos & slot_mask`. Full-attention layers append, so their slot is still the position. (The first fix used `window - 1`;
§14/R1 later showed the ring must be `window + prefill_chunk` for batched prefill.)

At that point the read side still refused a wrapped window (`lo > 0`); §12d subsequently implemented the cyclic read
in `attn_decode_partial` with the two appended fields. The copy-based alternative was rejected because
`kv_copy_prefix` cannot express a rotation of the ring (it copies from slot 0 and has no source offset).

### Still unverified, and honestly so

Everything in §6 that needs a GPU. The QA pass raised confidence in the *host-side* logic — config, kernel names,
buffer layouts, the activation choice, the norm conversion, the cache growth — and caught ten defects in it. It says
nothing whatsoever about whether the kernels compute the right answer: no GPU has executed a single one of them.

Specifically still unknown, and not to be asserted either way:

- Whether `spark_attn_prep`'s RoPE matches the reference numerically (the convention is verified by reading, the
  arithmetic is not).
- Whether the ring caches wrap correctly for contexts past 512 tokens — that path **does not exist yet** and stops with
  a fatal (see §6).
- Whether the interleaved q|gate layout is what `attn_decode_reduce` actually indexes at run time.
- Whether the GELU fold and the `_norm`-fused matvecs produce the same values as `transformers` on the same weights.
- Any speed number for any of it.

The PR stays "Not verified on a GPU", and the TODO's Mac checklist is the step that changes any of the above.

## 14. Prefill: why it is slow, and the decision it forces

Prefill ran one row at a time through the decode path. That is correct and it is not a bug — but it is at the
bandwidth floor, and the floor itself is the problem.

Every weight is re-streamed once per row, so a T-row prompt streams the whole model T times. For the 1.7B bf16
(1.71 G params, 3.42 GB) at, say, 273 GB/s that is 12.5 ms per row:

| prompt | row-at-a-time (the floor) | batched GEMM |
|---|---|---|
| 77 tok | 0.96 s | 0.013-0.053 s |
| 512 tok | 6.4 s | 0.09-0.35 s |
| 2048 tok | 25.6 s | 0.35-1.40 s |
| 5000 tok | 62.6 s | 0.85-3.42 s |

The server reported `prefill_ms: 831.7` for a 77-token prompt. That is the 0.96 s prediction. So the measurement and
the model agree, which is the useful part: there is nothing to hunt down here, and no amount of tuning the row loop
will beat the floor. The only fix is to stop streaming the weights once per row.

**Nothing in that list needs a new kernel.** Everything was already in the repo, benchmarked:

* `qrf_*` (`src/kernels/wfmt.metal`) — the format-generic prefill GEMM, and the bf16 tiles exist
  (`qrf_f16_m32_n128_mf`, `qrf_f16_m16_n64`, `qrf_f16_m8_n64`). `Qwen35::enc_qrf` is the dispatch ladder to copy
  verbatim: `M <= 8` -> `m8_n64`; `M < 64 || N <= 64` -> `m16_n64`; otherwise `m32_n128_mf`, with the M tiles along
  grid x so the threadgroups running together share a weight tile through the cache (bf16 weights are 4x the bytes of
  4-bit ones, which is why only the bf16 tile is built that way).
* `rows_to_half` (`gdn.metal`) — norm + float->half in one pass, already shared by four models. `qrf` reads `half`.
* `qattn_t_g4d2s8` (`src/kernels/tensor/qattn_t.metal`) — **G = 4, head_dim 256: Spark's exact geometry.** The M5 Max
  numbers are in `qwen35.cpp`: 3.09 / 7.34 / 15.95 ms against 4096 / 8192 / 16384 keys vs `qattn_t_s8k16`'s
  5.15 / 11.48 / 24.21 (1.52-1.67x), error 2.70 / 3.01 / 3.29e-4 against an fp64 reference. This is the find that
  makes the attention side cheap: Qwen3.5-4B has 16 query and 4 KV heads at D 256, the same shape Spark has.
* `spark_gelu_gu` stays. It is an elementwise pass over `T*ffn` — about 0.1% of a layer — so folding Spark's erf GELU
  into a shared GEMM's epilogue would buy nothing and would put the shipping models' numerics at risk (D4).

### The one genuine gap, and the decision

`qattn_t` reads the cache as `kb + slot * D` for `k0 = 0 .. tgl`, and its causal limit is `p.pos0 + min(tk, T - 1)`.
It has **no lower bound and no ring**. A full-attention layer is fine — its cache only appends, so slot == position.
A sliding layer is a ring holding the last 512 keys *wherever they fell*, so by the time a prompt is
longer than the window the kernel is reading the wrong slots entirely. On the 1.7B that is 21 of 28 layers. The ring
now has `window + prefill_chunk` slots; the slot mapping is still the position modulo that capacity.

The fix is two numbers: the oldest live key, and the slot mapping `(oldest + r) & slot_mask`. Where they go is the
decision:

**R1 — grow the shared params.** `AttnPrefParams` already ends `... max_seq, scale`; append `uint wrapped, slot_mask`
(or one packed field) so nothing before them moves. This is precisely what was done for the decode path in v0.2.28:
`AttnDecParams` gained `wrapped`/`slot_mask` the same way, and linear-cache models pass `{0, 0}` and read bit-identically.
Precedent, and the least new code. It does touch `attn.metal` and `tensor/qattn_t.metal` — kernels the shipping models
run — so it is a real (if small) risk to them, and the D4 rule exists to stop exactly that.

**R2 — a Spark-only `spark_attn_prefill`.** Nothing shared changes; D4 stays pure. The cost is a new kernel to
write, benchmark and validate against an fp64 reference — and it would be the only part of Spark's prefill with no
benchmark evidence behind it. I started writing one and stopped: it is a much worse bet than reusing a kernel that
already has published numbers for this exact head geometry.

Decision (taken): **R1** — the user chose it. The precedent was one release old, the change is two fields, and the
alternative traded a known benefit for an unmeasured kernel.

### R1, as landed
`AttnPrefParams` (attn.metal) and its hand-kept mirror `QaParams` (tensor/qattn_t.metal) both gained
`uint window; uint slot_mask;` **appended**, after `scale`. Appending is what keeps the change contained: a brace
initialiser with six fields zero-fills the rest, so `clover.cpp`, `qwen35.cpp` and `qwen4_exp.cpp` are untouched and pass
the linear `{0, 0}` — and with `slot_mask == 0` the slot is the position, with `window == 0` the first live key is 0 and
the floor mask (`k0 < kstart`) is `k0 < 0`, which never fires. `tools/ring_prefill_test.cpp` pins that identity.

Two things the change needed that were not obvious from outside the kernels:

* **The tile loop had to start at the ring's first live tile**, not just mask from 0. Masking alone is correct but makes
  a sliding layer's prefill proportional to the absolute position rather than to the window: at 200k tokens, starting
  at 0 would walk ~6,200 tiles that hold no live key. `it_begin = kstart / BK`, and for the split-K variants
  (`attn_prefill11gs`) `it_begin = max(it_begin, kstart / BK)`, since a split otherwise picks dead tiles.
* **`QaParams` is a mirror, not an include.** The tensor library is its own translation unit
  (cmake/embed_kernels_tensor.cmake globs `src/kernels/tensor/*.metal` separately), so it cannot see `attn.metal`'s
  struct. Appending a field to one and not the other would leave the tensor_ops prefill reading the wrong offsets
  *silently*, on the only GPUs that run it. Two `static_assert`s in `kparams.h` now pin `sizeof(AttnPref) == 32` and
  `offsetof(AttnPref, window) == 24`.

**A vacuously-passing test, again.** `ring_prefill_test`'s first version had the reference compute slots through the same
`slot_of` helper as the kernel side, so mutating `slot_of` to `return pos` broke both sides together and the comparison
still passed. The reference now computes the slot as an explicit `p % window`. Both mutations (no floor mask, slot == pos)
now fail: 7,099 and 6,304 cases.

### The batched prefill itself
`encode_prefill` now runs T rows per pass (`enc_attn_prefill` / `enc_mlp_prefill`), with `rows_to_half` + `qrf_proj`
(Qwen35::enc_qrf's tile ladder, verbatim; on M5, 4-bit g32/g64 takes `qmt_m64_n64*`) and `qattn_t_g4d2s8` (or
`attn_prefill11r` off the tensor-ops path) for the attention. `AttnPref fp` carries the ring:
`{window, ring_rows_ - 1}` for a sliding layer, `{0, 0}` for a full one.

Details that had to be right, each of which was wrong on the first attempt:

* **The residual starts as the embeddings.** `enc_attn_prefill`'s out_proj *adds* into `xb_`, and so do gate/up and
  down, so `enc_embed` must write `xb_` first and nothing may `add` into something stale.
* **k/v staging cannot be the per-row `kr_`/`vr_`.** `spark_attn_prep` reads `[t][Hkv][D]` from buffers 1 and 2; the
  decode path's buffers hold one row. They are sliced out of the fused projection's own T-row output instead.
* **The gate is read while the packed q|gate is written.** One dispatch, two buffers: `gub_` in, `qgb_` out. In place
  would race the gate reads of rows still to come.
* **`spark_activation_bytes` had to start counting the chunk.** It ignored `prefill_chunk` deliberately ("one row
  whatever the chunk"); batched prefill makes that false, and the old value under-reported ~150 MB on the 4B. It now
  counts fp32 and half rows separately — folding a `/2` into an expression multiplied by `sizeof(float)` is how the first
  version came out 3.6x too small. `spark_config_test` cross-checks the plan against `alloc_state` at four core counts
  and four chunk sizes.
* **A mid-pass snapshot becomes its own short pass**: rows `[0, snap_row)`, then `save`, then the rest. Two passes
  instead of one, and only when a snapshot is requested.
* **`encode_prefill` reserves `pos0 + T + 1`**, because a sampling pass writes the row after its last — the slot
  requirement `encode_step` already encoded as `pos + 2`.

### Also worth knowing before this is wired
* `qattn_t` runs only on `apple_family() >= 9` (tensor_ops, gated by `qmt_on()`). An M1-M4 needs a fallback —
  `attn_prefill11r` or the short-prefill split-K path — so whatever lands must keep a non-tensor_ops branch.
* The GELU means the MLP cannot fuse gate|up into one GEMM the way Qwen35's `enc_qmm_gu` does: `enc_qmm_gu` fuses
  silu *and* the gate>*up multiply inside the GEMM. Spark would run two `qrf` dispatches plus `spark_gelu_gu`. A
  third dispatch per layer over an `ffn`-sized elementwise pass, which is negligible next to the GEMMs.
* Struct-of-arrays detail: `x_` is `float`, `stg_`/`kr_`/`vr_`/`qr_`/`qg_`/`gate_`/`gu_`/`mh_` are per-row. Batched,
  they all need `x T`, and `xh_`/`hh_` (`half`, `H x T`) are new. `spark_activation_bytes` must grow with
  `prefill_chunk` — done: it counts the batched set (see "R1, as landed" above).
  lands"), so this is a coupled change, not an optional one.

## 15. Speed: the reference implementation says we are running the wrong weights

`github.com/XHToken/Spark-MLX-LLM` is XHToken's own inference project for Spark 2.6/2.5 (MLX). Reading it against this
port answers the speed question, and the answer is not in the kernels — it is in the checkpoint.

### The finding

Their converter's default profile (`spark_mlx_llm/conversion.py`) is:

| module | weights | activations |
|---|---|---|
| `q_k_v_proj`, `mlp.{gate,up,down}_proj` | **INT4 group 32**, asymmetric | BF16 |
| `out_proj` | INT8 per-channel → stored as group 32 | per-token INT8 (W8A8) or BF16 |
| `lm_head` | INT8 per-channel → group 32 | as above |
| `g_proj`, `embedding`, RMSNorm | BF16, unquantized | — |

All weights ~4 bits. This port is running **bf16**, because that is the only Spark checkpoint downloaded here. That is
the entire speed gap, and it is ~7x:

| 1.7B | size | decode ceiling @400 GB/s |
|---|---|---|
| bf16 (what we run) | 3.42 GB | 117 tok/s |
| INT4/g32 (their default) | 0.48 GB | ~840 tok/s |

A batched prefill pass streams the weights once, so the same ratio applies there: 8.6 ms vs 1.7 ms per pass at 400 GB/s.

### What is already runnable, today, with no code change

`mlx-community/Spark-X2.6-4B-OptiQ-4bit` exists and its `config.json` uses **group_size 64 with per-tensor mixed
4/8-bit** — i.e. `f4g64` / `f8g64` / `f16`, exactly the formats `tools/spark_kernel_names.py` confirms this engine
instantiates (`qmvg_f{3,4,5,6,8}g64`, `qrf_f{3,4,5,6,8}g64_*`, `qmv2_b8_g64_r*`). 8.08 GB bf16 → 2.76 GB, about 2.9x.
No kernel work. The 1.7B has no `mlx-community` 4-bit build (404/401); `XHToken/Spark-X2.6-1.7B-INT4` exists but is
compressed-tensors and would need `spark-mlx-convert`, which needs macOS >= 26.2 and the Metal 4 toolchain.

### What their INT4/g32 profile would need from us

Group 32 is now covered. The engine instantiates `f4g32` and `f8g32` in `NI_WF_LOW`, and `matvec_pipe` sends every
group-32 linear to `qmvg_f{4,8}g32` rather than to `qmv2_b*_g64`/`qmv3_b*`, which hard-code `GS == 64`. Prefill has
`qrf_f4g32/f8g32` tiles and `qmt_m64_n64_g32` for 4-bit on M5. The loader also accepts the reference's F32 `lm_head`
coefficients and rounds them to bf16 once, and reads a separate `lm_head` when `tie_word_embeddings` is false.

### The lesson, stated plainly

§14 concluded that prefill's problem was structural (row-at-a-time → batched) and that was true. But batching a bf16
checkpoint still streams 3.42 GB per pass. **Quantization multiplies the fix rather than substituting for it**: batched
INT4 is ~7x on top of batched bf16. The order of work should have been format first, then batching — the format is a
download, the batching was a week.

## 16. Current state (2026-10-30): the reference checkpoint, batched prefill, and the M5 path

### 16a. Checkpoint support

The loader now accepts the reference converter's default output, not just a bf16 folder:

| tensor group | reference output | this port |
|---|---|---|
| `q_k_v_proj`, `gate_proj`, `up_proj`, `down_proj` | INT4 affine, group 32 | `f4g32`; `qmvg` decode, `qrf`/`qmt` prefill |
| `out_proj`, `lm_head` | INT8 affine, group 32, F32 coefficients | `f8g32`; F32 scales/biases rounded to bf16 once at load |
| `g_proj`, `embedding`, norms | BF16 | bf16 path |
| `tie_word_embeddings` | converted to `false` for the mixed profile | separate `lm_head` is loaded; a leftover head on a tied config is ignored |

The public HF checkpoints are bf16 and tied; those still load (the format is chosen from the tensors, not the name).
The converter's W8A8 **activation** quantization is not reproduced: weights are dequantized and multiplied in bf16.

### 16b. Prefill path

`encode_prefill` processes a whole chunk (`prefill_chunk`, default 2048; `--prefill-chunk` may raise it) per pass:

1. `enc_embed` into the fp32 residual, then per layer `enc_attn_prefill` + `enc_mlp_prefill`.
2. Normalization once per matrix (`rows_to_half`), then one GEMM per projection. On M5, 4-bit g32/g64 uses
   `qmt_m64_n64*` (Metal 4 `tensor_ops`); off M5, `qrf` `m32/m16/m8` tiles. W8 and bf16 stay on `qrf`.
3. Attention: `qattn_t_g4d2s8` on tensor-ops GPUs (16 tokens × 4 heads per threadgroup), else `attn_prefill11r`.
   Both read the sliding ring with the block's first-query floor and mask each row's own floor.
4. Down projection adds into the fp32 residual; the pass's last row samples from that fp32 row with the final norm
   folded into the head matvec.

The ring is `next_pow2(window + prefill_chunk)` slots (4096 at 2048). Both prefill kernels and decode use
`pos & (ring_rows_ - 1)`, so the write and the three reads agree. Memory: 176.2 MB (1.7B) / 453.0 MB (4B) for all
sliding layers, independent of context.

### 16c. Defects fixed in this pass

- **Ring capacity**: a batched pass writes T rows before attention reads, so a 512-slot ring overwrote the first
  query's oldest keys. Capacity is now window + chunk; `ring_slot_test` and `ring_prefill_test` encode the per-query
  floor, not just the block's last-key floor.
- **Fused QKV layout**: prefill read `q`, `k`, `v` as three contiguous T-row blocks, but the GEMM writes one
  `[T][qkv_dim]` matrix. `SparkPrep` now carries row strides and per-tensor offsets.
- **MLP gate/up overlap**: the up view started at float offset F instead of `T*F`, so up row t overwrote gate row t+1.
- **Sampling input**: prefill sampled the half normalized row and normalized it again; it now samples the fp32
  residual row exactly as decode does.
- **Untied head / F32 coefficients**: accepted and rounded once, instead of refused.
- **Group-32 decode routing**: `qmv2_b*_g64`/`qmv3_b*` hard-code a 64-value group. Group-32 now goes to `qmvg_f4g32`
  / `f8g32`; the CPU sweep fails if the rule emits a missing or wrong-group kernel.
- **Prefill attention selection**: the M5 path uses `qattn_t_g4d2s8` for Spark's 4:1 GQA; `NI_ATTN` still forces the
  shared fallback.

### 16d. What is verified, and what is not

Run on this Linux box (no GPU, no Metal compiler):

- `python3 tests/test_metal_shared_helpers.py` → 49 cross-file helper arities match.
- `g++ tools/ring_slot_test.cpp` and `g++ tools/ring_prefill_test.cpp` → pass (ring slots, laps, per-query chunk sets,
  one slot per live key).
- `g++ tools/spark_config_test.cpp src/model/spark_config.cpp` against both published configs → pass (config, refusal
  set, plan vs allocation, matvec kernel names including f4g32/f8g32).
- `g++ -fsyntax-only src/model/spark.cpp` → pass.

Not verified here: Metal compilation (`xcrun metal` is macOS-only), kernel arithmetic on a GPU, logits against the
reference, and every speed number. The Mac pass should run, in order:

```bash
# converted checkpoint (or an MLX 4-bit pack), then:
kapitsa serve --model <dir> --prefill-chunk 4096
kapitsa bench --model <dir> --prefill-chunk 4096
# compare logits with tools/logit_compare.py against the reference implementation
```

Measure bf16 against the INT4/g32 conversion on the same prompts: the prefill delta should track the 7x weight-byte
reduction plus the qmt tensor-ops gain; decode should track the same reduction. Do not quote a number until it is run.

## 17. Decode dispatch fusion, and the half activation (2026-10-31)

Two changes, both in the decode path; neither changes prefill math beyond deleting a pass.

### 17a. Four projection dispatches per layer became two

Spark's decode attention already described q/k/v as three row ranges of one `q_k_v_proj`, but only took a
multi-segment matvec for 4/8-bit group-64 weights and then ran `g_proj` separately. `Spark::multi_proj` now chooses
the same two host-side pathways as `Qwen35::enc_qmv_multi`: `qmvg_multi_<fmt>` when every segment has one format, and
`qmv2/qmv3_multi_r2` for 4/8-bit group 64 (per-segment bits). It fuses:

- q/k/v/gate into one dispatch when `g_proj` shares the format (bf16 in both live checkpoints, or all-quantized), and
  q/k/v into one plus a separate gate otherwise (the reference converter leaves `g_proj` bf16 under a quantized qkv);
- `gate_proj` and `up_proj` into one dispatch when they share a format.

The old q/k/v multi path also bound the whole `q_k_v_proj` tensor for all three segments. `qmv2_multi` addresses each
segment from row 0, so k and v were computed from the q rows. The new path binds `w`/`s`/`b` at row 0, `qd`, and
`qd + kd`; the static test checks those three offsets rather than trusting the filler.

### 17b. One fewer prefill pass over `T*ffn`

Decode matvecs read fp32 x (`qmv2`, `qmv3` and `qmvg` all take `const device float* x`), so the decode epilogue's
fp32 output into `mh_` is correct and stays. The prefill down projection is a `qrf` GEMM and reads half; that path
wrote the product to an fp32 buffer and then ran `rows_to_half` over it. `spark_gelu_gu_half` now writes half straight
into `hh_`, so the separate pass is gone (the cast is the pass, at the only place the value is produced). Decode and
prefill therefore run two instantiations of the same four-line epilogue, each writing the type its consumer reads.

### 17c. What this does and does not fix

Decode is weight-bandwidth bound in bf16: the 1.7B streams 3.42 GB per token, which at ~80 tok/s is ~274 GB/s, the
ceiling of the machine that produced that number. Fusing four dispatch launches per layer (one fewer when `g_proj`'s
format differs) removes launch overhead, not bytes; each segment still normalizes x in its own threadgroups, so the norm
work is unchanged. Expect a few percent, not a multiplier. The 2.9x smaller MLX 4-bit pack is still the only large
decode win, and it now goes through the corrected multi path. Prefill remains limited by the GEMMs; this pass removes
one `T*F` read/write per layer, it does not change their FLOPs.

Verified on Linux (no Metal compiler): `tools/lx_syntax_check.sh`, `tests/test_spark_sampling_offset.py`,
`tests/test_metal_shared_helpers.py`, and `spark_config_test` against both published configs. GPU arithmetic, logits,
and speed still need the Mac pass.

## 18. Live context scaling (v0.2.36 endpoint, 1.7B bf16)

The running endpoint (`Spark-X2.6-1.7B`, one slot, default chunk; these changes are not in it yet) was probed with
greedy requests and measured from `usage.kapitsa.prefill_ms` plus wall time. Decode is `(wall - prefill_ms)` over the
completion tokens, so the 16-token rows carry scheduling noise; the shape is the point.

| prompt tokens | prefill ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 1,604 | 830 | 1,933 | 74.4 |
| 6,879 | 3,310 | 2,078 | 73.8 |
| 13,689 | 7,340 | 1,865 | 70.2 |
| 21,860 | 13,160 | 1,661 | 64.8 |
| 25,604 | 16,110 | 1,589 | 64.5 |
| 42,627 | 32,470 | 1,313 | 56.3 |

The 25.6k and 42.6k rows used 32- and 16-token completions respectively. The 42.6k row was taken with the server
otherwise idle; a 51k row measured while an interrupted 73k prefill was still finishing read 2.2 tok/s and is not a
valid data point.

Decode loses ~27% from 1.6k to 42.6k. That is the full-attention KV: 7 of 28 layers read `span * 2 * 256 * 2 * 2`
bytes each token (611 MB at 42.6k), while the other 21 read a fixed 512-key ring. Qwen3.5's GDN layers have a constant
state, so they do not pay this; a sliding-window transformer cannot match that without changing the model. The fixes
that do not change the architecture are quantized weights (fewer weight bytes), quantized KV (fewer cache bytes), or
speculative decoding (fewer passes over both); Spark has no MTP head for the last.

`--prefill-chunk 4096` changes neither the GEMM FLOPs nor the weight bytes. For a prompt longer than the chunk it only
reduces the number of chunks and the repeated attention over the prefix: with `C` chunks of `s` rows the extra pair
count is about `T*s/2` against a causal floor of `T^2/2` (8% at T = 25k, s = 2048). A/B it on the Mac before making it
a default; it doubles the ring (`next_pow2(window + chunk)`) and the prefill activation set.

## 19. KV q8_0: yes on memory, second on speed

Q8_0 is one fp16 scale per 32 values plus 32 int8 values: `34/32 = 1.0625` bytes per value, **53.1% of half**, a
46.9% reduction. Spark's `head_dim` is 256, so a row has eight blocks; 256 B of int8 followed by 16 B of fp16 scales
is 272 B and 16-byte aligned, which keeps every copy/regrow operation in whole `uint4`s. Per position, K and V
together cost `hkv * 2 * 272` bytes instead of `hkv * 2 * 512`:

| | half | q8_0 | saving |
|---|---:|---:|---:|
| 1.7B (hkv 2), full layers, B/pos | 14,336 | 7,616 | 46.9% |
| 1.7B, full KV at 42.6k | 611 MB | 325 MB | 286 MB |
| 1.7B, full KV at 73k | 1.05 GB | 0.56 GB | 0.49 GB |
| 4B (hkv 4), full layers, B/pos | 36,864 | 19,584 | 46.9% |

That is real memory, and it doubles the context a fixed budget can hold. It is **not** a proportional decode win while
the weights are bf16: at 42.6k the per-token stream is 3.42 GB of weights + 0.61 GB of KV; q8_0 makes it 3.42 + 0.33,
about 7% fewer bytes. With the 4-bit pack (~0.9 GB of weights) the same comparison is 1.51 -> 1.23 GB, ~19%; at 73k,
1.95 -> 1.46 GB, ~25%. Prefill is GEMM-bound, so expect less there; the attention kernels read the cache, but their
share is smaller than the projections'.

The implementation is a cache-format change, not a flag:

- `new_seq` / `reserve` / `kv_regrow`: 272 B per (head, row), `n16 = 17`, full layers only for the growing cache; the
  sliding rings can use the same row (they are fixed 4,096 slots at chunk 2048, so this saves memory, not time).
- `spark_kv_per_pos`, `MemPlan`, and `disk_format` (bump the version: stored caches change layout).
- `spark_attn_prep`: quantize each 32-value block after RMSNorm/RoPE. A threadgroup per (row, head) reduces the block
  max, writes the scale, then the int8 values.
- Every consumer needs a q8 variant: `attn_decode_partial` (decode), `attn_prefill11r` (prefill), and
  `qattn_t_g4d2s8` (M5 prefill). Dequantizing a tile into threadgroup half before the existing dot is the least
  invasive form; doing the dot in int8/`dp4a` is a second optimization, not required for the bandwidth win.
- Accuracy: Q8_0 K/V is usually close to lossless, but K enters the softmax directly, so compare logits and a
  long-context needle before trusting it. V can stay at a higher precision if K is the only problem.

Order of work: the regrow-stride fix above (correctness), then the 4-bit weight pack (the largest decode lever), then
q8_0 KV (memory and the long-context slope). Speculative decoding is fourth: Spark has no MTP head, so it needs a
separate 1.7B drafter or prompt-lookup, and at 42k both paths still read the same KV unless the draft is verified in a
batch.

## 20. 4-bit 1.7B endpoint: chunk 2048 vs 4096

`v0.2.37-test1`, the 1.7B 4-bit pack (`mlx_model` on the test Mac), one slot, greedy, `max_tokens` 160. Same prompts
and the same access-code needle as §18. Decode is `(wall - prefill_ms) / completion_tokens`.

| prompt tokens | chunk | prefill ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|---:|
| 4,355 | 2048 | 2,890 | 1,509 | 134.3 |
| 4,355 | 4096 | 2,780 | 1,564 | 132.1 |
| 19,675 | 2048 | 14,960 | 1,315 | 111.9 |
| 19,675 | 4096 | 14,760 | 1,333 | 109.4 |
| 40,951 | 2048 | 37,670 | 1,087 | 89.8 |
| 40,951 | 4096 | 37,640 | 1,088 | 89.6 |
| 68,185 | 2048 | 78,270 | 871 | 68.8 |
| 68,185 | 4096 | 77,510 | 880 | 71.6 |

Chunk 4096 is within noise of 2048 (largest difference +3.6% at 4.3k, +1% at 68k). That is the expected result for a
dense model: a wider chunk does not reduce the projection FLOPs or the weight bytes, it only removes repeated attention
over the prefix (the §18 estimate: ~8% of the causal pair count at T = 25k). The extra activation and ring memory is not
worth it; keep 2048. The prefill lever is the 4-bit GEMM/dequant path, not the chunk.

The 4-bit pack is a decode win, not a prefill win. Against §18's bf16 numbers the decode is ~1.6-1.8x (134 vs 74 tok/s
short, 90 vs 56 at ~42k), while prefill is lower at comparable lengths (1,315 vs 1,661 at ~20k; 1,087 vs 1,313 at
~42k). Those runs are not a controlled A/B (different builds and sessions), but there is no sign that 4-bit weights
speed up the GEMM-bound prefill, and the dequant work is a plausible cost.

The 68k needle (code at the start of a 68,185-token prompt) was recalled. Multi-turn runs crossing the initial
1024-token cache allocation (1,329 and 8,633 prompt tokens, with the first turn's code retained) also recalled it; the
second turn's prefill reused the prefix (~0.49 s for the 1.3k case, 5.36 s for the 8.6k case) rather than re-reading the
first turn. That exercises the fixed regrow path on the GPU, not only the host-side test.
