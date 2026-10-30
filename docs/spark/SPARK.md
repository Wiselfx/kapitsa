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
| MLP act | `gelu` (tanh approx) | `gelu` (tanh approx) |
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
Kapitsa's supported quant formats (`format_supported`, `src/model/qwen35.cpp`): bf16; group 64 at 3/4/5/6/8 bit;
group 32 at 4 bit.

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
1. **GELU-tanh MLP.** Kapitsa is SiLU throughout. Spark needs `gelu` (PyTorch's `gelu`, i.e. tanh approximation) in the
   fused gate/up+SwiGLU store sites of the GEMM/matvec kernels. Add it as a compile-time template flag (`GELU`) so SiLU
   instantiation stays bit-identical for every existing model — this is the key numerics-safety property: **existing
   models must compile to exactly the same kernel code as before.**
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
- **P3 — Attention path.** Prefill + decode with per-layer `rot`, window masking, all-ones q/k norm, per-head gate
   broadcast, correct KV cache sizing (sliding layers only need a 512-slot ring).
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
  - `tools/spark_config_test.cpp` — CPU test over both published configs + 13 refusal cases. **Runs and passes on
    this Linux box.** Verified it has teeth (deliberately broke `rot_dims` → 2 failures).
  - `src/model/spark.{h,cpp}` — the `Spark` model: load (per-tensor mixed quant, fused `q_k_v_proj` row-slicing, tied
    embeddings, HF vs MLX norms decided from values), KV caches (512-slot rings for sliding layers, full for the 9/7),
    the full decode path, and prefill as a loop over decode rows.
  - `src/kernels/spark/spark.metal` + `cmake/embed_kernels_spark.cmake` + CMake/CI wiring.
  - Engine dispatch (`engine.cpp`), the "model Kapitsa cannot run" refusal, the memory plan, and the no-drafter gate.
  - `tools/lx_syntax_check.sh` — local Linux syntax check (see §11).
- **A real bug caught by reading the kernel contracts, not by compiling:** `attn_prep` reads the query out of an
  **interleaved `[row][Hq][2][D]`** buffer (Qwen's `q_proj` emits query+gate together, `attn.metal:35`), while my first
  version handed it the dense `q_k_v_proj` output. Fixed with `spark_pack_qg`, which assembles the interleaved layout
  from the dense q and the broadcast per-head scalar. This is exactly the class of mistake only a GPU run confirms —
  see §6.
- **Known gaps (P3 remainder), all stated in code comments and `TODO.md`:**
  - A **wrapped sliding window** is not implemented: `enc_attn` runs the plain range while `pos < window` and stops
    with a clear fatal once the window wraps. Needs the ring-aware attention kernel.
  - Prefill is correct-but-slow (decode path per row); the batched path is not bound.
  - `encode_logits_rows` works but is slow for the same reason.
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
- Next, in order: ring-aware attention (the wrapped-window fatal), then batched prefill + GELU fold in the GEMM tile,
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

- **Host-side logic: reviewed, and it had four defects, now fixed.** Config parsing, kernel-name construction, the
  fused-qkv byte-offset slicing and the GELU choice are covered by `tools/spark_config_test.cpp` (runs in CI) or by
  explicit asserts.
- **Kernel arithmetic: entirely unverified.** No GPU has run any of it. Do not describe it as working.
- **Two known functional gaps**, both fatal-with-a-clear-message rather than silently wrong: a sliding window that
  wraps past 512 tokens, and no batched prefill (correct but slow). See §6.


1. Read §2 (constraints), §4 (decisions D1–D7), then the last §9 entry.
2. `docs/spark/TODO.md` is the live checklist — trust it over your memory of this file.
3. Run `bash tools/lx_syntax_check.sh` and the config test before changing anything, so you know the baseline is green.
4. Do not claim anything about Metal, speed or output equality. The PR says "Not verified on a GPU" because it is.

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
`Qwen35::grow_kv` uses. Note the parameter packing: `kv_regrow` takes `(positions, src cap, dst cap)` in 16-byte words,
and my first attempt passed `(old, target, hkv)`, which would have copied the wrong strides.

### Defect 11 (would have corrupted memory): the ring caches were written at absolute positions

`spark_attn_prep` stored k and v at `pos` — the absolute token position — into a cache `cache_rows()` sized
`window + 8` = 520 slots for a sliding layer. Past position 519 that writes outside the buffer. The wrapped-window
*fatal* I had written only fires on the **read** side and only `if (ring && lo > 0)`, so at positions 512..519 the model
would have written past the end while reading the plain range — a corruption bug that the "we refuse what we do not
implement" message did not actually cover.

Fixed: `SparkPrepParams.slot_mask` (0 for a full layer, else `window - 1`) and the prologue stores at
`pos & slot_mask`. Full-attention layers append, so their slot is still the position.

The read side still refuses a wrapped window (`lo > 0`), and the refusal now names the three ways to implement it and
says why one should be chosen after the Mac pass: every option changes the indexing of `attn_decode_partial`, which is
a kernel six other models depend on and which nothing on this machine can test. I started writing the copy-based
version, found that `kv_copy_prefix`'s kernel cannot express a rotation of the ring (it copies from slot 0 and has no
source offset, so it would have written garbage), and reverted rather than ship an untested rotation of a shared
kernel's contract.

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
