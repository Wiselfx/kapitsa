# Spark port — live checklist

Update this file as work lands. `[x]` done, `[~]` in progress, `[ ]` not started, `[!]` blocked.
The plan and rationale live in `SPARK.md` (read §2, §4, §12 after a compaction); this file is only the state.

## P0 — scaffolding  ✅
- [x] Fork `Wiselfx/kapitsa`, clone, constraints recorded (SPARK.md §2)
- [x] Confirm `Qwen35` cannot host Spark (GDN fields + q_norm/k_norm) — decision D1
- [x] Capture both `config.json` verbatim into `docs/spark/configs/`
- [x] `docs/spark/reference/` — reference `modeling_spark.py` + `configuration_spark.py`
- [x] `docs/spark/configs/NOTES.md` — derived dims, layer tables, KV costs, forward-pass order
- [x] `SPARK.md` + this file
- [x] `src/model/spark_config.{h,cpp}` — pure-CPU parse + validation + derived tables
- [x] `src/model/spark.{h,cpp}` — class, load, caches, decode path
- [x] Engine dispatch, refusal text, memory plan, no-drafter gate
- [x] CMake: kernel embed script + `add_test`
- [x] CI: build target, test step, metal-compile library
- [x] `tools/lx_syntax_check.sh` — local Linux verification loop

## P1 — loading  ✅ (unverified on a GPU)
- [x] Tensor mapping: fused `q_k_v_proj` sliced by exact byte offsets into q/k/v; `g_proj` as `[Hq, H]`
- [x] Tied or untied output: `tie_word_embeddings` decides whether `lm_head` or the embedding table is the head; a
      stray `lm_head` on a tied config is ignored, a missing one on an untied config is fatal
- [x] Per-tensor mixed quant (MLX affine 3/4/5/6/8 g64, 4/8 g32, bf16) from `config.json`'s `quantization`;
      group-32 decode routes to `qmvg_f4g32`/`f8g32`, never to the group-64 kernels
- [x] F32 `lm_head` scales/biases (the reference converter's output) rounded once to bf16 at load
- [x] HF-vs-MLX RMSNorm decided from the values (`zero_centred`)
- [ ] **Verify with a real checkpoint** (needs a Mac; `kapitsa doctor --models-dir <dir>` then a short generation)

## P2 — GELU  ✅
- [x] Own kernel library `src/kernels/spark/` (decision: additive, do **not** touch shipping MLP kernels, SPARK.md §10)
- [x] `spark_gelu_gu` — **exact** gelu (`x * Phi(x)`, erf form), gate*up fold, not the tanh approximation:
      `hidden_act: "gelu"` is ACT2FN's `GELUActivation` (`nn.functional.gelu(approximate='none')`), and
      `gelu_pytorch_tanh` is a *separate* ACT2CLS key Spark does not ask for. Two output forms: fp32 for decode
      (all decode matvecs read fp32 x) and half for prefill's qrf down projection, which removes the separate
      `rows_to_half` pass (SPARK.md §17b). This checklist said "tanh" until QA pass 1 caught it, so the code and the
      docs disagreed — see SPARK.md §13 defect 1. Metal has no erf, so
      spark_erf is A&S 7.1.26 (max error ~1.4e-7)
- [x] Existing models' kernel libraries byte-identical (nothing in them changed)

## P3 — attention  [~]
- [x] Per-layer `rot` (64 full / 256 sliding) and per-layer `theta` (5e6 / 1e4)
- [x] **`spark_attn_prep`** — Spark's own prologue (RoPE + q|gate packing + cache append). It replaces `attn_prep`,
      which cannot be reused: that kernel *always* RMS-normalizes q and k, and Spark has no q/k norm at all (the
      reference ropes the raw `q_k_v_proj` output; zero occurrences of `q_norm`/`k_norm` in modeling_spark.py).
      Passing an all-ones weight does not disable it, and rescaling q and k by a common per-row factor changes
      `softmax(q·k/√D)` — it is not a gauge freedom. The earlier "all-ones buffers" plan was wrong.
- [x] Per-head sigmoid gate: one scalar broadcast over the head's 256 dims
- [x] KV: `window + prefill_chunk` rings for sliding layers, full cache for full layers (`reserve` never grows a ring)
- [x] Decode path end to end
- [x] **Wrapped sliding window.** `attn_decode_partial` gained a cyclic read (`AttnDecParams::wrapped`/`slot_mask`),
      so a sliding layer attends over its ring past 512 tokens. The insight that made it small: attention is a
      *weighted sum over a set* of keys, so reading the ring in ring order is the same softmax as reading it linearly
      -- a change of rotation, not a reordering. No compaction pass, no second buffer, no kernel-side sort: the tile
      load just starts at the oldest live key and wraps. Depends on window being a power of two and a multiple of
      kTile (both true; `tools/ring_slot_test.cpp` checks them). Linear-cache models pass `{0, 0}` and read exactly as
      before.
- [x] **SSD prefix cache hooks.** `disk_format()` was returning a format while `disk_rows`/`disk_state` were the base
      class's no-ops, so the cache enabled itself, stored states that hashed as empty, and `disk_restore` matched them
      and reported success -- a conversation continuing from an uninitialised cache. Implemented the hooks (linear
      caches as rows, rings whole, as Dory does) and added `Model::disk_hooks_live` + `tests/test_disk_hooks.py` so no
      model can advertise a format it cannot describe.
- [x] **Snapshots** (`save`/`restore`/`can_snapshot`). Copies only the sliding-window rings, as Dory does: a full
      layer's cache only appends, so truncation reads the same bytes, while a ring must be saved before it is
      overwritten. `encode_prefill` now honours `snap`/`snap_row` (it must, since `can_snapshot()` is true and the
      engine relies on it).
- [x] Ring-aware attention read in decode (`attn_decode_partial`), prefill (`attn_prefill11r` / `qattn_t_g4d2s8`)
- [x] **Batched prefill** — `encode_prefill` runs T rows per pass: `rows_to_half` + `qrf_proj` (Qwen35::enc_qrf's tile
      ladder; `qmt_m64_n64*` for 4-bit on M5) for the projections, `qattn_t_g4d2s8`/`attn_prefill11r` for attention,
      `spark_gelu_gu_half` writes the exact-GELU product straight into `hh_` (no `rows_to_half` after it). A mid-pass
      snapshot runs as its own short pass.
- [x] **Ring-aware prefill attention (R1)** — `AttnPrefParams` / `QaParams` gained `window`/`slot_mask` **appended**, so
      the 6-field brace initialisers in `clover.cpp` / `qwen35.cpp` / `qwen4_exp.cpp` still compile and zero-fill them.
      Two `static_assert`s in `kparams.h` pin the layout (32 bytes, `window` at 24) against the hand-kept mirror.
      `tools/ring_prefill_test.cpp` checks the key set over laps, the linear identity, and per-query chunk floors
      (a ring of window + chunk); both mutations of the ring logic fail it. See SPARK.md §14, "R1, as landed".
- [ ] **Ring-aware prefill attention is not verified on a GPU** — the test proves which keys are selected, not that the
      Metal kernels compile or that the scores match. `metal-compile` in CI is the only check available off a Mac.

### Batched prefill — the plan, and the measurements behind it

**Measured cost model** (Spark-X2.6-1.7B bf16 = 1.71 G params = 3.42 GB; every weight is re-streamed once per row, so
a T-row prompt streams 3.42 GB x T):

| prompt | row-at-a-time (bandwidth floor) | batched GEMM (fp16 compute) |
|---|---|---|
| 77 tok | 0.96 s | 0.013-0.053 s |
| 512 tok | 6.4 s | 0.09-0.35 s |
| 2048 tok | 25.6 s | 0.35-1.40 s |
| 5000 tok | 62.6 s | 0.85-3.42 s |

The measured `prefill_ms: 831.7` for a 77-token prompt sits **on** the 0.96 s prediction, so the current path is not
merely slow: it is at the bf16 bandwidth floor, which is the fastest a row-at-a-time loop can be. Nothing here is a
bug to fix; it needs the batched form (19-30x).

**Do not write new kernels.** The repo already ships and benchmarks every piece; the work is wiring, not authoring:

1. **Projections** — `qrf_*` (`wfmt.metal`, the format-generic prefill GEMM). bf16 instantiations confirmed present:
   `qrf_f16_m32_n128_mf`, `qrf_f16_m16_n64`, `qrf_f16_m8_n64`. Dispatch ladder copied from `Qwen35::enc_qrf`
   (`M <= 8` -> m8_n64; `M < 64 || N <= 64` -> m16_n64; else m32_n128_mf with the M tiles along grid x).
   Input must be `half`: `rows_to_half` (`gdn.metal`) does norm + float->half in one pass and 4 models already reuse it.
2. **Attention** — `qattn_t_g4d2s8` (`src/kernels/tensor/qattn_t.metal`) is **Spark's exact geometry**: G = 4,
   head_dim 256, benchmarked on an M5 Max at 3.09 / 7.34 / 15.95 ms against 4096 / 8192 / 16384 keys vs
   `qattn_t_s8k16`'s 5.15 / 11.48 / 24.21 (1.52-1.67x), error 2.70-3.29e-4 against an fp64 reference. Only on
   `apple_family() >= 9` (tensor_ops, `qmt_on()`); older Macs need a `qattn_t`-free fallback.
3. **GELU fold** — `spark_gelu_gu` stays an elementwise pass over `T*ffn` (now writing half straight into `hh_`, so
   the separate `rows_to_half` pass is gone). No change to the shipping GEMM's epilogue, so D4 holds.

**The one real gap — ring addressing.** `qattn_t` reads `kb + slot*D` for `k0 = 0..tgl` with no lower bound, and its
causal limit is `p.pos0 + min(tk, T-1)`. A sliding layer's ring holds only the last 512 keys anywhere it may have
wrapped, so the kernel would need (a) the oldest live key and (b) slot = `(oldest + r) & slot_mask`. Full-attention
layers (7 of 28 on the 1.7B, 9 of 36 on the 4B) need nothing. Two ways:

* **R1 — add the two fields to the shared path.** `AttnPrefParams` already ends in `max_seq, scale`, so `uint wrapped,
  slot_mask` (or one packed `ring`) appends without moving anything, exactly as `AttnDecParams` already grew them in
  v0.2.28 for the decode path. Qwen35 passes `{0, 0}` and reads identically — precedent exists in this repo.
  Touches: `attn.metal`, `tensor/qattn_t.metal`, `kparams.h`, every `AttnPref` caller. These are shipping kernels.
* **R2 — a Spark-only `spark_attn_prefill`.** Nothing shared is touched (D4 pure). Cost: a new kernel to write,
  benchmark and validate against fp64, and no benchmark evidence — it would be the only unproven piece.

Estimated prefill after R1/R2: ~0.35-1.4 s for 2048 tokens instead of ~25 s.

## P4 — tests  ✅
- [x] `tools/spark_config_test.cpp`: both configs' shapes/tables, 12 refusal cases, flat-RoPE/untied-head forms,
      layer fallback, plan-vs-allocation, matvec names including group-32 routing
- [x] Runs green locally; verified it fails when a derived value is deliberately broken
- [x] In `CMakeLists.txt` (`add_test` + `SPARK_CONFIG_DIR`) and `.github/workflows/ci.yml`
- [ ] Consider a `spark_load_test` that checks tensor-name mapping against synthetic metadata (CPU only). Nice-to-have.

## P5 — docs  [~]
- [x] `docs/spark/SPARK.md`, `TODO.md`, `configs/NOTES.md`, reference sources
- [ ] README / `docs/usage.md` mention — only if this goes upstream; not needed for the fork
- [ ] `src/hub/catalog.cpp` entry — upstream concern only

## P6 — delivery  ✅
- [x] Prune workflows: `release.yml` + `community-bench.yml` deleted; `ci.yml` trimmed to lint, pr-size-script,
      docs-links, build-and-test, metal-compile; `pr-size.yml` kept
- [x] `lint` green on CI (the hook set runs there; `prek` is not installable on this box)
- [x] Branch `spark-x2.6` pushed to `Wiselfx/kapitsa`
- [x] PR #1 with **"Not verified on a GPU"** in the title and body
- [x] **CI green**: `lint`, `docs-links`, `pr-size-script`, `metal-compile`, `build-and-test` all success;
      `spark_config_test: ok` on a macOS runner
- [x] `pr-size` red by design and explained in the PR body (no reviewer on a fork to grant the override)
- [ ] **Your Mac pass**: build, load a checkpoint, generate, compare against `transformers`

## Performance: what was taken from upstream, and what was not

Reviewed against upstream's dense decode path (`Qwen35::enc_attn` / `enc_mlp` / `enc_qmv_multi`) before the first Mac
run, rather than after, so the baseline is measured on the final code.

**Done:**
- [x] **Split-K over the measured core count.** `cores` was a literal 32 where upstream uses `gpu_cores()`
      (`qwen35.cpp`). 1.6x too many chunks on a 20-core M4 Pro, too few on a 40-core M4 Max. The attention split
      buffers (`part_o_`, `part_ml_`) were sized from that same literal, so they now use the same count — sizing them
      from 32 while chunking over 40 cores would have written past the end.
- [x] **q | k | v | gate in one dispatch.** They are three row ranges of one fused matrix plus `g_proj`, all sharing
      one normalized input. `Spark::multi_proj` uses `qmvg_multi_<fmt>` for one format or `qmv2/qmv3_multi_r2` for
      4/8-bit group 64, as `Qwen35::enc_qmv_multi` does. Segment views are row-offset views (0, qd, qd+kd), which is
      what the earlier q/k/v multi attempt got wrong: it bound the whole tensor three times, so k and v were q.
- [x] **Gate+up in one dispatch.** Same helper, no new kernel and no change to any shipping GEMM: `qmvg_multi` takes
      the two matrices and writes their raw products; Spark's exact-GELU epilogue still runs after it. Falls back to
      two matvecs only when their formats differ.
- Net: three projection dispatches per decoder layer (qkv+gate, gate+up, down) instead of five. Each segment still
      normalizes its own x in its threadgroups, so the norm work is unchanged; the attention dispatches are unchanged.
- [x] **KV regrow copies into the new row stride.** `reserve` passed `(old, old, target)` to `kv_regrow`, which takes
      `(source stride, destination stride, Hkv)`; after the first growth every KV head after the first landed at the
      old offset and attention read zeros. A fresh long prompt did not expose it (all rows are rewritten), but a
      second turn crossing the initial 1024-token allocation did. Fixed and pinned by `check_kv_regrow`.

**Not done, deliberately — with the reason:**
- [ ] **q8_0 KV cache** (docs/spark/SPARK.md §19). Cuts KV bytes to 53% of half and halves the long-context memory;
      at bf16 weights the decode gain is only ~7% at 42k because the weights dominate, rising to ~19-25% once the 4-bit
      pack is used. Needs a 272-byte row (256 int8 + 8 fp16 scales, 16-aligned), new `kv_regrow` stride, a bump to
      `disk_format`, and q8 variants of the three attention consumers. Correctness first: the regrow fix above is a
      prerequisite for trusting any multi-turn cache measurement.
- [ ] **Fused GELU prefill GEMM.** Upstream's `qrf_gu` fuses gate+up and silu into one kernel. Using it for Spark needs
      an erf epilogue in the shared GEMM (the same D4 trade-off), and the two GEMMs' FLOPs do not change; this pass
      only deletes the intermediate fp32 product and its `rows_to_half`. Left for a measured need.
- [x] **Batched prefill** (`encode_prefill`; see P3 above).
- [x] **A ring-aware attention read** (decode, both prefill kernels; ring = window + chunk).
- [x] **4-bit tensor-ops prefill on M5** (`qmt_m64_n64`, group 32/64) and `qattn_t_g4d2s8` selection.
- [ ] **Measure on a Mac** — bf16 vs the INT4/g32 conversion, `--prefill-chunk 2048` vs `4096`, and logits against
      the reference. No speed number is claimed before this.

**Withdrawn claim:** I earlier reported that `attn_decode_partial` could produce NaN from `num/den` on a chunk with no
valid keys. It cannot: `nchunks` is derived from `L`, so every chunk has `start < p.L` and therefore `n >= 1`, so `m`
and `l` are always written. The padding does mean a chunk's `n` is rounded up to the 16-position tile — waste once the
chunk is smaller than the tile, which the core-count fix reduces. No NaN.

### The Mac commands for the verification pass

Build the branch (needs the Command Line Tools and CMake 3.24+; an older prebuilt release does not carry these
changes):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)" --target kapitsa logit_dump
ctest --test-dir build -R 'spark|ring' --output-on-failure
```

Fetch a pack. `mlx-community/Spark-X2.6-4B-OptiQ-4bit` is 3.04 GB, per-tensor mixed 4/8-bit group 64 (the OptiQ
path). The reference converter's default INT4/g32 + INT8/g32 output is the other path and exercises the new group-32
kernels; run both if you can:

```bash
huggingface-cli download mlx-community/Spark-X2.6-4B-OptiQ-4bit --local-dir ~/models/spark-4b-optiq
# reference profile (needs spark-mlx-convert on macOS >= 26.2):
#   python -m spark_mlx_llm.convert XHToken/Spark-X2.6-4B ~/models/spark-4b-g32
```

Does it load and plan? Does it speak?

```bash
build/kapitsa doctor --models-dir ~/models
build/kapitsa serve --model ~/models/spark-4b-optiq --prefill-chunk 4096
#    in another terminal:
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "spark", "max_tokens": 32, "messages": [{"role": "user", "content": "What is the capital of Anhui Province?"}]}'
#    expect something coherent. The server's terminal prints the detected model, the ring size and the plan.
#    A 2K+ token prompt now exercises the wrapped ring and the batched prefill; it must not stop at 512.
```

Measure, then compare logits:

```bash
build/kapitsa bench --model ~/models/spark-4b-optiq --prefill-chunk 2048
build/kapitsa bench --model ~/models/spark-4b-optiq --prefill-chunk 4096
# and the same for spark-4b-g32, if converted
```

**Then, if step 4 works**, the reference comparison (this is what turns "it speaks" into "it computes the right thing"):

```bash
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 \
  build/logit_dump --model ~/models/spark-4b --prompt-file bench/prompts/prose.txt --out /tmp/spark-dump
#    and transformers' own logits for the same prompt (transformers==4.57.1, per the model card) via
#    tools/logit_compare.py -- see CONTRIBUTING.md "Output check". Nothing there is claimed until it runs.
```

## Next

- [ ] Mac pass: build, load bf16 and the INT4/g32 conversion, run 2K+ prompts, compare logits with the reference,
      and run `kapitsa bench` at chunk 2048 and 4096. No speed number is claimed before this.

The earlier open questions (branch layout, docs in the branch, and whether to do the ring read + batched prefill now)
are settled: both are implemented and the docs stay with the code.
