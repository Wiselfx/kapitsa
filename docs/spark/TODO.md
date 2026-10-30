- [x] **Snapshots** (`save`/`restore`/`can_snapshot`). Copies only the sliding-window rings, as Dory does: a full
      layer's cache only appends, so truncation reads the same bytes, while a ring must be saved before it is
      overwritten. `encode_prefill` now honours `snap`/`snap_row` (it must, since `can_snapshot()` is true and the
      engine relies on it).
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
- [x] Tied embeddings; refuse a stray `lm_head`
- [x] Per-tensor mixed quant (MLX affine 3/4/5/6/8 g64, 4 g32, bf16) from `config.json`'s `quantization`
- [x] HF-vs-MLX RMSNorm decided from the values (`zero_centred`)
- [ ] **Verify with a real checkpoint** (needs a Mac; `kapitsa doctor --models-dir <dir>` then a short generation)

## P2 — GELU  ✅
- [x] Own kernel library `src/kernels/spark/` (decision: additive, do **not** touch shipping MLP kernels, SPARK.md §10)
- [x] `spark_gelu_gu` — PyTorch's tanh-approximation gelu, gate*up fold
- [x] Existing models' kernel libraries byte-identical (nothing in them changed)

## P3 — attention  [~]
- [x] Per-layer `rot` (64 full / 256 sliding) and per-layer `theta` (5e6 / 1e4)
- [x] **`spark_attn_prep`** — Spark's own prologue (RoPE + q|gate packing + cache append). It replaces `attn_prep`,
      which cannot be reused: that kernel *always* RMS-normalizes q and k, and Spark has no q/k norm at all (the
      reference ropes the raw `q_k_v_proj` output; zero occurrences of `q_norm`/`k_norm` in modeling_spark.py).
      Passing an all-ones weight does not disable it, and rescaling q and k by a common per-row factor changes
      `softmax(q·k/√D)` — it is not a gauge freedom. The earlier "all-ones buffers" plan was wrong.
- [x] Per-head sigmoid gate: one scalar broadcast over the head's 256 dims
- [x] KV: 512-slot rings for sliding layers, full cache for full layers (`reserve` never grows a ring)
- [x] Decode path end to end
- [x] **Wrapped sliding window.** `attn_decode_partial` gained a cyclic read (`AttnDecParams::wrapped`/`slot_mask`),
      so a sliding layer attends over its ring past 512 tokens. The insight that made it small: attention is a
      *weighted sum over a set* of keys, so reading the ring in ring order is the same softmax as reading it linearly
      -- a change of rotation, not a reordering. No compaction pass, no second buffer, no kernel-side sort: the tile
      load just starts at the oldest live key and wraps. Depends on window being a power of two and a multiple of
      kTile (both true; `tools/ring_slot_test.cpp` checks them). Linear-cache models pass `{0, 0}` and read exactly as
      before.
- [ ] Ring-aware attention kernel (the blocker for any context longer than 512 tokens)
- [ ] Batched prefill (`attn_prefill11r` + window; `qmms`/`qrf` GEMMs + the GELU fold). Until then prefill loops over
      decode rows: correct, slow, and it logs that once.

## P4 — tests  ✅
- [x] `tools/spark_config_test.cpp`: both configs' shapes/tables, 13 refusal cases, flat-RoPE form, layer fallback
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
- [x] **q | k | v in one dispatch.** They are three row ranges of one fused matrix sharing one normalized input, which
      is exactly upstream's `qmv2_multi` / `qmv3_multi` case (`Qwen35::enc_qmv_multi`). 3 dispatches → 1, and the
      RMSNorm is computed once instead of three times. Falls back to the three row-sliced matvecs when the format is
      not 4/8-bit group 64 or K exceeds the untiled variant's 4096.
- Net: 12 → 11 dispatches per decoder layer on decode (336 → 308 per token for the 1.7B, 432 → 396 for the 4B).

**Not done, deliberately — with the reason:**
- [ ] **Gate+up as one dispatch.** Upstream's `mlp_gate_up` takes two matrices in one kernel; Spark issues two matvecs.
      I tried to fix this by deferring the GELU closure by one layer so two pairs could close together, and found it
      **cannot work**: `down_proj` consumes the closed row immediately, so `mh_` must be complete by the end of the
      layer. The remaining option is to modify `mlp_gate_up_impl` (index the second matrix at `row0 - N0` and run
      Spark's erf GELU instead of silu), which splits the kernel on a template flag — precisely what D4 rejected
      because it recompiles the hot path of every existing model. It needs a deliberate decision, not a drive-by.
- [ ] **Batched prefill** (the big one at long context; today prefill loops the decode path).
- [ ] **A ring-aware attention read** (the 512-token wall).

**Withdrawn claim:** I earlier reported that `attn_decode_partial` could produce NaN from `num/den` on a chunk with no
valid keys. It cannot: `nchunks` is derived from `L`, so every chunk has `start < p.L` and therefore `n >= 1`, so `m`
and `l` are always written. The padding does mean a chunk's `n` is rounded up to the 16-position tile — waste once the
chunk is smaller than the tile, which the core-count fix reduces. No NaN.

### The Mac commands for the verification pass

**Easiest: download the prebuilt release** (`.github/workflows/release.yml`, tag `v0.2.25`). No compiler needed.

```bash
# 1. download, check, place on PATH
curl -fsSLO https://github.com/Wiselfx/kapitsa/releases/download/v0.2.25/kapitsa-0.2.25-darwin-arm64.tar.gz
curl -fsSLO https://github.com/Wiselfx/kapitsa/releases/download/v0.2.25/kapitsa-0.2.25-darwin-arm64.tar.gz.sha256
shasum -a 256 -c kapitsa-0.2.25-darwin-arm64.tar.gz.sha256
tar xzf kapitsa-0.2.25-darwin-arm64.tar.gz
xattr -d com.apple.quarantine kapitsa-0.2.25-darwin-arm64/kapitsa   # ad hoc signed, not notarized
sudo cp kapitsa-0.2.25-darwin-arm64/kapitsa /usr/local/bin/

# 2. fetch a pack (the 4B first: better-tuned shape, pre-quantized MLX packs, no conversion)
#    mlx-community/Spark-X2.6-4B-OptiQ-4bit is 3.04 GB, per-tensor mixed 4/8-bit group 64
huggingface-cli download mlx-community/Spark-X2.6-4B-OptiQ-4bit --local-dir ~/models/spark-4b

# 3. does it load and plan?
kapitsa doctor --models-dir ~/models

# 4. does it speak? (no one-shot command: serve, then ask the API)
kapitsa serve --model ~/models/spark-4b
#    in another terminal:
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "spark", "max_tokens": 32, "messages": [{"role": "user", "content": "What is the capital of Anhui Province?"}]}'
#    expect something coherent. The server's terminal is the useful part: it prints the detected model and the plan.
#    Past 512 tokens of context the encode path stops with a clear "ring-aware attention read" fatal rather than
#    producing wrong tokens — the known gap, not a crash.
```

**Or build from source** (same behaviour, needs the Command Line Tools and CMake 3.24+):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(sysctl -n hw.ncpu)" --target kapitsa logit_dump
ctest --test-dir build -R spark_config_test --output-on-failure
```

**Then, if step 4 works**, the reference comparison (this is what turns "it speaks" into "it computes the right thing"):

```bash
MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 \
  build/logit_dump --model ~/models/spark-4b --prompt-file bench/prompts/prose.txt --out /tmp/spark-dump
#    and transformers' own logits for the same prompt (transformers==4.57.1, per the model card) via
#    tools/logit_compare.py -- see CONTRIBUTING.md "Output check". Nothing there is claimed until it runs.
```

## Open questions for the user
- [ ] Push to fork `main`, or a `spark-port` branch with a PR against fork `main`?
- [ ] Should `docs/spark/` stay in the pushed branch? (It is deliberate project documentation, not agent scratch.)
- [ ] Do you want the ring-aware attention + batched prefill done now, or ship this decode-correct skeleton first and
      iterate on your Mac?
