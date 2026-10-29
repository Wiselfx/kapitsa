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
- [ ] **Verify with a real checkpoint** (needs a Mac; `kapitsa doctor <dir>` then a short generation)

## P2 — GELU  ✅
- [x] Own kernel library `src/kernels/spark/` (decision: additive, do **not** touch shipping MLP kernels, SPARK.md §10)
- [x] `spark_gelu_gu` — PyTorch's tanh-approximation gelu, gate*up fold
- [x] Existing models' kernel libraries byte-identical (nothing in them changed)

## P3 — attention  [~]
- [x] Per-layer `rot` (64 full / 256 sliding) and per-layer `theta` (5e6 / 1e4)
- [x] All-ones q/k norm buffers (no q/k RMSNorm in this family)
- [x] `spark_pack_qg` — builds the interleaved `[Hq][2][D]` q|gate layout `attn_prep` and its epilogue expect
- [x] Per-head sigmoid gate: one scalar broadcast over the head's 256 dims
- [x] KV: 512-slot rings for sliding layers, full cache for full layers (`reserve` never grows a ring)
- [x] Decode path end to end
- [!] **Wrapped sliding window**: `enc_attn` runs the plain key range while `pos < window`, then stops with a clear
      fatal. Needs a ring-aware attention kernel (offset the cache to the window's first slot, handle wrap).
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

## P6 — delivery  [ ]
- [ ] Prune workflows: delete `release.yml` and `community-bench.yml`, keep `ci.yml` + `pr-size.yml`
- [ ] `prek run --all-files` (lint) — cannot run here without the tool; CI `lint` job covers it
- [ ] Branch + push to `Wiselfx/kapitsa`
- [ ] PR with **"Not verified on a GPU"** in the title and body + the exact Mac commands (SPARK.md §6)

## Open questions for the user
- [ ] Push to fork `main`, or a `spark-port` branch with a PR against fork `main`?
- [ ] Should `docs/spark/` stay in the pushed branch? (It is deliberate project documentation, not agent scratch.)
- [ ] Do you want the ring-aware attention + batched prefill done now, or ship this decode-correct skeleton first and
      iterate on your Mac?
