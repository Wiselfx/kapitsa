# Spark-X2.6 architecture notes (derived — never re-derive these by hand)

Everything here is **computed** from `docs/spark/configs/*.config.json` and read off
`docs/spark/reference/spark_modelling.py`. The JSON files are the facts; this file is the convenience layer.

## Derived dimensions

| quantity | 1.7B | 4B | how |
|---|---|---|---|
| hidden `H` | 2048 | 2560 | `hidden_size` |
| ffn `I` | 6656 | 10240 | `intermediate_size` |
| layers `L` | 28 | 36 | `num_hidden_layers` |
| `Hq` / `Hkv` | 8 / 2 | 16 / 4 | G = `Hq/Hkv` **= 4 for both** |
| head_dim `D` | 256 | 256 | `head_dim` |
| q rows `QD` | 2048 | 4096 | `Hq*D` |
| kv rows `KD` | 512 | 1024 | `Hkv*D` |
| fused qkv out | 3072 | 6144 | `QD + 2*KD` |
| gate rows | 8 | 16 | `Hq` (`g_proj` is `[Hq, H]`) |
| vocab | 131072 | 131072 | tied to `model.embedding.weight` |
| full-attn layers | 7 | 9 | indices below |
| sliding layers | 21 | 27 | window 512 |
| rot dims, full layers | 64 | 64 | `0.25 * 256` |
| rot dims, sliding layers | 256 | 256 | `1.0 * 256` |
| θ full / sliding | 5e6 / 1e4 | 5e6 / 1e4 | `rope_parameters` |

## Layer tables

**1.7B — 28 layers**, full attention at **3, 7, 11, 15, 19, 23, 27** (every 4th, 0-indexed); the other 21 slide.

**4B — 36 layers**, full attention at **3, 7, 11, 15, 19, 23, 27, 31, 35**; the other 27 slide.

Rule for both: `layer_types[i] == "full_attention"` iff `(i + 1) % 4 == 0`. Do **not** hardcode the lists; read
`layer_types` (Spark's own config always ships it, and the reference `configuration_spark.py` falls back to the same
modulo rule when absent — mirror that fallback).

## KV cache cost (it is tiny — take advantage of it)

- Full layers only: 1.7B **14,336 B/position**, 4B **36,864 B/position**. At 1M positions that is 14 GB / 36 GB, so a
  long context still needs the full layers capped, but it is far cheaper than an all-full model.
- Sliding layers need only a **512-slot ring each**: 1.7B **22.0 MB** total, 4B **56.6 MB** total — fixed, independent
  of context length. A naive implementation that gives sliding layers a `cap`-sized cache throws away the whole
  advantage of the architecture.

## Forward pass, exactly as the reference does it

Per layer:

```
residual = hidden
h = input_layernorm(hidden)                       # RMSNorm, eps 1e-6
h = h.to(weight.dtype)                           # bf16

qkv  = q_k_v_proj(h)                             # [T, QD + 2*KD], ONE matmul
q    = qkv[..., :QD]        -> [T, Hq,  D]
k    = qkv[..., QD:QD+KD]   -> [T, Hkv, D]
v    = qkv[..., QD+KD:]     -> [T, Hkv, D]
g    = g_proj(h)            -> [T, Hq, 1]        # NOTE: no activation yet

q = rope(q, cos, sin)                            # rope dims depend on layer type
k = rope(k, cos, sin)
# *** NO q_norm / k_norm ***   (Spark has no q_norm.weight or k_norm.weight at all)

attn = softmax(q @ k^T / sqrt(D) + mask) @ v     # causal, plus sliding window on sliding layers
attn = attn * sigmoid(g.float()).to(attn.dtype)  # PER-HEAD scalar, broadcast over all 256 dims
attn = out_proj(attn)
hidden = residual + attn

residual = hidden
h = post_attention_layernorm(hidden)
h = mlp(h)                                       # gelu(gate_proj(h)) * up_proj(h) -> down_proj
hidden = residual + h
```

Then `model.norm(hidden)` → tied `lm_head` = `model.embedding.weight`.

### The four things that differ from Qwen (and from Kapitsa's existing paths)

1. **No q/k RMSNorm.** Kapitsa's `attn_prep` takes `qn_w`/`kn_w` buffers. Spark has no such tensors.
   Plan (D3): pass an all-ones vector — costs 256 floats, keeps the kernel untouched, zero numerics risk.
2. **Per-head sigmoid gate, not per-channel.** `g_proj: [H, Hq]`. The scalar is broadcast across all `D=256` dims of
   that head. Plan (D4.2): host writes the same scalar into all 256 gate slots so Kapitsa's existing per-channel
   epilogue consumes it unchanged.
3. **Activation is `gelu` (tanh approx), not SiLU.** `ACT2FN["gelu"]` in transformers = `gelu_pytorch_tanh`
   (`0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))`). This is the one genuine new kernel requirement.
4. **Per-layer RoPE.** 64 rotary dims + θ=5e6 on full layers; 256 rotary dims + θ=1e4 on sliding layers.
   `AttnPrepParams.rot` and `.theta` are already per-dispatch, so this is host-side bookkeeping only.

### Confirmed easy / no work

- **Tied embeddings**: `lm_head` is the embedding table. `Qwen35.cpp` already handles this
  (`lm_head_ = ws_.has("lm_head.weight") ? qlin("lm_head") : embed_`).
- **No biases anywhere** (`attention_bias: false`, `mlp_bias: false`).
- **Fused `q_k_v_proj`** maps directly onto Kapitsa's multi-matrix matvec dispatch (`enc_qmv_multi`, up to 4 segments),
  which `Qwen35` already uses as `{{&L.q,&L.k,&L.v}}` — we slice one `QLinear` into three row-ranges.
- **`g_proj`** is a plain bf16 vector of `Hq` elements; it is not quantized in the trunk packs.
- **Tokenizer**: BPE with `vocab.json` + `merges.txt` + `tokenizer.json` (standard); Kapitsa's tokenizer already reads
  `model.merges` (verified at `src/tokenizer/tokenizer.cpp:491`).

## Reference file pointers (not vendored — read them on the Hub)

Vendoring HF's `modeling_spark.py` / `configuration_spark.py` here would mean carrying ~600 lines of someone else's
code that the repo's own lint and PR-size checks would then have an opinion about. They are cited instead:

- `https://huggingface.co/XHToken/Spark-X2.6-4B/blob/main/modeling_spark.py`
  - `Spark2_5Attention.__init__` (l.135): the `q_dim` / `kv_dim` / `gate_score` layout.
  - `Spark2_5Attention.forward` (l.158): the exact order in the forward-pass block above.
  - `Spark2_5MLP` (l.116): enforces `hidden_act == 'gelu'`.
- `https://huggingface.co/XHToken/Spark-X2.6-4B/blob/main/configuration_spark.py`
  - `get_partial_rotary_factor(layer_type)` (l.113): the per-layer RoPE accessor.
  - `layer_types` fallback (l.93): "full_attention" every `full_attention_interval` layers.

Note the reference uses `eager_attention_forward`, i.e. an ordinary dense causal path: there is no exotic attention
mechanism to replicate.
