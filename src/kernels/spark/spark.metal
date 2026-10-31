#include "common.h"

// ---- Spark-X2.6 (src/model/spark.cpp) ----
//
// Spark's MLP is gelu(gate) * up. The activation is NOT the tanh approximation: the reference sets
// `self.act_fn = ACT2FN[config.hidden_act]` with hidden_act == "gelu", and in transformers 4.57.1
// (src/transformers/activations.py) the mapping is
//     ACT2CLS["gelu"] = GELUActivation,   whose forward is nn.functional.gelu(input)
// i.e. with no `approximate` argument, and torch.nn.functional.gelu's default is approximate='none':
//     gelu(x) = x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
// sources: docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu (signature
// `gelu(input, approximate='none')`); huggingface/transformers v4.57.1 activations.py l.66-85 and l.314-321. Note that
// ACT2CLS has a *separate* key "gelu_pytorch_tanh" -> GELUTanh; Spark's config does not ask for it. Using the tanh form
// here is the classic mistake (it is 1.5e-4 off per element, worst near |x| ~ 1, and it compounds over 28-36 layers).
// Everywhere else Kapitsa is SiLU (silu in common.h). Rather than add an activation flag to the fused gate/up store
// sites of the shipping GEMM kernels — which would recompile the hot path of every existing model and put their
// numerics at risk for a new family's sake — Spark runs its own tiny epilogue kernels here. The GEMMs themselves are
// the existing generic ones (qmv2/qmv3/qmvg for rows), and these kernels do the activation.
//
// This keeps the change additive: no existing kernel is touched, so no existing model's outputs can move.

// erf to 1.5e-7 (Abramowitz & Stegun 7.1.26). Metal has no erf — the same reason vit.metal carries vit_erf (its
// comment says so) — so this is the same rational approximation, kept local to this library rather than shared, so
// that Spark's library does not depend on the base library's contents changing.
NI_INLINE float spark_erf(float x) {
  const float z = metal::abs(x), t = 1.0f / (1.0f + 0.3275911f * z);
  const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  const float r = 1.0f - poly * metal::precise::exp(-z * z);
  return x >= 0.0f ? r : -r;
}

// Exact (erf) GELU, fp32: x * Phi(x) = x * 0.5 * (1 + erf(x / sqrt(2))). The reference's inputs are bf16-rounded and
// nn.functional.gelu runs in the activation's dtype, then the gate*up product goes to the output dtype.
NI_INLINE float spark_gelu(float x) { return 0.5f * x * (1.0f + spark_erf(x * 0.7071067811865476f)); }

// gate|up -> activation. Both inputs are the raw GEMM results of the MLP's gate_proj and up_proj, fp32, laid out
// with `ld` floats per row and `rows` rows of `n` columns.
//   out[r, c] = spark_gelu(gate[r, c]) * up[r, c]
// The decode matvecs all take `const device float* x` (qmv2/qmv3 and qmvg alike), so the decode epilogue writes fp32.
// The prefill down_proj is a qrf GEMM and takes half; spark_gelu_gu_half writes that form directly, which is the same
// rounding as the rows_to_half pass it replaces but one dispatch and one read/write shorter.
// grid: ceil(n / 256) x rows; threadgroup 256.
struct SparkGuParams {
  uint32_t n;   // columns (the FFN's intermediate size)
  uint32_t ld;  // floats between columns of one row (== n for a dense product)
};

[[kernel]] void spark_gelu_gu(const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                              device float* out [[buffer(2)]], constant SparkGuParams& p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
  const uint32_t c = tg.x * 256u + tid.x;
  if (c >= p.n) return;
  const size_t i = size_t(tg.y) * p.ld + c;
  out[i] = spark_gelu(gate[i]) * up[i];
}

[[kernel]] void spark_gelu_gu_half(const device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                                   device half* out [[buffer(2)]], constant SparkGuParams& p [[buffer(3)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint2 tid [[thread_position_in_threadgroup]]) {
  const uint32_t c = tg.x * 256u + tid.x;
  if (c >= p.n) return;
  const size_t i = size_t(tg.y) * p.ld + c;
  out[i] = half(spark_gelu(gate[i]) * up[i]);
}

// ---- assembling the gate layout attn_prep and its epilogue expect ----
//
// Kapitsa's attention path assumes the Qwen3.5 layout, where `q_proj` emits BOTH the query and the output gate, so the
// buffer attn_prep reads (and the epilogue later takes its gate from) is [row][Hq][2][D]: the head's query in the
// first D, the head's gate in the second D. attn_prep reads the query half, applies q/k RMSNorm and RoPE, and writes
// a DENSE [row][Hq][D] query to q_out; attn_decode_partial then reads that dense q, and attn_decode_reduce reads the
// gate back from the interleaved buffer at +D.
//
// Spark has no gate rows: q_proj is the fused q_k_v_proj (Q|K|V only) and the gate is a separate matrix g_proj: [Hq, H]
// that emits ONE scalar per head, broadcast over that head's whole 256-wide dim. This kernel assembles the layout
// both stages expect: the dense q rows into the interleaved buffer's +0 halves, and the broadcast per-head gate into
// its +D halves. The epilogue's own sigmoid is then the only one applied, so the gate is written raw here.
// ---- the attention prologue: RoPE, the q|gate packing, and the cache append ----
//
// This replaces attn_prep for Spark, and it had to: attn_prep *always* applies an RMSNorm to q and k
// (attn.metal: `x * rsqrt(ss/D + eps) * qn_w[tid]`), which is why it exists at all — it is the Qwen3.5 prologue, whose
// checkpoints carry q_norm.weight and k_norm.weight. Spark has neither; the reference applies RoPE to the raw
// q_k_v_proj output (modeling_spark.py: `q = q.view(...); q = apply_rotary_pos_emb(q, cos, sin)`, with no norm step
// anywhere and zero occurrences of q_norm/k_norm in the file). Passing an all-ones vector as the norm weight does NOT
// disable it: the normalization itself is the problem, and rescaling q and k by a common per-row factor changes
// softmax(q.k/sqrt(D)) — it is not a gauge freedom.
//
// One threadgroup per (head, token), D threads. The q and k heads rotate with the layer's rot dims and theta; the
// value heads are a plain cast into the cache.
//
// grid: x = Hq + 2*Hkv (q heads, k heads, v heads), y = T. threadgroup: D threads.
struct SparkPrepParams {
  uint32_t T;      // rows (1 on the decode path)
  uint32_t Hq;     // query heads
  uint32_t Hkv;    // kv heads
  uint32_t D;      // head dim (256)
  uint32_t rot;    // rotary dims for this layer (64 full / 256 sliding)
  uint32_t pos;    // position of the first row
  uint32_t max_seq;  // slots per kv head in the cache
  uint32_t g_ld;   // floats between rows of the per-head gate scalars
  float theta;     // RoPE theta for this layer (5e6 full / 1e4 sliding)
  uint32_t slot_mask;  // 0 for a full-attention layer, else window - 1: the ring slot of position p is p & slot_mask
  // Row strides/offsets of q_in, k_in and v_in, in floats. Decode: the three slices of stg_ (stride
  // qd/kd/kd, offsets 0/qd/qd+kd). Prefill: the fused qrf output [T][qkvd], all three buffers the
  // same base with offsets 0/qd/qd+kd and row stride qkvd. qrf writes row-major, not q-only then k.
  uint32_t q_ld, k_ld, v_ld, q_off, k_off, v_off;
};

[[kernel]] void spark_attn_prep(const device float* q_in [[buffer(0)]], const device float* k_in [[buffer(1)]],
                               const device float* v_in [[buffer(2)]], const device float* g_in [[buffer(3)]],
                               device float* q_out [[buffer(4)]], device float* qg_out [[buffer(5)]],
                               device half* kc [[buffer(6)]], device half* vc [[buffer(7)]],
                               constant SparkPrepParams& p [[buffer(8)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tid2 [[thread_position_in_threadgroup]]) {
  const uint tid = tid2.x;
  const uint h = tg.x, t = tg.y;
  if (tid >= p.D) return;
  const uint pos = p.pos + t;

  // The slot a position occupies. A full-attention layer appends (slot == position, its cache grows with the
  // conversation); a sliding layer is a ring of `window` slots, so position p lives in p % window == p & (window - 1).
  // Getting this wrong is not a slow path: writing absolute positions into a 520-slot ring runs off the end of the
  // buffer at position 520 and corrupts whatever follows it.
  const uint slot = p.slot_mask ? (pos & p.slot_mask) : pos;

  // value heads: plain copy into the cache
  if (h >= p.Hq + p.Hkv) {
    const uint hv = h - p.Hq - p.Hkv;
    vc[(hv * p.max_seq + slot) * p.D + tid] = half(v_in[p.v_off + t * p.v_ld + hv * p.D + tid]);
    return;
  }

  const bool is_q = h < p.Hq;
  const uint hv = h - p.Hq;
  const float x = is_q ? q_in[p.q_off + t * p.q_ld + h * p.D + tid]
                       : k_in[p.k_off + t * p.k_ld + hv * p.D + tid];

  // RoPE over the first `rot` dims, GPT-NeoX (half-split) convention: pair tid with tid +- rot/2, as the reference's
  // rotate_half does (cos = cat(f, f), x*cos + rotate_half(x)*sin). Dims >= rot pass through untouched.
  float y = x;
  if (tid < p.rot) {
    const uint half_rot = p.rot / 2;
    const uint i = tid % half_rot;
    // inv_freq = theta^(-2i/rot), i in [0, rot/2) -- the reference's compute_rope_cos_sin
    const float inv_freq = metal::precise::pow(p.theta, -2.0f * float(i) / float(p.rot));
    const float ang = float(pos) * inv_freq;
    const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
    // the partner lane's *unrotated* value. Each threadgroup is a whole head, so the head is in registers/registers-per
    // thread only; re-read the partner from the source instead of a barrier, which is cheap for one row and keeps this
    // kernel stateless for the batched prefill case later.
    const uint partner = tid < half_rot ? tid + half_rot : tid - half_rot;
    const float xp = is_q ? q_in[p.q_off + t * p.q_ld + h * p.D + partner]
                          : k_in[p.k_off + t * p.k_ld + hv * p.D + partner];
    y = tid < half_rot ? x * cs - xp * sn : x * cs + xp * sn;
  }

  if (is_q) {
    q_out[(t * p.Hq + h) * p.D + tid] = y;
    // the interleaved [Hq][2][D] buffer attn_decode_reduce takes the gate from: the query at +0, and the per-head
    // scalar broadcast across the channels at +D. The reference applies sigmoid(g) per head after attention
    // (gate_score is built as [b, s, Hq, 1]), and the reduce does sigmoid() itself, so the raw scalar goes here.
    const size_t o = (size_t(t) * p.Hq + h) * 2 * p.D;
    qg_out[o + tid] = y;
    qg_out[o + p.D + tid] = g_in[size_t(t) * p.g_ld + h];
  } else {
    kc[(hv * p.max_seq + slot) * p.D + tid] = half(y);
  }
}

// A cache's ring copied: the checkpoints of the sliding-window layers. Spark's own copy of dory_copy4 rather than a
// dependency on the base library's (D4: this library is purely additive, and adding a model must not force a recompile
// of kernels the shipping models run). grid: ceil(n / 256) x 256
[[kernel]] void spark_copy4(const device uint4* src [[buffer(0)]], device uint4* dst [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < n) dst[i] = src[i];
}
