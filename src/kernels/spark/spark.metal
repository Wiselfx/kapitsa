#include "common.h"

// ---- Spark-X2.6 (src/model/spark.cpp) ----
//
// Spark's MLP is gelu(gate) * up, where gelu is PyTorch's default ("gelu", the tanh approximation):
//     gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
// Everywhere else Kapitsa is SiLU (silu in common.h). Rather than add an activation flag to the fused gate/up store
// sites of the shipping GEMM kernels — which would recompile the hot path of every existing model and put their
// numerics at risk for a new family's sake — Spark runs its own tiny epilogue kernels here. The GEMMs themselves are
// the existing generic ones (qmv2/qmv3/qmvg for rows, qmms/qrf for prefill), and these kernels do the activation.
//
// This keeps the change additive: no existing kernel is touched, so no existing model's outputs can move.

// The tanh approximation of gelu, in fp32.
NI_INLINE float spark_gelu(float x) {
  return 0.5f * x * (1.0f + metal::precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// gate|up -> activation. Both inputs are the raw GEMM results of the MLP's gate_proj and up_proj, fp32, laid out
// with `ld` floats per row and `rows` rows of `n` columns.
//   out[r, c] = spark_gelu(gate[r, c]) * up[r, c]
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
struct SparkGateParams {
  uint32_t Hq;    // query heads
  uint32_t D;     // head dim (256)
  uint32_t T;     // rows
  uint32_t g_ld;  // floats between rows of the per-head scalars
};

// q_in: dense [T][Hq][D]; gate_in: [T][g_ld] scalars; out: interleaved [T][Hq][2][D].
// grid: Hq x T threadgroups, D threads each.
[[kernel]] void spark_pack_qg(const device float* q_in [[buffer(0)]], const device float* gate_in [[buffer(1)]],
                              device float* out [[buffer(2)]], constant SparkGateParams& p [[buffer(3)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
  const uint32_t h = tg.x, t = tg.y, d = tid.x;
  if (h >= p.Hq || d >= p.D || t >= p.T) return;
  const size_t o = (size_t(t) * p.Hq + h) * 2 * p.D;
  out[o + d] = q_in[(size_t(t) * p.Hq + h) * p.D + d];  // the query half attn_prep reads
  out[o + p.D + d] = gate_in[size_t(t) * p.g_ld + h];   // the gate half the epilogue reads (sigmoid applied there)
}
