// Spark-X2.6 (XHToken): the spark2_5 family, Spark-X2.6-1.7B and -4B. See docs/spark/SPARK.md for the plan and
// docs/spark/configs/NOTES.md for the derived shapes.
//
// A dense transformer, simpler than anything else Kapitsa runs: no Gated DeltaNet, no MoE, no Mamba, no sparse
// indexer, no q/k RMSNorm. What it needs that the Qwen path does not do:
//   * per-layer-type RoPE: 64 rotary dims at theta 5e6 on full-attention layers, 256 at theta 1e4 on sliding ones
//     (attn_prep already takes `rot` and `theta` per dispatch, so this is host-side bookkeeping)
//   * a per-head sigmoid output gate (g_proj is [Hq, H], one scalar broadcast over the head's 256 dims; Kapitsa's
//     attention epilogue applies a per-channel gate vector, so src/kernels/spark/spark.metal expands it)
//   * GELU-tanh in the MLP (Kapitsa is SiLU throughout; sparks's own epilogue kernel does the activation)
//   * a sliding window: the mask is the key range, and a sliding layer's KV is a ring of `window` slots
// There is no q_norm / k_norm: attn_prep is handed an all-ones vector, which leaves the kernel path (and therefore its
// arithmetic) untouched.
//
// The first implementation runs the DECODE path in full and runs a prompt through it one row at a time. That is
// correct and slow: the batched prefill kernels (attn_prefill11r with a window parameter, the qmms GEMM tiles with the
// GELU fold) are the next change, and until then `encode_prefill` says so in its log line. Correctness first.
#pragma once
#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "model/model.h"
#include "model/spark_config.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

extern const char* const kSparkKernelSource;

// One dense weight matrix, quantized or not: the subset of Qwen35's QLinear this model needs. Spark's MLX packs carry
// a per-tensor {bits, group_size}, so different matrices in one model can differ (OptiQ: 8-bit attention projections,
// 4-bit MLP).
struct SparkLinear {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;  // scales (quantized only)
  const Tensor* b = nullptr;  // biases (quantized only)
  uint32_t N = 0, K = 0;
  int bits = 16, gs = 64;
  explicit operator bool() const { return w != nullptr; }
  bool dense() const { return bits == 16; }
  bool q4() const { return bits == 4 && gs == 64; }
  bool q8() const { return bits == 8 && gs == 64; }
  bool q() const { return bits != 16; }
  // The suffix the kernel names carry (wfmt.metal's NI_WF_LOW: "f16" for bf16, else "f{bits}g{gs}"), NOT a human
  // description. pipe("qmvg_" + fmt() + "_r2") depends on this being exactly the kernel's spelling.
  std::string fmt() const;
  std::string name() const;  // "4-bit group 64", "bf16", ... for logs and error messages
};

struct SparkLayer {
  bool sliding = true;
  const Tensor *in_norm = nullptr, *post_norm = nullptr;
  SparkLinear qkv;   // fused q | k | v: [Hq*D + 2*Hkv*D, H]
  SparkLinear gate;  // g_proj: [Hq, H]
  SparkLinear o;
  SparkLinear mg, mu, md;
  int rot = 0;          // rotary dims for this layer
  float theta = 1e4f;   // its RoPE base
};

class Spark : public Model {
 public:
  static constexpr int kMaxRows = 8;  // rows a pass may hold (only 1 is used without a drafter)

  Spark(const std::string& path, int prefill_chunk = 2048);
  ~Spark() override;
  static bool is_model(const std::string& dir);
  static SparkConfig read_config(const std::string& dir);
  static MemPlan plan(const std::string& path, int prefill_chunk);

  const SparkConfig& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  std::string formats_summary() const;

  // ---- Model ----
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override;
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override;
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  // True now that save/restore copy the sliding-window rings (the full-attention caches append, so they need no copy).
  // The engine reads this to cut a prefill chunk at a turn boundary inside the pass instead of splitting the chunk.
  bool can_snapshot() const override { return true; }
  std::string disk_format() const override;
  // SSD prefix cache. Spark has both kinds of cache, so it needs both hooks: the full-attention layers are linear (row
  // i is position i) and go as rows; the sliding layers are rings -- a ring holds the last `window` keys wherever they
  // are in the conversation, so a position does not index it -- and go whole, as state. Dory's split is the same, and
  // the base class's both default to doing nothing, which is why leaving them out made the cache claim to work while
  // writing no bytes (see docs/spark/SPARK.md 12f).
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;
  void disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const override;

  // ---- tools ----
  float* logits() { return logits_.as<float>(); }
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);

 private:
  void load(int prefill_chunk);
  void check_shapes() const;
  void alloc_state();
  bool ring_layer(int li) const { return cfg_.kinds[size_t(li)] == SparkLayerKind::Sliding; }
  int cache_rows(const Seq& s, int li) const;

  // one row through one layer (the decode path; prefill runs this per row for now)
  // The matvec kernel for one matrix and its geometry (rows per simdgroup, simdgroups per threadgroup). One rule in
  // one place: the kernel names are a run-time lookup, so a wrong suffix is a fatal "no such pipeline" at load.
  struct Matvec {
    Pipeline p;
    uint32_t rows = 2, nsg = 8;
  };
  Matvec matvec_pipe(const SparkLinear& W, bool norm, bool add);
  // A float tensor as bf16, with `offset` added (the 1 + w conversion for Hugging Face zero-centred RMSNorm weights).
  const Tensor* norm_copy(const Tensor* t, float offset);
  void enc_attn(Encoder& e, Seq& s, int li, int pos);
  void enc_mlp(Encoder& e, const SparkLayer& L);
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out);
  void enc_head(Encoder& e, const Buffer& x, const Buffer& out);
  void enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step);

  std::string path_;
  std::unique_ptr<WeightStore> ws_;
  SparkConfig cfg_;
  std::vector<SparkLayer> layers_;
  // RMSNorm tensors rewritten as 1 + w when the checkpoint stores Hugging Face's zero-centred layout (see hf_norms).
  std::vector<Buffer> owned_bufs_;
  std::vector<Tensor> owned_;
  SparkLinear embed_, head_;
  const Tensor* final_norm_ = nullptr;  // bf16 1.0 x head_dim, handed to attn_prep as the absent q/k norm
  int tmax_ = 2048;
  size_t mem_budget_ = 0;

  // activations, one row
  Buffer x_, xh_, stg_, kr_, vr_, qo_, qr_, qg_, ao_, gate_, gu_, mh_, logits_, part_o_, part_ml_, cand_v_, cand_i_, sprob_;
};

}  // namespace ni
