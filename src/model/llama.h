// Dense Llama (LlamaForCausalLM, model_type "llama"): pre-norm decoder layers of grouped-query attention with RoPE (no
// q / k norm, no bias, no sinks, no sliding window) and a SwiGLU MLP. Written for Nanbeige 4.3 33B (52 layers, hidden
// 7168, 56 query and 8 KV heads of 128, intermediate 22016, RoPE theta 1.2e8, untied embeddings, ChatML).
// Checkpoints: MLX safetensors, affine 4- or 8-bit group-64 projections, a 4-bit embedding, a 4/6/8-bit LM head.
// The attention runs on Flint's kernels (src/kernels/flint/flint.metal: flint_attn_prefill, flint_attn_partial,
// flint_attn_reduce, flint_kv_store; head dim 128) with a sink of -1e30 per head, which drops out of the softmax: a
// change to those kernels has to keep this model's sinkless use of them right as well. Only the RoPE step is Llama's own
// (llama_prep, in src/kernels/flint/llama.metal, which joins Flint's library). Every layer is full attention: one KV cache per layer that grows with the
// conversation, no recurrent state, so a checkpoint is a position.
#pragma once
#include "loader/safetensors.h"
#include "model/model.h"

#include <memory>
#include <string>
#include <vector>

namespace ni {

struct LlamaConfig {
  int hidden = 0, vocab = 0, layers = 0, hq = 0, hkv = 0, hd = 0, ffn = 0;
  double theta = 1e4;
  float eps = 1e-5f;
  std::vector<int> eos;
  int bos = -1;
  int bits = 4, gs = 64;  // the projections' format
};

// An MLX affine linear: codes (uint32 words or bytes), bf16 scale and bias per group of gs inputs
struct LlamaMat {
  const Tensor* w = nullptr;
  const Tensor* s = nullptr;
  const Tensor* b = nullptr;
  uint32_t N = 0, K = 0;
  int bits = 4, gs = 64;
};

struct LlamaLayer {
  const Tensor *in_norm = nullptr, *post_norm = nullptr;
  LlamaMat q, k, v, o, mg, mu, md;
};

class Llama : public Model {
 public:
  static constexpr int kMaxRows = 8;  // rows of a decode pass

  explicit Llama(const std::string& path, int prefill_chunk = 2048);
  ~Llama() override;
  static bool is_model(const std::string& dir);  // config.json's model_type is "llama"
  static LlamaConfig read_config(const std::string& dir);
  static MemPlan plan(const std::string& path, int prefill_chunk = 2048);
  static size_t kv_per_pos(const LlamaConfig& c);  // KV cache bytes per position, every layer

  const LlamaConfig& config() const { return cfg_; }
  int vocab() const override { return cfg_.vocab; }
  const std::vector<int>& eos_ids() const override { return cfg_.eos; }
  int bos_id() const { return cfg_.bos; }

  // ---- Model ----
  void set_mem_budget(size_t bytes) override { mem_budget_ = bytes; }
  std::unique_ptr<Seq> new_seq(int limit, int initial = -1) override;
  bool reserve(Seq& s, int need) override;
  void drop_kv(Seq& s) override;
  size_t kv_bytes_per_pos() const override { return kv_per_pos(cfg_); }
  void reset(Seq& s) override;
  Checkpoint new_checkpoint() override { return Checkpoint{}; }
  void save(Encoder& e, const Seq& s, Checkpoint& c, int pos) override;
  void restore(Encoder& e, Seq& s, const Checkpoint& c) override;
  void encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample = true) override;
  void encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                      Checkpoint* snap = nullptr, int snap_row = 0) override;
  int prefill_chunk() const override { return tmax_; }
  bool can_snapshot() const override { return false; }
  // SSD prefix cache: every layer's cache as rows, no other state
  std::string disk_format() const override;
  void disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const override;

  // ---- tools ----
  float* logits() { return logits_.as<float>(); }
  // teacher-forced scoring: tokens [pos0, pos0 + T) as one pass, every row's next-token logits into out [T][vocab]
  void encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out);

 private:
  void load(int prefill_chunk);
  void alloc_buffers();
  void alloc_prefill();

  void enc_rows(Encoder& e, Seq& s, int pos0, int R);  // R <= kMaxRows rows; final hidden rows in x_
  void enc_layer_rows(Encoder& e, Seq& s, int li, int pos0, int R);
  void enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T);  // GEMM pass; final hidden rows in px_
  void enc_layer_prefill(Encoder& e, Seq& s, int li, int pos0, int T);
  // RoPE on q and k of R rows (fused q | k | v rows, ld qkv_ld) into q (fp32) and K / V (half) at kc / vc:
  // chunk = 1 a chunk buffer [Hkv][R][hd], else the cache [Hkv][rows][hd] at pos0 + r
  void enc_prep(Encoder& e, const Buffer& qkv, uint32_t qkv_ld, const Buffer& q, const Buffer& kc, const Buffer& vc,
                int R, int pos0, uint32_t rows, bool chunk);
  void enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld);
  void enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                  uint32_t step);
  void enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out);
  // up to 4 matrices sharing x (R <= kMaxRows rows, 4/8-bit group 64), an RMSNorm of x folded in; gu: segments 0 / 1
  // are a SwiGLU gate / up (y = silu(gate) up into segment 0's y)
  struct Seg {
    const LlamaMat* M;
    Buffer y;
    bool add = false;
  };
  void enc_mv(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm, uint32_t y_ld = 0,
              bool gu = false);
  // GEMM of M half rows: y [M][y_ld] (+)= xh [M][K] W^T
  void enc_gemm(Encoder& e, const LlamaMat& W, const Buffer& xh, const Buffer& y, int M, bool add, uint32_t y_ld = 0);
  void enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T);

  std::string path_;
  LlamaConfig cfg_;
  WeightStore ws_;
  std::vector<LlamaLayer> layers_;
  LlamaMat embed_, head_;
  const Tensor* final_norm_ = nullptr;
  Buffer inv_freq_;  // fp32 [hd / 2]: RoPE's inverse frequencies
  Buffer sinks_;     // bf16 [hq]: -1e30, Flint's attention kernels' sink input, out of the softmax
  int tmax_ = 2048;
  size_t mem_budget_ = 0;

  // decode activations (kMaxRows rows)
  Buffer x_, qkv_, q_, ao_, mh_, part_o_, part_ml_;
  Buffer logits_, cand_v_, cand_i_, sprob_, samp_;
  // prefill activations (tmax_ rows), allocated on the first prefill of more than kMaxRows rows
  bool prefill_alloc_ = false;
  Buffer px_, pxh_, pqkv_, pq_, pck_, pcv_, pao_, phg_, phu_, phh_, ys_, pscale_;
};

}  // namespace ni
