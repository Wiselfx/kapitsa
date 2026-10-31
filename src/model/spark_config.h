// Spark-X2.6 config.json files (XHToken's spark2_5 family: Spark-X2.6-1.7B and -4B) read from the config alone,
// without the weights or the GPU. Split out of the model class (as qwen4_config is) so that the CPU tests can check
// every shape rule and every derived table with no Metal and no model folder, and so the Hub check can refuse a
// checkpoint before a download.
//
// What this family is (docs/spark/SPARK.md, docs/spark/configs/NOTES.md): a dense transformer, 256-wide heads, GQA
// with 4 query heads per KV head, 3:1 sliding-window / full attention with a per-layer-type RoPE, a per-head sigmoid
// output gate, GELU MLP, tied embeddings, no q/k RMSNorm, no biases. It is NOT a Qwen3.5: no Gated DeltaNet, no
// q_norm/k_norm, no MoE, no sparse indexer.
#pragma once
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ni {

// One layer's attention kind. Spark ships layer_types for every layer; the reference configuration falls back to
// (i + 1) % 4 == 0 for full attention when it is absent, and so does this.
enum class SparkLayerKind { Sliding, Full };

struct SparkConfig {
  std::string model_type;  // "spark2_5"
  int hidden = 0;          // hidden_size (2048 / 2560)
  int ffn = 0;             // intermediate_size (6656 / 10240)
  int layers = 0;          // num_hidden_layers (28 / 36)
  int vocab = 0;           // vocab_size (131072)
  int hq = 0;              // num_attention_heads (8 / 16)
  int hkv = 0;             // num_key_value_heads (2 / 4)
  int head_dim = 0;        // head_dim (256)
  int window = 0;          // sliding_window (512)
  float eps = 1e-6f;       // rms_norm_eps
  // RoPE, per layer kind (config.json's rope_parameters.{full,sliding}_attention).
  int rot_full = 0, rot_swa = 0;        // rotary dims: head_dim * partial_rotary_factor (64 / 256)
  float theta_full = 1e4f, theta_swa = 1e4f;  // rope_theta (5e6 / 1e4)
  bool gate_sigmoid = true;  // gate_attn_act_mode: "sigmoid" (false: "silu")
  bool headwise_gate = true; // headwise_attn_output_gate
  bool tie_embeddings = true;
  bool hf_norms = false;     // RMSNorm weights stored zero-centred (1 + w): set by the loader, not the config
  std::vector<SparkLayerKind> kinds;  // per layer
  std::vector<int> eos;

  int q_dim() const { return hq * head_dim; }        // rows of q within the fused q_k_v_proj
  int kv_dim() const { return hkv * head_dim; }      // rows of k (and of v)
  int qkv_dim() const { return q_dim() + 2 * kv_dim(); }  // q_k_v_proj's N (3072 / 6144)
  int group() const { return hkv > 0 ? hq / hkv : 0; }    // query heads per KV head (4 for both models)
  int n_full() const;
  int n_swa() const;
  // The layers whose attention is full; the other layers are sliding-window rings of `window` slots.
  std::vector<int> full_layers() const;
  std::string formats() const;  // one line for the log: the shapes this model runs at
};

// config.json's model_type is the spark2_5 family
bool spark_family(const nlohmann::json& config);

// Reads and validates. A field this path cannot run is a line in the returned vector; empty means it runs.
// Does not touch the disk or the GPU.
std::vector<std::string> spark_unsupported(const nlohmann::json& config);

// Parses a config already known to be `spark_family`. Throws (fatal-free: callers decide) on a malformed file; the
// shapes it refuses are `spark_unsupported`'s, so check that first.
SparkConfig spark_read_config(const nlohmann::json& config);

// Where the layer tables come from, exposed for the tests and for the loader's log.
std::vector<SparkLayerKind> spark_layer_kinds(const nlohmann::json& config);

// The refusal of a Spark-family model whose spark_unsupported is not empty, for the engine and the Hub check: what it
// is (`what`: a folder or a repo) and each line. "" when it runs.
std::string spark_refusal(const nlohmann::json& config, const std::string& what);

// KV bytes one position of the full-attention layers costs (k and v, half), and the fixed bytes the sliding-window
// rings take. The sliding layers never need a `cap`-sized cache: that is the whole point of the architecture.
size_t spark_kv_per_pos(const SparkConfig& c);
// Slots per sliding-layer ring. A prefill pass writes all T rows of the chunk before attention reads
// them, so a ring of exactly `window` slots is overwritten ahead of the pass's first query: that
// query's oldest keys are replaced by keys from later in the same chunk (past its causal mask), and
// its attention loses real context. The ring must therefore hold the window plus one whole chunk.
// Rounded up to a power of two because every kernel uses `pos & (slots - 1)`.
int spark_ring_rows(const SparkConfig& c, int prefill_chunk);
size_t spark_ring_bytes(const SparkConfig& c, int prefill_chunk);

// The bytes Spark::alloc_state allocates for one row's activations, which is what MemPlan::activations must report.
// Here rather than in spark.cpp because it must be checkable without a GPU or a checkpoint: the plan's number reaches
// the engine's fit checks and doctor, and it was wrong in both directions before (x `prefill_chunk` on one side, and a
// formula that did not match alloc_state on the other). `cores` is Gpu::core_count() (the split buffers' count).
// `prefill_chunk` is taken but does not yet change the answer: prefill runs a row at a time, so one row is allocated
// whatever the chunk. It is a parameter so that batched prefill changes this function and not its callers.
size_t spark_activation_bytes(const SparkConfig& c, int prefill_chunk, uint32_t cores);

}  // namespace ni
