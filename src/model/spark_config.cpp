#include "model/spark_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace ni {

using json = nlohmann::json;

namespace {

std::string type_of(const json& c) {
  return c.is_object() && c.contains("model_type") && c["model_type"].is_string() ? c["model_type"].get<std::string>()
                                                                                  : std::string();
}

// a number setting, `def` when it is absent or null
long long num(const json& t, const char* k, long long def) {
  return t.contains(k) && t[k].is_number() ? t[k].get<long long>() : def;
}

double real(const json& t, const char* k, double def) {
  return t.contains(k) && t[k].is_number() ? t[k].get<double>() : def;
}

std::string str(const json& t, const char* k) {
  return t.contains(k) && t[k].is_string() ? t[k].get<std::string>() : std::string();
}

bool flag(const json& t, const char* k, bool def) {
  return t.contains(k) && t[k].is_boolean() ? t[k].get<bool>() : def;
}

std::string fmt(const char* f, long long a) {
  char b[192];
  snprintf(b, sizeof b, f, a);
  return b;
}

// "rope_parameters": {"full_attention": {...}, "sliding_attention": {...}}. Transformers also accepts the flat form
// (rope_theta / partial_rotary_factor at the top level), which the Qwen 3.5 family checkpoints use; both shapes are
// read here so a converter that writes either one loads.
const json& rope_of(const json& t, const char* which) {
  static const json empty = json::object();
  if (!t.contains("rope_parameters") || !t["rope_parameters"].is_object()) return empty;
  const json& rp = t["rope_parameters"];
  if (rp.contains(which) && rp[which].is_object()) return rp[which];
  // a flat rope_parameters (no per-layer-type keys) applies to both kinds
  if (!rp.contains("full_attention") && !rp.contains("sliding_attention")) return rp;
  return empty;
}

int rot_dims(const json& layer_rope, const json& top, int head_dim) {
  double prf = 1.0;
  if (layer_rope.contains("partial_rotary_factor") && layer_rope["partial_rotary_factor"].is_number())
    prf = layer_rope["partial_rotary_factor"].get<double>();
  else if (top.contains("partial_rotary_factor") && top["partial_rotary_factor"].is_number())
    prf = top["partial_rotary_factor"].get<double>();
  return int(std::lround(head_dim * prf));
}

float theta_of(const json& layer_rope, const json& top, float def) {
  if (layer_rope.contains("rope_theta") && layer_rope["rope_theta"].is_number())
    return float(layer_rope["rope_theta"].get<double>());
  if (top.contains("rope_theta") && top["rope_theta"].is_number()) return float(top["rope_theta"].get<double>());
  return def;
}

void add_ids(const json& v, std::vector<int>& out) {
  if (v.is_number_integer()) out.push_back(v.get<int>());
  if (v.is_array())
    for (const json& x : v)
      if (x.is_number_integer() && std::find(out.begin(), out.end(), x.get<int>()) == out.end())
        out.push_back(x.get<int>());
}

}  // namespace

bool spark_family(const json& c) {
  if (!c.is_object()) return false;
  const std::string mt = type_of(c);
  return mt == "spark2_5" || mt.rfind("spark2", 0) == 0;
}

std::vector<SparkLayerKind> spark_layer_kinds(const json& c) {
  const long long layers = num(c, "num_hidden_layers", 0);
  std::vector<SparkLayerKind> v;
  if (layers <= 0) return v;
  v.reserve(size_t(layers));
  if (c.contains("layer_types") && c["layer_types"].is_array() && c["layer_types"].size() == size_t(layers)) {
    for (const json& x : c["layer_types"])
      v.push_back(x.is_string() && x.get<std::string>() == "full_attention" ? SparkLayerKind::Full
                                                                           : SparkLayerKind::Sliding);
    return v;
  }
  // configuration_spark.py's fallback: "full_attention" every full_attention_interval layers (Spark's own 3:1 pattern
  // is interval 4), everything else sliding.
  const long long interval = num(c, "full_attention_interval", 4);
  for (long long i = 0; i < layers; ++i)
    v.push_back(interval > 0 && (i + 1) % interval == 0 ? SparkLayerKind::Full : SparkLayerKind::Sliding);
  return v;
}

std::vector<std::string> spark_unsupported(const json& c) {
  std::vector<std::string> v;
  if (!c.is_object()) return {"config.json is not a JSON object"};
  const std::string mt = type_of(c);
  if (mt != "spark2_5")
    v.push_back("model_type " + (mt.empty() ? std::string("(none)") : mt) +
                ": Kapitsa runs the spark2_5 family (Spark-X2.6-1.7B and -4B); a new Spark architecture runs once its "
                "differences are read");

  const long long hidden = num(c, "hidden_size", 0);
  if (hidden <= 0 || hidden % 64)
    v.push_back(fmt("hidden_size %lld: a multiple of 64 (the quantized matvec reads whole weight groups)", hidden));
  if (hidden > 4096) v.push_back(fmt("hidden_size %lld: at most 4096", hidden));
  const long long ffn = num(c, "intermediate_size", 0);
  if (ffn <= 0 || ffn % 64) v.push_back(fmt("intermediate_size %lld: a multiple of 64", ffn));
  if (const long long l = num(c, "num_hidden_layers", 0); l <= 0) v.push_back(fmt("num_hidden_layers %lld", l));

  // 256-wide heads are what the attention kernels take (attn_prep / attn_prefill11r / qattn_t).
  const long long hq = num(c, "num_attention_heads", 0);
  const long long hd = num(c, "head_dim", hq > 0 && hidden > 0 ? hidden / hq : 0);
  if (hd != 256) v.push_back(fmt("head_dim %lld: the attention kernels take 256", hd));
  if (hq <= 0) v.push_back(fmt("num_attention_heads %lld", hq));
  const long long hkv = num(c, "num_key_value_heads", hq);
  if (hkv <= 0) v.push_back(fmt("num_key_value_heads %lld", hkv));
  if (hq > 0 && hkv > 0) {
    if (hq % hkv) v.push_back(fmt("num_attention_heads %lld: a whole number of query heads per KV head", hq));
    else if (const long long g = hq / hkv; g != 4)
      v.push_back(fmt("query heads per KV head %lld: 4 (both Spark-X2.6 models; the kernel is written for 4)", g));
  }

  // The activation the MLP kernels are templated for.
  if (const std::string act = str(c, "hidden_act"); !act.empty() && act != "gelu")
    v.push_back("hidden_act " + act + ": gelu (tanh approximation)");

  // The per-head sigmoid output gate is part of this family's attention kernel, not an option.
  if (!flag(c, "headwise_attn_output_gate", true))
    v.push_back("headwise_attn_output_gate false: no output gate (the attention kernels apply one)");
  if (const std::string m = str(c, "gate_attn_act_mode"); !m.empty() && m != "sigmoid")
    v.push_back("gate_attn_act_mode " + m + ": sigmoid");

  // No q/k RMSNorm in this family: the attention path folds no norm there. A checkpoint that has one needs the
  // kernel variant that does, so refuse it rather than silently skipping a norm.
  if (c.contains("use_qk_norm") && flag(c, "use_qk_norm", false))
    v.push_back("use_qk_norm true: this family has no q/k RMSNorm");

  // Biases: every kernel path here is bias-free except the quantized scales/biases of the weight format itself.
  if (flag(c, "attention_bias", false)) v.push_back("attention_bias true: no attention biases");
  if (flag(c, "mlp_bias", false)) v.push_back("mlp_bias true: no MLP biases");

  // Sliding window: the ring cache and the mask both need a real window.
  const auto kinds = spark_layer_kinds(c);
  const bool any_swa = std::count(kinds.begin(), kinds.end(), SparkLayerKind::Sliding) > 0;
  if (any_swa) {
    const long long w = num(c, "sliding_window", 0);
    if (w <= 0) v.push_back(fmt("sliding_window %lld: a window for the sliding layers", w));
    // the prefill attention tiles keys 16 or 32 at a time and the ring keeps whole positions: a window that is not a
    // multiple of 16 would need a partial tile
    else if (w % 16) v.push_back(fmt("sliding_window %lld: a multiple of 16", w));
  }

  if (c.contains("layer_types") && c["layer_types"].is_array()) {
    const json& lt = c["layer_types"];
    if (num(c, "num_hidden_layers", 0) != (long long)lt.size())
      v.push_back(fmt("layer_types names %lld layers, not num_hidden_layers", (long long)lt.size()));
    std::set<std::string> other;
    for (const json& x : lt)
      if (!x.is_string() || (x != "sliding_attention" && x != "full_attention"))
        other.insert(x.is_string() ? x.get<std::string>() : x.dump());
    for (const std::string& o : other) v.push_back("layer_types has " + o + " layers: sliding_attention or full_attention");
  }

  // RoPE: both kinds must be plain (no YaRN / linear scaling), and the rotary dims must be usable.
  const json& rp = c.contains("rope_parameters") && c["rope_parameters"].is_object() ? c["rope_parameters"]
                                                                                     : json::object();
  for (const char* which : {"full_attention", "sliding_attention"}) {
    const json& r = rope_of(c, which);
    const std::string ty = str(r, "rope_type");
    if (!ty.empty() && ty != "default")
      v.push_back(std::string("rope_type ") + ty + " for " + which + ": plain RoPE");
    const int dims = rot_dims(r, rp, int(hd > 0 ? hd : 0));
    if (hd == 256 && (dims <= 0 || dims > 256 || dims % 2))
      v.push_back(fmt("rope dims %lld: an even count from 2 to head_dim", dims));
  }

  // MoE / sparse-attention settings this dense path does not run.
  if (num(c, "num_experts", 0) > 0) v.push_back(fmt("num_experts %lld: a dense MLP", num(c, "num_experts", 0)));
  if (c.contains("indexer_n_heads")) v.push_back("indexer_n_heads: full attention without a sparse indexer");
  // Gated DeltaNet / Mamba settings belong to other families
  for (const char* k : {"linear_num_key_heads", "linear_conv_kernel_dim", "mamba_ssm_dtype"})
    if (c.contains(k)) v.push_back(std::string(k) + ": this family has no recurrent (GDN / Mamba) layers");

  return v;
}

SparkConfig spark_read_config(const json& c) {
  SparkConfig s;
  s.model_type = type_of(c);
  s.hidden = int(num(c, "hidden_size", 0));
  s.ffn = int(num(c, "intermediate_size", 0));
  s.layers = int(num(c, "num_hidden_layers", 0));
  s.vocab = int(num(c, "vocab_size", 0));
  s.hq = int(num(c, "num_attention_heads", 0));
  s.hkv = int(num(c, "num_key_value_heads", s.hq));
  s.head_dim = int(num(c, "head_dim", s.hq > 0 ? s.hidden / s.hq : 0));
  s.window = int(num(c, "sliding_window", 0));
  s.eps = float(real(c, "rms_norm_eps", 1e-6));
  s.gate_sigmoid = str(c, "gate_attn_act_mode").empty() || str(c, "gate_attn_act_mode") == "sigmoid";
  s.headwise_gate = flag(c, "headwise_attn_output_gate", true);
  s.tie_embeddings = flag(c, "tie_word_embeddings", true);
  s.kinds = spark_layer_kinds(c);

  const json& rp = c.contains("rope_parameters") && c["rope_parameters"].is_object() ? c["rope_parameters"]
                                                                                     : json::object();
  const json& rf = rope_of(c, "full_attention");
  const json& rs = rope_of(c, "sliding_attention");
  s.rot_full = rot_dims(rf, rp, s.head_dim);
  s.rot_swa = rot_dims(rs, rp, s.head_dim);
  s.theta_full = theta_of(rf, rp, 1e4f);
  s.theta_swa = theta_of(rs, rp, 1e4f);

  if (c.contains("eos_token_id")) add_ids(c["eos_token_id"], s.eos);
  return s;
}

std::string spark_refusal(const json& c, const std::string& what) {
  const std::vector<std::string> bad = spark_unsupported(c);
  if (bad.empty()) return "";
  std::string out = what + " is a Spark model Kapitsa does not run yet:";
  for (const std::string& l : bad) out += "\n  " + l;
  return out;
}

int SparkConfig::n_full() const {
  return int(std::count(kinds.begin(), kinds.end(), SparkLayerKind::Full));
}

int SparkConfig::n_swa() const {
  return int(std::count(kinds.begin(), kinds.end(), SparkLayerKind::Sliding));
}

std::vector<int> SparkConfig::full_layers() const {
  std::vector<int> v;
  for (size_t i = 0; i < kinds.size(); ++i)
    if (kinds[i] == SparkLayerKind::Full) v.push_back(int(i));
  return v;
}

std::string SparkConfig::formats() const {
  char b[256];
  snprintf(b, sizeof b, "dense %d layers (%d full, %d sliding window %d), hidden %d, ffn %d, %d/%d heads x %d",
           layers, n_full(), n_swa(), window, hidden, ffn, hq, hkv, head_dim);
  return b;
}

size_t spark_kv_per_pos(const SparkConfig& c) {
  // k and v per full-attention layer, as half: [Hkv, cap, D]
  return size_t(c.n_full()) * size_t(c.hkv) * size_t(c.head_dim) * 2 * sizeof(uint16_t);
}

int spark_ring_rows(const SparkConfig& c, int prefill_chunk) {
  // window + a whole chunk, so writing T rows before attention cannot overwrite any query's window.
  const int need = std::max(1, c.window) + std::max(1, prefill_chunk);
  int rows = 1;
  while (rows < need) rows <<= 1;
  return rows;
}

size_t spark_ring_bytes(const SparkConfig& c, int prefill_chunk) {
  // The sliding layers' rings: the window plus one prefill chunk (spark_ring_rows), not the context
  // length. The chunk term is what makes a batched prefill correct on a sliding layer; it also means
  // this number depends on prefill_chunk, so plan() passes it.
  return size_t(c.n_swa()) * size_t(c.hkv) * size_t(spark_ring_rows(c, prefill_chunk)) * size_t(c.head_dim) * 2 *
         sizeof(uint16_t);
}

size_t spark_activation_bytes(const SparkConfig& c, int prefill_chunk, uint32_t cores) {
  const size_t H = size_t(c.hidden), Q = size_t(c.q_dim()), KV = size_t(c.kv_dim()), F = size_t(c.ffn),
               V = size_t(c.vocab), HQ = size_t(c.hq), D = size_t(c.head_dim), n = size_t(cores ? cores : 1);
  // The batched-prefill set, T = prefill_chunk rows. This used to ignore its argument on the grounds that prefill ran one
  // row whatever the chunk; it does not any more, and forgetting to grow these would have the plan promise a peak the
  // allocator then exceeds -- on the 4B at the default chunk that is ~1.2 GB of half/float rows.
  // The fp32 rows are counted in floats (the unit of the sum below); hh_ and xh_ are half and are added as bytes, so they
  // are kept out of this term rather than scaled inside it -- mixing a /2 into an expression multiplied by sizeof(float)
  // is how the first version of this came out 3.6x too small.
  const size_t T = size_t(prefill_chunk > 0 ? prefill_chunk : 1);
  const size_t batch_f32 = T * (H                     // xb_  fp32, the residual
                                + size_t(c.qkv_dim())  // stgb_ fp32, q|k|v over T rows
                                + Q + 2 * Q            // qrb_, qgb_ fp32
                                + Q                    // aob_  fp32
                                + 2 * F);              // gub_  fp32, gate and up over T rows
  const size_t batch_half = T * (F + H);              // hh_ (F halves), xh_ (H halves)
  const size_t batch = batch_f32 * sizeof(float) + batch_half * 2;
  return batch + (H              // x_
          + size_t(c.qkv_dim())  // stg_  (q | k | v in one matvec's output)
          + 2 * KV               // kr_, vr_
          + Q + 2 * Q            // qr_, qg_
          + Q                    // ao_
          + HQ                   // gate_
          + 2 * F + F            // gu_ (gate and up), mh_
          + V                    // logits_
          + HQ * n * D           // part_o_
          + HQ * n * 2           // part_ml_
          + V + V                // cand_v_, cand_i_ (int32, one float wide)
          + V                    // lgb_ (the prefill head's logits, sized by the vocabulary)
          + 64                   // sprob_
          ) *
         sizeof(float);
}

}  // namespace ni
