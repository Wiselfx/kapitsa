// Spark-X2.6 configs (src/model/spark_config.h) against both published checkpoints: the shapes and the derived tables
// the Spark path runs at, and each setting it does not run refused alone. CPU only, no model, no GPU.
//
// The two configs below are verbatim copies of the Hub's, kept in docs/spark/configs/ so this test and the docs cannot
// drift apart; the test reads them from there rather than restating them (a restated copy is a second source of truth).
#include "model/spark_config.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                      \
  do {                                                     \
    if (!(c)) {                                            \
      ++g_fail;                                            \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

// The configs come from docs/spark/configs/, found from the source tree. CI runs the binary from the build
// directory, so the search walks up from the working directory and from the executable's directory; the source dir is
// also baked in by CMake as SPARK_CONFIG_DIR where it is configured.
json load(const std::string& rel) {
  std::vector<std::string> roots;
  if (const char* env = getenv("SPARK_CONFIG_DIR")) roots.push_back(std::string(env) + "/");
#ifdef SPARK_CONFIG_DIR
  roots.push_back(std::string(SPARK_CONFIG_DIR) + "/");
#endif
  for (const char* prefix : {"docs/spark/configs/", "../docs/spark/configs/", "../../docs/spark/configs/",
                             "../../../docs/spark/configs/"})
    roots.push_back(prefix);
  for (const std::string& r : roots) {
    std::ifstream f(r + rel);
    if (!f) continue;
    json j;
    f >> j;
    return j;
  }
  fprintf(stderr, "FAIL: cannot find docs/spark/configs/%s (set SPARK_CONFIG_DIR)\n", rel.c_str());
  ++g_fail;
  return json::object();
}

void check_17b() {
  const json c = load("spark-x2.6-1.7b.config.json");
  CHECK(ni::spark_family(c), "1.7B: spark_family");
  CHECK(ni::spark_unsupported(c).empty(), "1.7B: a published config must run as it is");
  const ni::SparkConfig s = ni::spark_read_config(c);

  CHECK(s.hidden == 2048, "1.7B hidden %d", s.hidden);
  CHECK(s.ffn == 6656, "1.7B ffn %d", s.ffn);
  CHECK(s.layers == 28, "1.7B layers %d", s.layers);
  CHECK(s.hq == 8 && s.hkv == 2, "1.7B heads %d/%d", s.hq, s.hkv);
  CHECK(s.head_dim == 256, "1.7B head_dim %d", s.head_dim);
  CHECK(s.vocab == 131072, "1.7B vocab %d", s.vocab);
  CHECK(s.window == 512, "1.7B window %d", s.window);
  CHECK(s.group() == 4, "1.7B group %d", s.group());

  // the fused q_k_v_proj and the per-head gate
  CHECK(s.q_dim() == 2048, "1.7B q_dim %d", s.q_dim());
  CHECK(s.kv_dim() == 512, "1.7B kv_dim %d", s.kv_dim());
  CHECK(s.qkv_dim() == 3072, "1.7B qkv_dim %d", s.qkv_dim());

  // layer pattern: full attention every 4th, 0-indexed at 3, 7, ...
  const std::vector<int> full = s.full_layers();
  const std::vector<int> want = {3, 7, 11, 15, 19, 23, 27};
  CHECK(full == want, "1.7B full layers differ from the documented table");
  CHECK(s.n_full() == 7, "1.7B n_full %d", s.n_full());
  CHECK(s.n_swa() == 21, "1.7B n_swa %d", s.n_swa());

  // per-layer-type RoPE
  CHECK(s.rot_full == 64, "1.7B rot_full %d", s.rot_full);
  CHECK(s.rot_swa == 256, "1.7B rot_swa %d", s.rot_swa);
  CHECK(s.theta_full == 5000000.0f, "1.7B theta_full %g", s.theta_full);
  CHECK(s.theta_swa == 10000.0f, "1.7B theta_swa %g", s.theta_swa);

  CHECK(s.gate_sigmoid && s.headwise_gate, "1.7B gate settings");
  CHECK(s.tie_embeddings, "1.7B tied embeddings");

  // the derived memory facts the docs quote
  CHECK(ni::spark_kv_per_pos(s) == 14336, "1.7B kv_per_pos %zu", ni::spark_kv_per_pos(s));
  CHECK(ni::spark_ring_bytes(s) == 22020096, "1.7B ring bytes %zu", ni::spark_ring_bytes(s));
}

void check_4b() {
  const json c = load("spark-x2.6-4b.config.json");
  CHECK(ni::spark_family(c), "4B: spark_family");
  CHECK(ni::spark_unsupported(c).empty(), "4B: a published config must run as it is");
  const ni::SparkConfig s = ni::spark_read_config(c);

  CHECK(s.hidden == 2560, "4B hidden %d", s.hidden);
  CHECK(s.ffn == 10240, "4B ffn %d", s.ffn);
  CHECK(s.layers == 36, "4B layers %d", s.layers);
  CHECK(s.hq == 16 && s.hkv == 4, "4B heads %d/%d", s.hq, s.hkv);
  CHECK(s.head_dim == 256, "4B head_dim %d", s.head_dim);
  CHECK(s.vocab == 131072, "4B vocab %d", s.vocab);
  CHECK(s.group() == 4, "4B group %d", s.group());

  CHECK(s.q_dim() == 4096, "4B q_dim %d", s.q_dim());
  CHECK(s.kv_dim() == 1024, "4B kv_dim %d", s.kv_dim());
  CHECK(s.qkv_dim() == 6144, "4B qkv_dim %d", s.qkv_dim());

  const std::vector<int> full = s.full_layers();
  const std::vector<int> want = {3, 7, 11, 15, 19, 23, 27, 31, 35};
  CHECK(full == want, "4B full layers differ from the documented table");
  CHECK(s.n_full() == 9, "4B n_full %d", s.n_full());
  CHECK(s.n_swa() == 27, "4B n_swa %d", s.n_swa());

  CHECK(s.rot_full == 64, "4B rot_full %d", s.rot_full);
  CHECK(s.rot_swa == 256, "4B rot_swa %d", s.rot_swa);
  CHECK(s.theta_full == 5000000.0f, "4B theta_full %g", s.theta_full);
  CHECK(s.theta_swa == 10000.0f, "4B theta_swa %g", s.theta_swa);

  CHECK(ni::spark_kv_per_pos(s) == 36864, "4B kv_per_pos %zu", ni::spark_kv_per_pos(s));
  CHECK(ni::spark_ring_bytes(s) == 56623104, "4B ring bytes %zu", ni::spark_ring_bytes(s));
}

// The layer table without `layer_types` falls back to full attention every `full_attention_interval`, as
// configuration_spark.py does. Both published configs ship layer_types, so the fallback is only reachable for a
// converter that drops it: check it agrees with what they ship.
void check_layer_fallback() {
  for (const char* f : {"spark-x2.6-1.7b.config.json", "spark-x2.6-4b.config.json"}) {
    json c = load(f);
    const std::vector<ni::SparkLayerKind> with = ni::spark_layer_kinds(c);
    c.erase("layer_types");
    const std::vector<ni::SparkLayerKind> without = ni::spark_layer_kinds(c);
    CHECK(with == without, "%s: the (i+1)%%4 fallback disagrees with layer_types", f);
  }
  // and a different interval is honoured
  json c = load("spark-x2.6-1.7b.config.json");
  c.erase("layer_types");
  c["full_attention_interval"] = 2;
  const std::vector<ni::SparkLayerKind> k = ni::spark_layer_kinds(c);
  CHECK(k.size() == 28, "interval 2: %zu layers", k.size());
  CHECK(k[0] == ni::SparkLayerKind::Sliding && k[1] == ni::SparkLayerKind::Full, "interval 2: wrong pattern");
}

// Each setting this path does not run is refused alone, and the refusal names the model.
void check_refusals() {
  struct Case {
    const char* what;
    const char* key;
    json value;
  };
  const std::vector<Case> cases = {
      {"model_type", "model_type", "spark9_9"},
      {"head_dim", "head_dim", 128},
      {"hidden_act", "hidden_act", "silu"},
      {"query heads per KV head", "num_key_value_heads", 8},  // 8/8 -> G=1, not 4
      {"mlp_bias", "mlp_bias", true},
      {"attention_bias", "attention_bias", true},
      {"headwise_attn_output_gate", "headwise_attn_output_gate", false},
      {"gate_attn_act_mode", "gate_attn_act_mode", "silu"},
      {"tie_word_embeddings", "tie_word_embeddings", false},
      {"num_experts", "num_experts", 64},
      {"sliding_window", "sliding_window", 500},  // not a multiple of 16
      {"indexer_n_heads", "indexer_n_heads", 4},
      {"linear_num_key_heads", "linear_num_key_heads", 16},
  };
  for (const Case& cs : cases) {
    json c = load("spark-x2.6-1.7b.config.json");
    c[cs.key] = cs.value;
    const std::vector<std::string> bad = ni::spark_unsupported(c);
    bool found = false;
    for (const std::string& l : bad)
      if (l.find(cs.what) != std::string::npos) found = true;
    CHECK(found, "expected a refusal naming '%s' for %s=%s; got %zu lines", cs.what, cs.key, cs.value.dump().c_str(),
          bad.size());
    // and the assembled refusal is non-empty and mentions the folder
    const std::string r = ni::spark_refusal(c, "/models/spark");
    CHECK(!r.empty() && r.find("/models/spark") != std::string::npos, "refusal text for %s", cs.key);
  }
  // a truly unsupported rope_type
  json c = load("spark-x2.6-1.7b.config.json");
  c["rope_parameters"]["full_attention"]["rope_type"] = "yarn";
  CHECK(!ni::spark_unsupported(c).empty(), "yarn rope must be refused");
  // and the documents run
  CHECK(ni::spark_refusal(load("spark-x2.6-4b.config.json"), "x").empty(), "4B must have no refusal");
}

// The flat rope_parameters shape (rope_theta at the top level, as the Qwen 3.5 checkpoints write it) is read for both
// layer kinds.
void check_flat_rope() {
  json c = load("spark-x2.6-1.7b.config.json");
  c.erase("rope_parameters");
  c["rope_theta"] = 10000.0;
  c["partial_rotary_factor"] = 1.0;
  CHECK(ni::spark_unsupported(c).empty(), "flat rope must be accepted");
  const ni::SparkConfig s = ni::spark_read_config(c);
  CHECK(s.rot_full == 256 && s.rot_swa == 256, "flat rope rot dims %d/%d", s.rot_full, s.rot_swa);
  CHECK(s.theta_full == 10000.0f && s.theta_swa == 10000.0f, "flat rope theta %g/%g", s.theta_full, s.theta_swa);
}

}  // namespace

int main() {
  check_17b();
  check_4b();
  check_layer_fallback();
  check_refusals();
  check_flat_rope();
  if (g_fail) {
    fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  printf("spark_config_test: ok\n");
  return 0;
}
