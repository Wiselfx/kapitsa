// Spark-X2.6 configs (src/model/spark_config.h) against both published checkpoints: the shapes and the derived tables
// the Spark path runs at, and each setting it does not run refused alone. CPU only, no model, no GPU.
//
// The two configs below are verbatim copies of the Hub's, kept in docs/spark/configs/ so this test and the docs cannot
// drift apart; the test reads them from there rather than restating them (a restated copy is a second source of truth).
#include "model/spark_config.h"

// generated at build time from src/kernels/{qmv,wfmt}.metal (tools/spark_kernel_names.py): every matvec kernel the
// sources instantiate. Included so the sweep below checks Spark::matvec_pipe against the real set.
#include "spark_kernel_names.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
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

// The activation is exact (erf) GELU, not the tanh approximation. This check exists because getting it wrong is silent:
// the config says "gelu", ACT2FN maps that to transformers' GELUActivation, and that is nn.functional.gelu with its
// default approximate='none'. Metalside the kernel is a thin wrapper over an Abramowitz-Stegun erf; what is checked
// here is the *decision* (exact vs tanh) at the reference's own values, since a later "simplification" to the tanh
// form would pass every other test in this file.
void check_gelu_is_exact_not_tanh() {
  // erf to 1.5e-7, Abramowitz & Stegun 7.1.26 — the same rational form src/kernels/spark/spark.metal carries.
  auto erf_as = [](double x) {
    const double z = std::fabs(x), t = 1.0 / (1.0 + 0.3275911 * z);
    const double poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    const double r = 1.0 - poly * std::exp(-z * z);
    return x >= 0.0 ? r : -r;
  };
  auto gelu_exact = [&](double x) { return 0.5 * x * (1.0 + erf_as(x * 0.7071067811865476)); };
  auto gelu_tanh = [](double x) { return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x))); };

  // The erf approximation against the true erf: A&S 7.1.26 states 1.5e-7, and measured over x in [-40, 40] the max
  // |error| is 1.4e-7 (at |x| ~ 0.5); carried through gelu the max is 2.1e-7 (near |x| ~ 3). Bound it at 4e-7, i.e.
  // comfortably inside A&S's claim but tight enough to fail if the polynomial is mistyped.
  double worst_erf = 0, worst_gelu = 0;
  for (double x = -40.0; x <= 40.0; x += 0.01) {
    worst_erf = std::max(worst_erf, std::fabs(erf_as(x) - std::erf(x)));
    worst_gelu = std::max(worst_gelu, std::fabs(gelu_exact(x) - 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476))));
  }
  CHECK(worst_erf < 4e-7, "spark_erf max |error| %.2e vs the true erf", worst_erf);
  CHECK(worst_gelu < 4e-7, "gelu max |error| %.2e vs the true erf gelu", worst_gelu);

  // and it must NOT be the tanh form: the two differ by up to ~1.5e-4, which is why the choice matters.
  int differing = 0;
  for (double x = -6.0; x <= 6.0; x += 0.25)
    if (std::fabs(gelu_exact(x) - gelu_tanh(x)) > 1e-5) ++differing;
  CHECK(differing > 8, "the exact and tanh GELUs barely differ here, so this check proves nothing (%d samples)", differing);
  // spot-check a known pair (exact vs tanh at x=1)
  CHECK(std::fabs(gelu_exact(1.0) - 0.8413447460685429) < 1e-7, "gelu_exact(1) = %.9f", gelu_exact(1.0));
  CHECK(std::fabs(gelu_tanh(1.0) - 0.8411919906082768) < 1e-7, "gelu_tanh(1) = %.9f", gelu_tanh(1.0));
}

// The HF-vs-MLX RMSNorm decision (src/model/spark.cpp's zero_centred + the 1 + w conversion). Getting this backwards
// is silent and total: read an HF norm as MLX and every layer is scaled by a number near 0; read an MLX norm as HF and
// every layer is scaled by ~2. Both give fluent-looking garbage, so the rule is pinned here.
void check_norm_layout_rule() {
  // Same rule as zero_centred(): some negative entries and a mean below 0.5 means zero-centred (HF).
  auto zero_centred = [](const std::vector<float>& w) {
    if (w.empty() || w.size() > (1u << 16)) return false;
    double sum = 0;
    int neg = 0;
    for (float v : w) {
      sum += v;
      if (v < 0) ++neg;
    }
    return neg > 0 && sum / double(w.size()) < 0.5;
  };
  // A Hugging Face norm: zero-centred, ~half negative (Spark2_5RMSNorm initialises to ones() but checkpoints converge
  // to a distribution around 0 with both signs).
  CHECK(zero_centred({0.31f, -0.12f, 0.07f, 0.44f, -0.28f, 0.15f}), "an HF-style norm must read as zero-centred");
  // An MLX norm: centred on 1, almost all positive.
  CHECK(!zero_centred({1.02f, 0.98f, 1.05f, 0.95f, 1.01f, 0.99f}), "an MLX-style norm must not read as zero-centred");
  // All-positive but centred low is MLX's shape too (never flips on a monotone-shifted norm).
  CHECK(!zero_centred({0.5f, 0.4f, 0.6f, 0.45f, 0.55f, 0.5f}), "a positive low-mean norm must not flip");
  // A wide tensor is never treated as a norm.
  std::vector<float> wide((1u << 16) + 1, -0.5f);
  CHECK(!zero_centred(wide), "a tensor wider than any norm must not be classified");

  // and the conversion itself: HF's 1 + w must land near 1, which is what the kernels assume.
  auto convert = [](float w) { return 1.0f + w; };
  CHECK(std::fabs(convert(-0.12f) - 0.88f) < 1e-6f, "1 + w must move a zero-centred weight near 1");
}

// Every matvec kernel name Spark's selection rule can emit must exist in the kernel sources. This is checked here, on
// the CPU, because the alternative is finding out at run time as a "no such pipeline" on a Mac -- which is exactly how
// two defects in this port were found.
//
// The first version of this test hand-typed the kernel list and got it wrong twice in ways that made it useless:
// it claimed `qmvg_f16g64_*`, but wfmt.metal's bf16 suffix is `_f16` (SparkLinear::fmt returns "f16"), so it never
// checked the format a bf16 checkpoint actually uses -- and it claimed `qmvg_f8g64_*` decode kernels, which are not
// instantiated at all (the decode group is NI_WF_NO8; 8-bit takes its own qmv2_b8/qmv3_b8 path). A test built from
// memory of the sources is not a check of the sources, so the list is now generated from them at build time
// (cmake/spark_kernel_names.cmake writes spark_kernel_names.h).
void check_matvec_names_exist() {
  std::set<std::string> have;
  for (const char* n = ni::kSparkMatvecNames; *n;) {
    const char* e = n;
    while (*e && *e != '\n') ++e;
    if (e != n) have.insert(std::string(n, size_t(e - n)));
    n = *e ? e + 1 : e;
  }
  CHECK(have.count("qmvg_f16_r2") == 1, "the generated kernel list is readable (bf16 is qmvg_f16_r2, not f16g64)");
  CHECK(have.count("qmv2_b4_g64_r2") == 1, "the generated kernel list has qmv2_b4_g64_r2");
  CHECK(have.count("qmv2_b4_g64_r2_add_norm") == 0, "no _add_norm instantiation is generated");

  // Spark::matvec_pipe's rule, transcribed. Note it never emits _add_norm: no such instantiation exists anywhere.
  auto pick = [](int bits, int K, int N, bool norm, bool add) {
    if (add && norm) return std::string("INVALID");
    const std::string rows4 = N >= 65536 ? "4" : "2";
    std::string name, suffix = add ? "_add" : (norm ? "_norm" : "");
    if (bits == 4) {
      name = K > 4096 ? "qmv3_b4_r" + std::string(add ? "2" : rows4) : "qmv2_b4_g64_r" + rows4;
    } else if (bits == 8) {
      if (!norm) {
        name = "qmv3_b8_r2";
        suffix = add ? "_add" : "";
      } else {
        name = K > 4096 ? "qmv3_b8_r2" : "qmv2_b8_g64_r" + rows4;
      }
    } else {
      // SparkLinear::fmt(): bf16 is "f16" (wfmt.metal's own suffix), not "f16g64". The first version of this sweep
      // used f16g64, so it never checked the format a bf16 checkpoint actually uses.
      const std::string fmt = bits == 16 ? "f16" : "f" + std::to_string(bits) + "g64";
      name = "qmvg_" + fmt + "_r" + (add ? std::string("2") : rows4);
    }
    return name + suffix;
  };

  int checked = 0;
  for (int bits : {4, 8, 16, 6, 5, 3}) {
    for (int K : {2048, 2560, 6656, 10240}) {
      for (int N : {128, 1024, 3072, 6144, 10240, 16384, 65536, 131072}) {
        for (int mode = 0; mode < 3; ++mode) {  // 0 norm, 1 add, 2 neither
          const std::string n = pick(bits, K, N, mode == 0, mode == 1);
          ++checked;
          CHECK(have.count(n) != 0, "matvec_pipe would ask for %s (bits %d, K %d, N %d), which no kernel defines",
                n.c_str(), bits, K, N);
        }
      }
    }
  }
  CHECK(checked > 400, "the matvec sweep only covered %d combinations", checked);
  // and the combination that does not exist must be refused rather than emitted
  CHECK(pick(4, 2560, 2560, true, true) == "INVALID", "_add_norm must be refused: no kernel has both");
}

// MemPlan::activations must equal what alloc_state() actually allocates. It was `prefill_chunk` rows of one row's
// activations, which reserved ~310 MB on the 4B that nothing ever allocates; the number reaches the engine's fit
// checks and doctor, so it made the model look larger than it is and shrank the context a fit allows. A wrong number
// here is not cosmetic in either direction: too high refuses contexts that would fit, too low agrees to one the model
// cannot allocate. Kept as a test rather than a comment because the two expressions live in different functions and
// drifted once already.
void check_plan_matches_alloc() {
  // alloc_state()'s buffers, summed as it allocates them. `plan` is the real function the engine calls, so this is a
  // genuine cross-check: an earlier version of this test re-implemented plan's formula, which meant it passed while the
  // formula was wrong (it multiplied by prefill_chunk and counted ffn twice).
  auto alloc_bytes = [](const ni::SparkConfig& c, uint32_t cores) {
    const size_t H = size_t(c.hidden), Q = size_t(c.q_dim()), KV = size_t(c.kv_dim()), F = size_t(c.ffn),
                 V = size_t(c.vocab), HQ = size_t(c.hq), D = size_t(c.head_dim);
    size_t n = 0;
    n += H * sizeof(float);                        // x_
    n += size_t(c.qkv_dim()) * sizeof(float);      // stg_
    n += KV * sizeof(float) + KV * sizeof(float);  // kr_, vr_
    n += Q * sizeof(float);                        // qr_
    n += size_t(2 * c.q_dim()) * sizeof(float);    // qg_
    n += Q * sizeof(float);                        // ao_
    n += size_t(c.hq) * sizeof(float);             // gate_
    n += size_t(c.ffn) * sizeof(float) * 2;        // gu_
    n += F * sizeof(float);                        // mh_
    n += V * sizeof(float);                        // logits_
    n += HQ * size_t(cores) * D * sizeof(float);   // part_o_
    n += HQ * size_t(cores) * 2 * sizeof(float);   // part_ml_
    n += V * sizeof(float);                        // cand_v_
    n += V * sizeof(int32_t);                      // cand_i_
    n += 64 * sizeof(float);                       // sprob_
    return n;
  };
  for (const char* which : {"spark-x2.6-1.7b.config.json", "spark-x2.6-4b.config.json"}) {
    const ni::SparkConfig c = ni::spark_read_config(load(which));
    for (uint32_t cores : {8u, 20u, 32u, 40u}) {
      const size_t plan = ni::spark_activation_bytes(c, 2048, cores), alloc = alloc_bytes(c, cores);
      CHECK(plan == alloc, "%s: MemPlan::activations is %zu bytes but alloc_state allocates %zu (cores %u)", which,
            plan, alloc, cores);
    }
    // and the chunk size must not change it while prefill is one row at a time
    for (int chunk : {1, 512, 2048, 8192})
      CHECK(ni::spark_activation_bytes(c, chunk, 32) == ni::spark_activation_bytes(c, 1, 32),
            "%s: chunk %d changes the activation plan, but only one row is allocated", which, chunk);
  }
}

}  // namespace

int main() {
  check_17b();
  check_4b();
  check_layer_fallback();
  check_refusals();
  check_flat_rope();
  check_gelu_is_exact_not_tanh();
  check_norm_layout_rule();
  check_plan_matches_alloc();
  check_matvec_names_exist();
  if (g_fail) {
    fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  printf("spark_config_test: ok\n");
  return 0;
}
