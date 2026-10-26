#include "model/llama.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <unordered_map>

namespace ni {

extern const char* const kFlintKernelSource;

namespace {

// host mirrors of src/kernels/flint/flint.metal's parameter structs (FlintPrepP, FlintAttnP, FlintAttnPfP,
// FlintStoreP, the SwiGLU's)
struct PrepP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, rope, chunk, qkv_ld;
  float eps;
  uint32_t tree;
};
struct AttnP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, window, k0, chunk, nchunks;
  float scale;
};
struct AttnPfP {
  uint32_t T, pos0, Hq, Hkv, rows, ring, window;
  float scale;
};
struct StoreP {
  uint32_t T, t0, pos0, Hkv, rows, ring;
};
struct SwigluP {
  uint32_t I, mode, k;
};

Pipeline pipe(const std::string& n) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return {};
  }
}

// GPU shader validation's instrumentation doubles a kernel's threadgroup memory: flint_attn_prefill's 32-key tiles
// (17,408 B) are over its limit, the 16-key ones are not (NI_FLINT_K16=1 selects them too)
bool small_tiles() {
  static const bool v = getenv("MTL_SHADER_VALIDATION") || env_on("NI_FLINT_K16");
  return v;
}

uint32_t gpu_cores() {
  static const uint32_t c = uint32_t(std::max(1, Gpu::get().core_count()));
  return c;
}

// The bits of an MLX affine linear from its headers: codes per row (bytes or uint32 words) over inputs per row
int bits_of(int64_t code_cols, bool u8, int64_t scale_cols, int gs) {
  const int64_t K = scale_cols * gs, b = code_cols * (u8 ? 8 : 32);
  return K > 0 && b % K == 0 ? int(b / K) : 0;
}

}  // namespace

// ---- config ----
bool Llama::is_model(const std::string& dir) {
  const nlohmann::json j = read_json(dir + "/config.json");
  return j.is_object() && j.value("model_type", std::string()) == "llama";
}

namespace {

// config.json's fields as the model reads them, nothing refused yet
LlamaConfig parse_config(const nlohmann::json& j) {
  auto num = [&](const char* k, double d) {
    const auto it = j.find(k);
    return it != j.end() && it->is_number() ? it->get<double>() : d;
  };
  LlamaConfig c;
  c.hidden = int(num("hidden_size", 0));
  c.vocab = int(num("vocab_size", 0));
  c.layers = int(num("num_hidden_layers", 0));
  c.hq = int(num("num_attention_heads", 0));
  c.hkv = int(num("num_key_value_heads", c.hq));
  c.hd = int(num("head_dim", c.hq ? c.hidden / c.hq : 0));
  c.ffn = int(num("intermediate_size", 0));
  c.eps = float(num("rms_norm_eps", 1e-5));
  c.theta = num("rope_theta", 1e4);
  if (j.contains("rope_parameters") && j["rope_parameters"].is_object())
    c.theta = j["rope_parameters"].value("rope_theta", c.theta);
  if (j.contains("eos_token_id")) {
    const auto& e = j["eos_token_id"];
    if (e.is_array()) c.eos = e.get<std::vector<int>>();
    else if (e.is_number()) c.eos = {e.get<int>()};
  }
  c.bos = int(num("bos_token_id", -1));
  const nlohmann::json q = j.value("quantization", nlohmann::json::object());
  if (q.is_object()) {
    c.bits = q.value("bits", 4);
    c.gs = q.value("group_size", 64);
  }
  return c;
}

}  // namespace

std::string Llama::refusal(const nlohmann::json& j) {
  if (!j.is_object()) return "no readable config.json";
  if (j.contains("rope_parameters") && j["rope_parameters"].is_object() &&
      j["rope_parameters"].value("rope_type", std::string("default")) != "default")
    return "Llama RoPE other than the default is not supported";
  if (j.contains("rope_scaling") && !j["rope_scaling"].is_null())
    return "Llama with rope_scaling is not supported (plain RoPE only)";
  if (j.value("attention_bias", false) || j.value("mlp_bias", false))
    return "Llama with attention or MLP biases is not supported";
  if (const std::string act = j.value("hidden_act", std::string("silu")); act != "silu")
    return "Llama with a " + act + " MLP is not supported (SwiGLU)";
  const nlohmann::json q = j.value("quantization", nlohmann::json::object());
  if (!q.is_object() || q.empty())
    return "an unquantized Llama checkpoint; Kapitsa runs MLX 4- or 8-bit exports (mlx_lm.convert -q)";
  if (q.value("mode", std::string("affine")) != "affine")
    return "Llama in MLX " + q.value("mode", std::string()) + " quantization is not supported (affine only)";
  const LlamaConfig c = parse_config(j);
  if ((c.bits != 4 && c.bits != 8) || c.gs != 64)
    return "Llama in " + std::to_string(c.bits) + "-bit group " + std::to_string(c.gs) +
           " is not supported (MLX 4- or 8-bit, group 64)";
  // hd 128: Flint's attention kernels; at most 8 query heads a KV head: flint_attn_prefill's 256 threads; widths in
  // 512s: the matvecs' 32 lanes x 16 inputs a step (qmvf)
  if (c.hidden <= 0 || c.hidden % 512 || c.vocab <= 0 || c.layers <= 0 || c.hq <= 0 || c.hkv <= 0 || c.hq % c.hkv ||
      c.hq / c.hkv > 8 || c.hd != 128 || (c.hq * c.hd) % 512 || c.ffn <= 0 || c.ffn % 512)
    return "a Llama shape Kapitsa does not run (hidden " + std::to_string(c.hidden) + ", heads " +
           std::to_string(c.hq) + " / " + std::to_string(c.hkv) + " of " + std::to_string(c.hd) + ", intermediate " +
           std::to_string(c.ffn) + ")";
  return {};
}

LlamaConfig Llama::read_config(const std::string& dir) {
  const nlohmann::json j = read_json(dir + "/config.json");
  if (const std::string why = refusal(j); !why.empty()) fatal("%s: %s", dir.c_str(), why.c_str());
  return parse_config(j);
}

size_t Llama::kv_per_pos(const LlamaConfig& c) { return size_t(c.layers) * 2 * size_t(c.hkv) * size_t(c.hd) * 2; }

MemPlan Llama::plan(const std::string& path, int prefill_chunk) {
  MemPlan m;
  const LlamaConfig c = read_config(path);
  const SafetensorsIndex idx(path);
  size_t embed = 0;
  for (const auto& [name, t] : idx.all()) {
    m.weights += t.len;
    if (name.compare(0, 19, "model.embed_tokens.") == 0) embed += t.len;
  }
  const size_t D = size_t(c.hidden), T = size_t(std::max(kMaxRows, prefill_chunk)), F = size_t(c.ffn);
  const size_t QD = size_t(c.hq) * c.hd, KVD = size_t(c.hkv) * c.hd, QKV = QD + 2 * KVD, V = size_t(c.vocab);
  // the prefill buffers (alloc_prefill) and the decode rows' logits
  m.activations = T * D * 4 + T * std::max(D, QD) * 2 + T * QKV * 4 + T * QD * 4 + 2 * T * KVD * 2 + T * QD * 2 +
                  2 * T * F * 4 + T * F * 2 + T * D * 4 + T * 4 + size_t(kMaxRows) * V * 4;
  m.kv_per_pos = kv_per_pos(c);
  m.decode_params = size_t(c.layers) * (D * (QKV + QD) + 3 * D * F) + D * V;
  // a decode step reads every weight but the embedding (one row of it)
  const double bpw = (c.bits + 32.0 / c.gs) / 8.0;
  m.decode_bytes = m.weights - embed + size_t(double(D) * bpw);
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = c.ffn;
  m.dense = true;
  m.ffn_ane = false;
  auto bits = [&](const std::string& n, int def) {
    const SafetensorsIndex::Entry* w = idx.find(n + ".weight");
    const SafetensorsIndex::Entry* sc = idx.find(n + ".scales");
    if (!w || !sc || w->shape.empty() || sc->shape.empty()) return def;
    const int b = bits_of(w->shape.back(), w->dtype == DType::U8, sc->shape.back(), c.gs);
    return b ? b : def;
  };
  const int eb = bits("model.embed_tokens", c.bits);
  const int hb = idx.find("lm_head.weight") ? bits("lm_head", c.bits) : eb;
  m.head_row_bytes = size_t(double(D) * (hb + 32.0 / c.gs) / 8.0);
  m.formats = std::to_string(c.bits) + "-bit group " + std::to_string(c.gs) + " + " + std::to_string(eb) +
              "-bit embedding" + (hb != c.bits ? ", " + std::to_string(hb) + "-bit LM head" : "");
  return m;
}

// ---- load ----
Llama::Llama(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not a Llama checkpoint folder (config.json)", path.c_str());
  cfg_ = read_config(path_);
  Gpu::get().compile_extra("Flint", kFlintKernelSource);  // Flint's attention kernels and llama_prep
  load(prefill_chunk);
}

Llama::~Llama() = default;

void Llama::load(int prefill_chunk) {
  const double t0 = now_ms();
  tmax_ = std::max(kMaxRows, prefill_chunk);
  ws_.open_dir(path_);
  // a linear: codes (uint8 or uint32 words of the same stream) with bf16 scales / biases per group of 64 inputs. The
  // bits come from the shapes (MLX's per-module overrides), the group size from the config.
  auto mat = [&](const std::string& base) {
    LlamaMat m;
    const Tensor* w = ws_.find(base + ".weight");
    if (!w) fatal("%s: tensor %s.weight missing", path_.c_str(), base.c_str());
    const Tensor* s = ws_.find(base + ".scales");
    const Tensor* b = ws_.find(base + ".biases");
    if (!s || !b || (w->dtype != DType::U8 && w->dtype != DType::U32) || w->shape.size() != 2 || s->shape.size() != 2)
      fatal("%s: %s is not an MLX affine-quantized linear", path_.c_str(), base.c_str());
    m.w = w, m.s = s, m.b = b;
    m.gs = cfg_.gs;
    m.N = uint32_t(w->shape[0]);
    m.K = uint32_t(s->shape[1]) * uint32_t(m.gs);
    m.bits = bits_of(w->shape[1], w->dtype == DType::U8, s->shape[1], m.gs);
    if (!m.bits) fatal("%s: %s's codes do not match its scales", path_.c_str(), base.c_str());
    return m;
  };
  auto vec = [&](const std::string& name, int64_t n) {
    const Tensor* t = ws_.find(name);
    if (!t || t->dtype != DType::BF16 || t->numel() != n)
      fatal("%s: bf16 tensor %s [%lld] missing", path_.c_str(), name.c_str(), (long long)n);
    return t;
  };
  const LlamaConfig& c = cfg_;
  embed_ = mat("model.embed_tokens");
  head_ = ws_.find("lm_head.weight") ? mat("lm_head") : embed_;  // tied embeddings: the head reads the embedding
  final_norm_ = vec("model.norm.weight", c.hidden);
  if (embed_.N != uint32_t(c.vocab) || embed_.K != uint32_t(c.hidden) || head_.N != uint32_t(c.vocab) ||
      head_.K != uint32_t(c.hidden))
    fatal("%s: embedding or LM head shape differs from the config", path_.c_str());
  // embed_q4 / embed_f{3,8}g64 (wfmt.metal); the head on qmvf_b4 / b6 / b8
  if (embed_.bits != 3 && embed_.bits != 4 && embed_.bits != 8)
    fatal("%s: a %d-bit embedding (Llama: 3, 4 or 8)", path_.c_str(), embed_.bits);
  if (head_.bits != 4 && head_.bits != 6 && head_.bits != 8)
    fatal("%s: a %d-bit LM head (Llama: 4, 6 or 8)", path_.c_str(), head_.bits);
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd), D = uint32_t(c.hidden), F = uint32_t(c.ffn);
  layers_.resize(size_t(c.layers));
  for (int l = 0; l < c.layers; ++l) {
    LlamaLayer& L = layers_[size_t(l)];
    const std::string p = "model.layers." + std::to_string(l) + ".";
    L.in_norm = vec(p + "input_layernorm.weight", c.hidden);
    L.post_norm = vec(p + "post_attention_layernorm.weight", c.hidden);
    L.q = mat(p + "self_attn.q_proj");
    L.k = mat(p + "self_attn.k_proj");
    L.v = mat(p + "self_attn.v_proj");
    L.o = mat(p + "self_attn.o_proj");
    L.mg = mat(p + "mlp.gate_proj");
    L.mu = mat(p + "mlp.up_proj");
    L.md = mat(p + "mlp.down_proj");
    if (L.q.N != QD || L.k.N != KD || L.v.N != KD || L.q.K != D || L.k.K != D || L.v.K != D || L.o.N != D ||
        L.o.K != QD || L.mg.N != F || L.mu.N != F || L.mg.K != D || L.mu.K != D || L.md.N != D || L.md.K != F)
      fatal("%s: layer %d's shapes differ from the config", path_.c_str(), l);
    // one matvec takes q | k | v (and gate | up) together: one format for each group; 4 or 8 bits (the matvecs' and the
    // prefill GEMMs' formats)
    for (const LlamaMat* m : {&L.q, &L.k, &L.v, &L.o, &L.mg, &L.mu, &L.md})
      if (m->bits != 4 && m->bits != 8)
        fatal("%s: layer %d has a %d-bit projection (Llama: 4 or 8)", path_.c_str(), l, m->bits);
    if (L.k.bits != L.q.bits || L.v.bits != L.q.bits || L.mu.bits != L.mg.bits)
      fatal("%s: layer %d's q / k / v or gate / up differ in bits", path_.c_str(), l);
  }
  // RoPE's inverse frequencies, 1 / theta^(2i / D) (HF's default rotary embedding; mlx_lm's nn.RoPE, traditional=False)
  Gpu& g = Gpu::get();
  inv_freq_ = g.alloc(size_t(c.hd / 2) * 4, "llama_rope", false);
  for (int i = 0; i < c.hd / 2; ++i)
    inv_freq_.as<float>()[i] = float(1.0 / std::pow(c.theta, double(2 * i) / double(c.hd)));
  sinks_ = g.alloc(size_t(c.hq) * 2, "llama_sinks", false);
  for (int h = 0; h < c.hq; ++h) sinks_.as<uint16_t>()[h] = f32_to_bf16(-1e30f);
  alloc_buffers();
  g.make_resident(ws_.buffers());
  logi("Llama: %d layers, %d / %d heads, %d-bit (embedding %d-bit, LM head %d-bit%s), loaded in %.1f s", c.layers, c.hq,
       c.hkv, c.bits, embed_.bits, head_.bits, ws_.find("lm_head.weight") ? "" : ", tied",
       (now_ms() - t0) / 1000.0);
}

void Llama::alloc_buffers() {
  Gpu& g = Gpu::get();
  const LlamaConfig& c = cfg_;
  const size_t R = kMaxRows, D = size_t(c.hidden), QD = size_t(c.hq) * c.hd, V = size_t(c.vocab);
  const size_t QKV = QD + 2 * size_t(c.hkv) * c.hd;
  const size_t max_chunks = 64;  // the decode attention's key chunks (enc_layer_rows: at most 64)
  x_ = g.alloc(R * D * 4, "llama_x");
  qkv_ = g.alloc(R * QKV * 4, "llama_qkv");
  q_ = g.alloc(R * QD * 4, "llama_q");
  ao_ = g.alloc(R * QD * 4, "llama_ao");
  mh_ = g.alloc(R * size_t(c.ffn) * 4, "llama_mlp_h");
  part_o_ = g.alloc(R * size_t(c.hq) * max_chunks * c.hd * 4, "llama_part_o");
  part_ml_ = g.alloc(R * size_t(c.hq) * max_chunks * 2 * 4, "llama_part_ml");
  logits_ = g.alloc(R * V * 4, "llama_logits");
  cand_v_ = g.alloc(R * (V / 2048 + 2) * 64 * 4, "llama_cand_v");
  cand_i_ = g.alloc(R * (V / 2048 + 2) * 64 * 4, "llama_cand_i");
  sprob_ = g.alloc(64 * 4, "llama_sprob");
  samp_ = g.alloc(64 * 4, "llama_samples");
}

void Llama::alloc_prefill() {
  if (prefill_alloc_) return;
  Gpu& g = Gpu::get();
  const LlamaConfig& c = cfg_;
  const size_t T = size_t(tmax_), D = size_t(c.hidden), QD = size_t(c.hq) * c.hd, KVD = size_t(c.hkv) * c.hd;
  const size_t F = size_t(c.ffn);
  px_ = g.alloc(T * D * 4, "llama_px");
  pxh_ = g.alloc(T * std::max(D, QD) * 2, "llama_pxh");
  pqkv_ = g.alloc(T * (QD + 2 * KVD) * 4, "llama_pqkv");
  pq_ = g.alloc(T * QD * 4, "llama_pq");
  pck_ = g.alloc(T * KVD * 2, "llama_pck");
  pcv_ = g.alloc(T * KVD * 2, "llama_pcv");
  pao_ = g.alloc(T * QD * 2, "llama_pao");
  phg_ = g.alloc(T * F * 4, "llama_phg");
  phu_ = g.alloc(T * F * 4, "llama_phu");
  phh_ = g.alloc(T * F * 2, "llama_phh");
  ys_ = g.alloc(T * D * 4, "llama_ys");
  pscale_ = g.alloc(T * 4, "llama_pscale");
  prefill_alloc_ = true;
}

// ---- sequences, KV caches, checkpoints ----
std::unique_ptr<Seq> Llama::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  s->kc.resize(size_t(cfg_.layers));
  s->vc.resize(size_t(cfg_.layers));
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token KV cache", cap);
  return s;
}

// Every layer's cache [Hkv][cap][hd] grows by a quarter, in 1024s (256s when that does not fit), the old rows copied
bool Llama::reserve(Seq& s, int need) {
  if (need <= s.cap) return true;
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_per_pos(cfg_), row = size_t(cfg_.hkv) * cfg_.hd * 2;
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  auto fits = [&](int c) {
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || g.allocated_bytes() + size_t(c - s.cap) * per + copy <= mem_budget_;
  };
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;
  const int old = s.cap;
  for (int l = 0; l < cfg_.layers; ++l)
    for (Buffer* b : {&s.kc[size_t(l)], &s.vc[size_t(l)]}) {
      Buffer nbuf = g.alloc(row * size_t(target), "llama_kv");
      memset(nbuf.ptr(), 0, nbuf.size);  // (no kernel reads a position before it is written; zeroed all the same)
      if (old > 0 && *b) {
        const uint32_t n16 = uint32_t(cfg_.hd) * 2 / 16;
        const uint32_t p[4] = {uint32_t(old) * n16, uint32_t(target) * n16, uint32_t(cfg_.hkv), 0};
        auto e = g.begin();
        e->use(pipe("kv_regrow"));
        e->buf(0, *b);
        e->buf(1, nbuf);
        e->bytes(2, p, sizeof p);
        e->dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
        e->commit_and_wait();
      }
      if (*b) g.release(*b);
      *b = nbuf;
    }
  s.cap = target;
  return true;
}

void Llama::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Llama::reset(Seq& s) { (void)s; }  // no recurrent state: every position is written before it is read

// Every layer's cache keeps every earlier position: a checkpoint is its position
void Llama::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  (void)e;
  (void)s;
  c.pos = pos;
}

void Llama::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  (void)e;
  (void)s;
  (void)c;
}

std::string Llama::disk_format() const {
  char b[256];
  snprintf(b, sizeof b, "llama %d layers; K, V f16 [%d][cap][%d]; %d-bit group %d (embedding %d-bit, LM head %d-bit)",
           cfg_.layers, cfg_.hkv, cfg_.hd, cfg_.bits, cfg_.gs, embed_.bits, head_.bits);
  return b;
}

void Llama::disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const {
  (void)mtp;
  for (int l = 0; l < cfg_.layers; ++l)
    for (const Buffer* b : {&s.kc[size_t(l)], &s.vc[size_t(l)]}) out.push_back({*b, cfg_.hkv, size_t(cfg_.hd) * 2});
}

// ---- encoders ----
void Llama::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out) {
  const uint32_t D = uint32_t(cfg_.hidden);
  const kp::Embed ep{D, uint32_t(embed_.gs), uint32_t(tok_off)};
  e.use(pipe(embed_.bits == 4 ? std::string("embed_q4") : "embed_f" + std::to_string(embed_.bits) + "g64"));
  e.buf(0, embed_.w->buf);
  e.buf(1, embed_.s->buf);
  e.buf(2, embed_.b->buf);
  e.buf(3, tokens);
  e.buf(4, out);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, uint32_t(R), 1, 64);
}

void Llama::enc_mv(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm, uint32_t y_ld,
                   bool gu) {
  if (segs.empty() || segs.size() > 4 || R < 1 || R > kMaxRows || (gu && segs.size() != 2)) fatal("Llama: bad matvec");
  const int bits = segs[0].M->bits;
  const uint32_t K = segs[0].M->K;
  for (const Seg& s : segs)  // 4/8-bit group 64, or one 6-bit matrix (the LM head, qmvf_b6)
    if (s.M->bits != bits || s.M->K != K || (bits != 4 && bits != 8 && !(bits == 6 && segs.size() == 1 && !gu)) ||
        s.M->gs != 64)
      fatal("Llama: matvec segments in another format than 4/8-bit group 64");
  // one threadgroup per token row and 8 weight rows (2 simdgroups x 4 rows)
  constexpr int rows = 4, nsg = 2;
  char name[64];
  snprintf(name, sizeof name, "qmvf%s_b%d_t1_r%d%s", gu ? "_gu" : "", bits, rows, norm ? "_norm" : "");
  e.use(pipe(name));
  kp::Qmvf p{};
  p.K = K;
  p.R = uint32_t(R);
  p.x_ld = K;
  p.nseg = uint32_t(segs.size());
  p.eps = cfg_.eps;
  p.write_xn = 0;
  const uint32_t per_tg = uint32_t(rows * nsg);
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = s.M->N;
      p.y_ld[i] = y_ld ? y_ld : s.M->N;
      p.add[i] = s.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      if (!gu || i == 0) tg += (s.M->N + per_tg - 1) / per_tg;
    }
    e.buf(int(i * 4 + 0), s.M->w->buf);
    e.buf(int(i * 4 + 1), s.M->s->buf);
    e.buf(int(i * 4 + 2), s.M->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  if (gu) tg = p.tg_begin[1];
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, x);
  e.dispatch(uint32_t(R), tg, 1, 32 * uint32_t(nsg));
}

void Llama::enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T) {
  const kp::Rows p{uint32_t(cfg_.hidden), norm ? 1u : 0u, cfg_.eps};
  e.use(pipe("rows_to_half"));
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 256);
}

void Llama::enc_gemm(Encoder& e, const LlamaMat& W, const Buffer& xh, const Buffer& y, int M, bool add, uint32_t y_ld) {
  const kp::Qmm p{uint32_t(M), W.N, W.K, W.K, y_ld ? y_ld : W.N, add ? 1u : 0u, uint32_t(W.gs), 0, 0};
  std::string n;
  uint32_t bm = 32, bn = 128, th = 128;
  if (W.bits == 4) {
    if (M <= 8) n = "qmr_k64_w12_m8_n64", bm = 8, bn = 64, th = 64;
    else if (M < 64 || W.N <= 64) n = "qmr_k64_w14_m16_n64", bm = 16, bn = 64;
    else if (Gpu::get().apple_family() <= 7) n = "qmr_k64_w12_m32_n64", bn = 64, th = 64;
    else n = "qmr_k32_w14_m32_n128";
  } else {  // 8-bit group 64
    if (M < 64 || W.N <= 64) n = "qmr8_k32_w14_m16_n64", bm = 16, bn = 64;
    else n = "qmr8_k32_w14_m32_n128";
  }
  e.use(pipe(n));
  e.buf(0, xh);
  e.buf(1, W.w->buf);
  e.buf(2, W.s->buf);
  e.buf(3, W.b->buf);
  e.buf(4, y);
  e.val(5, p);
  e.dispatch((W.N + bn - 1) / bn, (uint32_t(M) + bm - 1) / bm, 1, th);
}

void Llama::enc_prep(Encoder& e, const Buffer& qkv, uint32_t qkv_ld, const Buffer& q, const Buffer& kc, const Buffer& vc,
                     int R, int pos0, uint32_t rows, bool chunk) {
  const LlamaConfig& c = cfg_;
  const PrepP pp{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, 0, 1, chunk ? 1u : 0u, qkv_ld,
                 c.eps, 0};
  e.use(pipe("llama_prep"));
  e.buf(0, qkv);
  e.buf(1, q);
  e.buf(2, inv_freq_);
  e.buf(3, kc);
  e.buf(4, vc);
  e.val(5, pp);
  e.dispatch(uint32_t(c.hq + 2 * c.hkv), uint32_t(R), 1, 128);
}

// One layer of R <= kMaxRows rows (decode, short prompts): x_ in place
void Llama::enc_layer_rows(Encoder& e, Seq& s, int li, int pos0, int R) {
  const LlamaConfig& c = cfg_;
  const LlamaLayer& L = layers_[size_t(li)];
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd), QKV = QD + 2 * KD;
  if (!s.kc[size_t(li)]) fatal("Llama: layer %d's cache is not allocated", li);
  if (pos0 + R > s.cap) fatal("context full (%d)", s.cap);
  const uint32_t rows = uint32_t(s.cap);
  // q | k | v of the normalized rows, RoPE, k / v into the cache
  enc_mv(e, {{&L.q, qkv_}, {&L.k, qkv_.view(size_t(QD) * 4, qkv_.size - size_t(QD) * 4)},
             {&L.v, qkv_.view(size_t(QD + KD) * 4, qkv_.size - size_t(QD + KD) * 4)}},
         x_, R, L.in_norm, QKV);
  enc_prep(e, qkv_, QKV, q_, s.kc[size_t(li)], s.vc[size_t(li)], R, pos0, rows, false);
  // keys [0, pos0 + R) in chunks: one per GPU core and KV head, at least 64 keys, at most 64 chunks (part_o_'s rows)
  const uint32_t Lk = uint32_t(pos0 + R);
  uint32_t chunk = std::max<uint32_t>(64, (Lk + gpu_cores() - 1) / gpu_cores());
  chunk = std::max(chunk, (Lk + 63) / 64);
  chunk = (chunk + 31) / 32 * 32;
  const uint32_t nch = (Lk + chunk - 1) / chunk;
  const AttnP ap{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, 0, 0, 0,
                 chunk, nch, 1.0f / std::sqrt(float(c.hd))};
  e.use(pipe(small_tiles() ? "flint_attn_partial_k16" : "flint_attn_partial"));
  e.buf(0, q_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, part_o_);
  e.buf(4, part_ml_);
  e.val(5, ap);
  e.buf(6, part_ml_);  // (no tree)
  e.dispatch(nch, uint32_t(c.hkv), uint32_t(R), 32 * uint32_t(c.hq / c.hkv));
  e.use(pipe("flint_attn_reduce"));
  e.buf(0, part_o_);
  e.buf(1, part_ml_);
  e.buf(2, sinks_);
  e.buf(3, ao_);
  e.val(4, ap);
  e.dispatch(uint32_t(c.hq), uint32_t(R), 1, uint32_t(c.hd));
  enc_mv(e, {{&L.o, x_, true}}, ao_, R, nullptr);
  // MLP: silu(gate) up of the normalized rows, then down into the residual
  enc_mv(e, {{&L.mg, mh_}, {&L.mu, mh_}}, x_, R, L.post_norm, 0, true);
  enc_mv(e, {{&L.md, x_, true}}, mh_, R, nullptr);
}

void Llama::enc_rows(Encoder& e, Seq& s, int pos0, int R) {
  if (R < 1 || R > kMaxRows) fatal("Llama: %d rows in one pass", R);
  enc_embed(e, s.tok, pos0, R, x_);
  for (int l = 0; l < cfg_.layers; ++l) enc_layer_rows(e, s, l, pos0, R);
}

// One layer of a prefill chunk of T rows (GEMMs on half rows): px_ in place. The chunk's keys go into the layer's cache
// after its attention, which reads them from pck_ / pcv_.
void Llama::enc_layer_prefill(Encoder& e, Seq& s, int li, int pos0, int T) {
  const LlamaConfig& c = cfg_;
  const LlamaLayer& L = layers_[size_t(li)];
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd), QKV = QD + 2 * KD, D = uint32_t(c.hidden);
  if (!s.kc[size_t(li)]) fatal("Llama: layer %d's cache is not allocated", li);
  if (pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  const uint32_t rows = uint32_t(s.cap);
  enc_rows_half(e, px_, L.in_norm, pxh_, T);
  enc_gemm(e, L.q, pxh_, pqkv_, T, false, QKV);
  enc_gemm(e, L.k, pxh_, pqkv_.view(size_t(QD) * 4, pqkv_.size - size_t(QD) * 4), T, false, QKV);
  enc_gemm(e, L.v, pxh_, pqkv_.view(size_t(QD + KD) * 4, pqkv_.size - size_t(QD + KD) * 4), T, false, QKV);
  enc_prep(e, pqkv_, QKV, pq_, pck_, pcv_, T, pos0, uint32_t(T), true);
  const AttnPfP ap{uint32_t(T), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, 0, 0,
                   1.0f / std::sqrt(float(c.hd))};
  e.use(pipe(small_tiles() ? "flint_attn_prefill_k16" : "flint_attn_prefill"));
  e.buf(0, pq_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, pck_);
  e.buf(4, pcv_);
  e.buf(5, sinks_);
  e.buf(6, pao_);
  e.val(7, ap);
  e.dispatch((uint32_t(T) + 7) / 8, uint32_t(c.hkv), 1, 32 * uint32_t(c.hq / c.hkv));
  const StoreP sp{uint32_t(T), 0, uint32_t(pos0), uint32_t(c.hkv), rows, 0};
  e.use(pipe("flint_kv_store"));
  e.buf(0, pck_);
  e.buf(1, pcv_);
  e.buf(2, s.kc[size_t(li)]);
  e.buf(3, s.vc[size_t(li)]);
  e.val(4, sp);
  e.dispatch(uint32_t(T), 2 * uint32_t(c.hkv), 1, 32);
  enc_gemm(e, L.o, pao_, px_, T, true);
  // MLP: gate and up (fp32), silu(gate) up into half rows (a row whose largest |h| passes 2^15 stored scaled down, its
  // scale in pscale_), down, then the residual += scale x down
  enc_rows_half(e, px_, L.post_norm, pxh_, T);
  enc_gemm(e, L.mg, pxh_, phg_, T, false);
  enc_gemm(e, L.mu, pxh_, phu_, T, false);
  const SwigluP swp{L.mg.N, 2, 0};
  e.use(pipe("flint_swiglu_scaled"));
  e.buf(0, phg_);
  e.buf(1, phu_);
  e.buf(2, phh_);
  e.buf(3, pscale_);  // (routing weights: not read in mode 2)
  e.buf(4, pscale_);
  e.val(5, swp);
  e.buf(6, pscale_);  // (sorted rows: not read in mode 2)
  e.dispatch(uint32_t(T), 1, 1, 256);
  enc_gemm(e, L.md, phh_, ys_, T, false);
  e.use(pipe("flint_add_scaled"));
  e.buf(0, px_);
  e.buf(1, ys_);
  e.buf(2, pscale_);
  e.val(3, D);
  e.dispatch(uint32_t(T), 1, 1, 256);
}

void Llama::enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T) {
  if (T > tmax_) fatal("Llama: prefill of %d rows > chunk %d", T, tmax_);
  alloc_prefill();
  enc_embed(e, s.tok, pos0, T, px_);
  for (int l = 0; l < cfg_.layers; ++l) {
    enc_layer_prefill(e, s, l, pos0, T);
    // a command buffer per layer of a long chunk, so the desktop's own GPU work is not held back behind one long one
    if (T >= 256 && l + 1 < cfg_.layers) e.split();
  }
}

void Llama::enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld) {
  enc_mv(e, {{&head_, out}}, x, R, final_norm_, uint32_t(ld));
}

void Llama::enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                       uint32_t step) {
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const kp::TopkStage ts{uint32_t(cfg_.vocab), K, 2048};
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  e.use(pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Llama::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  enc_rows(e, s, pos, 1);
  if (!sample) return;
  enc_head(e, x_, 1, logits_, size_t(cfg_.vocab));
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos + 1, sp, step);
}

void Llama::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                           Checkpoint* snap, int snap_row) {
  (void)snap;
  (void)snap_row;
  if (T > tmax_) fatal("Llama: prefill of %d rows > chunk %d", T, tmax_);
  Buffer last;
  const size_t row = size_t(cfg_.hidden) * 4;
  if (T <= kMaxRows) {  // the decode rows' kernels
    enc_rows(e, s, pos0, T);
    last = x_.view(size_t(T - 1) * row, row);
  } else {
    enc_prefill_rows(e, s, pos0, T);
    last = px_.view(size_t(T - 1) * row, row);
  }
  if (!sample) return;
  enc_head(e, last, 1, logits_, size_t(cfg_.vocab));
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  enc_sample(e, logits_, s.tok, pos0 + T, sp, step);
}

void Llama::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  const size_t row = size_t(cfg_.hidden) * 4, V = size_t(cfg_.vocab);
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T);
    enc_head(e, x_, T, out, V);
    return;
  }
  enc_prefill_rows(e, s, pos0, T);
  for (int r0 = 0; r0 < T; r0 += kMaxRows) {
    const int R = std::min(kMaxRows, T - r0);
    enc_head(e, px_.view(size_t(r0) * row, size_t(R) * row), R, out.view(size_t(r0) * V * 4, size_t(R) * V * 4), V);
  }
}

}  // namespace ni
