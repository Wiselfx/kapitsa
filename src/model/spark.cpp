#include "model/spark.h"

#include "loader/safetensors.h"
#include "metal/gpu.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>

namespace ni {

extern const char* const kTensorKernelSource;
bool qmt_on();  // qwen35.h: Metal 4 tensor_ops GEMMs available (same gate Qwen35 uses)
using json = nlohmann::json;

// ---------------------------------------------------------------------------------------------------------------
// formats / config
// ---------------------------------------------------------------------------------------------------------------

std::string SparkLinear::fmt() const {
  // Exactly wfmt.metal's name suffix (QLinear::fmt in qwen35.cpp): "f16" for bf16, else "f{bits}g{gs}".
  return dense() ? std::string("f16") : "f" + std::to_string(bits) + "g" + std::to_string(gs);
}

std::string SparkLinear::name() const {
  return dense() ? std::string("bf16") : std::to_string(bits) + "-bit group " + std::to_string(gs);
}

namespace {

bool format_supported(int bits, int gs) {
  if (bits == 16) return true;
  if (gs == 64) return bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
  // Spark-MLX-LLM's default Spark conversion: INT4 group 32 for QKV/MLP, INT8 group 32 for
  // out_proj and lm_head. f8g32's kernels are format-generic (qmvg/qrf/qmmsf), not qmv2_b8.
  return gs == 32 && (bits == 4 || bits == 8);
}

// config.json's "quantization": a default {bits, group_size} plus per-module overrides, as mlx-lm writes it (OptiQ
// packs mix 8-bit attention projections with 4-bit MLP rows).
struct QuantCfg {
  json q = json::object();
  int bits = 16, gs = 64;
  std::string mode = "affine";

  void read(const json& root) {
    if (!root.contains("quantization") || !root["quantization"].is_object()) return;
    q = root["quantization"];
    auto whole = [](const json& o, const char* k, int d) {
      return o.contains(k) && o[k].is_number_integer() ? o[k].get<int>() : d;
    };
    bits = whole(q, "bits", 4);
    gs = whole(q, "group_size", 64);
    if (q.contains("mode") && q["mode"].is_string()) mode = q["mode"].get<std::string>();
  }

  // the {bits, group_size} of one module, by its full module path ("model.layers.3.mlp.gate_proj")
  void of(const std::string& full, int& b, int& g) const {
    b = bits, g = gs;
    if (!q.is_object()) b = 16;  // no quantization block: unquantized
    for (const auto& [key, v] : q.items()) {
      if (!v.is_object() || key == "mode" || key == "bits" || key == "group_size") continue;
      if (full == key || full.rfind(key + ".", 0) == 0) {
        if (v.contains("bits") && v["bits"].is_number_integer()) b = v["bits"].get<int>();
        if (v.contains("group_size") && v["group_size"].is_number_integer()) g = v["group_size"].get<int>();
        return;
      }
    }
  }
};

// Whether a checkpoint's RMSNorm weights are stored zero-centred (HF layout, applied as 1 + w) rather than around 1
// (MLX conversions). Spark's own Hub checkpoints are HF-format bf16 and zero-centred; reading them as MLX norms would
// feed every layer a weight near 0 and the logits would collapse. Decided from the values, as Qwen35 does: an MLX
// norm sits near 1 and is almost all positive, an HF one sits near 0 and has negatives.
bool zero_centred(const Tensor& t) {
  const size_t n = size_t(t.numel());
  if (!n || !t.buf) return false;
  // Qwen35's zero_centred_norm refuses anything wider than a norm; the same guard applies here, so a mis-keyed wide
  // tensor can never flip the whole checkpoint's norm layout.
  if (n > (1u << 16)) return false;
  int neg = 0;
  double sum = 0;
  if (t.dtype == DType::BF16) {
    const uint16_t* p = t.buf.as<uint16_t>();
    for (size_t i = 0; i < n; ++i) {
      const uint32_t u = uint32_t(p[i]) << 16;
      float f;
      std::memcpy(&f, &u, sizeof f);
      if (f < 0) ++neg;
      sum += f;
    }
  } else if (t.dtype == DType::F32) {
    const float* p = t.buf.as<float>();
    for (size_t i = 0; i < n; ++i) {
      if (p[i] < 0) ++neg;
      sum += p[i];
    }
  } else {
    return false;
  }
  return neg > 0 && sum / double(n) < 0.5;
}

// GPU cores (Gpu::core_count; NI_GPU_CORES overrides), read once. The dispatch heuristics here mirror upstream's
// (qwen35.cpp:gpu_cores) so a 20-core M4 Pro and a 40-core M4 Max split the same work the same way.
uint32_t gpu_cores() {
  static const uint32_t n = uint32_t(std::max(1, Gpu::get().core_count()));
  return n;
}

// A pipeline by name, cached. Spark's own kernels live in the "spark" extra library (src/kernels/spark/), compiled on
// first use like Dory's and the tensor library.
Pipeline pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) {
    if (n.rfind("spark_", 0) == 0) Gpu::get().compile_extra("spark", kSparkKernelSource);
    // The prefill GEMM and attention kernels on the M5's matrix units live in the tensor library; like
    // Qwen35::pipe, compile it on first use so a process that never loaded another model still finds them.
    else if (n.rfind("qmt", 0) == 0 || n.rfind("qattn_t", 0) == 0)
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
    it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  }
  return it->second;
}

}  // namespace

// A float tensor as bf16 (t itself when it is bf16, else an owned bf16 copy); `offset` is added first — the
// zero-centred RMSNorm weights of a Hugging Face checkpoint are applied as 1 + w. Mirrors Qwen35::bf16_copy, so both
// families convert a value to the same bf16 result (rounded once). Also used for Spark-MLX-LLM's F32 lm_head
// scales/biases: every kernel takes bf16 coefficients, and the conversion is done once at load rather than per token.
const Tensor* Spark::bf16_copy(const Tensor* t, float offset) {
  if (!t) return t;
  if (t->dtype == DType::BF16 && offset == 0.0f) return t;
  if (t->dtype != DType::BF16 && t->dtype != DType::F32 && t->dtype != DType::F16)
    fatal("tensor: unsupported dtype %d for a bf16 conversion", int(t->dtype));
  const size_t n = size_t(t->numel());
  Buffer b = Gpu::get().alloc(n * 2 + 16, "spark_bf16_copy", /*tracked=*/false);
  uint16_t* d = b.as<uint16_t>();
  for (size_t i = 0; i < n; ++i) {
    const float v = t->dtype == DType::F32 ? t->buf.as<float>()[i]
                    : t->dtype == DType::F16 ? float(t->buf.as<_Float16>()[i])
                                             : bf16_to_f32(t->buf.as<uint16_t>()[i]);
    d[i] = f32_to_bf16(v + offset);
  }
  owned_bufs_.push_back(b);
  Tensor c = *t;
  c.buf = b.view(0, n * 2);
  c.dtype = DType::BF16;
  owned_.push_back(c);
  return &owned_.back();
}


// ---------------------------------------------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------------------------------------------

bool Spark::is_model(const std::string& dir) {
  auto root = json_file(dir + "/config.json");
  return root && spark_family(*root);
}

SparkConfig Spark::read_config(const std::string& dir) {
  auto root = json_file(dir + "/config.json");
  if (!root) fatal("%s: no config.json", dir.c_str());
  const std::string bad = spark_refusal(*root, dir);
  if (!bad.empty()) fatal("%s", bad.c_str());
  return spark_read_config(*root);
}

// ---------------------------------------------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------------------------------------------

Spark::Spark(const std::string& path, int prefill_chunk) : path_(path), tmax_(std::max(1, prefill_chunk)) {
  if (!is_model(path_))
    fatal("%s: not a Spark checkpoint folder (config.json's model_type is not spark2_5)", path_.c_str());
  load(tmax_);
}

Spark::~Spark() = default;

void Spark::load(int chunk) {
  (void)chunk;
  auto root = json_file(path_ + "/config.json");
  if (!root) fatal("%s: no config.json", path_.c_str());
  const std::string bad = spark_refusal(*root, path_);
  if (!bad.empty()) fatal("%s", bad.c_str());
  cfg_ = spark_read_config(*root);
  ring_rows_ = spark_ring_rows(cfg_, tmax_);
  logi("Spark: %s; sliding rings %d slots (window %d + prefill chunk %d)", cfg_.formats().c_str(), ring_rows_,
       cfg_.window, tmax_);

  QuantCfg quant;
  quant.read(*root);
  if (quant.mode != "affine") fatal("%s: MLX \"%s\" quantization is not supported", path_.c_str(), quant.mode.c_str());

  ws_ = std::make_unique<WeightStore>();
  ws_->open_dir(path_);

  auto qlin = [&](const std::string& name, bool required = true) {
    SparkLinear L;
    L.w = ws_->find(name + ".weight");
    if (!L.w) {
      if (required) fatal("missing %s.weight", name.c_str());
      return L;
    }
    L.s = ws_->find(name + ".scales");
    L.b = ws_->find(name + ".biases");
    if (!L.s && !L.b) {  // unquantized
      if (L.w->shape.size() != 2) fatal("%s.weight: expected a 2-D matrix", name.c_str());
      if (L.w->dtype != DType::BF16) fatal("%s.weight: weights are not bf16 and carry no affine scales", name.c_str());
      L.bits = 16;
      L.gs = 64;
      L.N = uint32_t(L.w->shape[0]);
      L.K = uint32_t(L.w->shape[1]);
      L.s = L.b = L.w;
      return L;
    }
    if (!L.s || !L.b) fatal("%s: scales without biases (not an MLX affine quantization)", name.c_str());
    int bits = 0, gs = 0;
    quant.of(name, bits, gs);
    if (!format_supported(bits, gs))
      fatal("%s: %d-bit group %d weights are not supported (bf16, 3/4/5/6/8-bit group 64, 4/8-bit group 32)", name.c_str(),
            bits, gs);
    // Spark-MLX-LLM's converted lm_head stores F32 scales and biases (the reference's W8A16 per-channel
    // path). Every kernel here takes bf16 coefficients, so round once at load. Other converted modules use
    // BF16 and are returned unchanged. This must happen before the L.s/L.b shape checks below.
    L.s = bf16_copy(L.s, 0.0f);
    L.b = bf16_copy(L.b, 0.0f);
    L.bits = bits;
    L.gs = gs;
    L.K = uint32_t(L.w->shape.back() * 32 / bits);
    L.N = uint32_t(L.w->shape[L.w->shape.size() - 2]);
    if (L.K % uint32_t(gs) || int64_t(L.K / uint32_t(gs)) != L.s->shape.back())
      fatal("%s: bits/group mismatch", name.c_str());
    return L;
  };

  embed_ = qlin("model.embedding");
  // Spark-MLX-LLM's default conversion unties the output layer (`tie_word_embeddings: false`) and writes a
  // separately quantized lm_head: INT8 group 32, F32 scales. The config is the authority, as in the
  // reference; a leftover lm_head on a tied checkpoint is ignored rather than fatal.
  head_ = cfg_.tie_embeddings ? embed_ : qlin("lm_head");
  if (!cfg_.tie_embeddings && !ws_->has("lm_head.weight"))
    fatal("%s: tie_word_embeddings is false but lm_head.weight is missing", path_.c_str());
  final_norm_ = &ws_->get("model.norm.weight");

  const Tensor* n0 = ws_->find("model.layers.0.input_layernorm.weight");
  if (n0 && zero_centred(*n0)) cfg_.hf_norms = true;
  if (cfg_.hf_norms) logi("Spark: RMSNorm weights read as 1 + w (Hugging Face layout)");

  layers_.resize(size_t(cfg_.layers));
  for (int i = 0; i < cfg_.layers; ++i) {
    SparkLayer& L = layers_[size_t(i)];
    const std::string p = "model.layers." + std::to_string(i) + ".";
    L.sliding = cfg_.kinds[size_t(i)] == SparkLayerKind::Sliding;
    L.in_norm = &ws_->get(p + "input_layernorm.weight");
    L.post_norm = &ws_->get(p + "post_attention_layernorm.weight");
    L.qkv = qlin(p + "self_attn.q_k_v_proj");
    L.gate = qlin(p + "self_attn.g_proj");
    L.o = qlin(p + "self_attn.out_proj");
    L.mg = qlin(p + "mlp.gate_proj");
    L.mu = qlin(p + "mlp.up_proj");
    L.md = qlin(p + "mlp.down_proj");
    L.rot = L.sliding ? cfg_.rot_swa : cfg_.rot_full;
    L.theta = L.sliding ? cfg_.theta_swa : cfg_.theta_full;
  }
  if (cfg_.hf_norms) {
    // 1 + w, rounded once to bf16 (as mlx-lm's sanitize and Qwen35's loader both do). Without this the whole network
    // reads weights centred on 0 instead of 1 and every activation collapses — the weights are not "close enough", so
    // this is a load-time conversion, not a comment.
    for (SparkLayer& L : layers_) {
      L.in_norm = bf16_copy(L.in_norm, 1.0f);
      L.post_norm = bf16_copy(L.post_norm, 1.0f);
    }
    final_norm_ = bf16_copy(final_norm_, 1.0f);
  }
  check_shapes();

  Gpu::get().make_resident(ws_->buffers());
  alloc_state();
  logi("Spark loaded: %d layers, %d sliding (window %d) / %d full", cfg_.layers, cfg_.n_swa(), cfg_.window, cfg_.n_full());
}

void Spark::check_shapes() const {
  // Both published models are head_dim 256 with G = Hq/Hkv = 4 (1.7B 8/2, 4B 16/4), and the shared decode kernels are
  // written for that shape: kHD is a compile-time 256, and attn_decode_partial puts one of the G query heads in each of
  // its four simdgroups. A future variant with a different head dim or group count needs its own attention kernel, so
  // refuse it clearly here instead of computing something wrong.
  if (cfg_.head_dim != 256) fatal("head_dim %d: the attention kernels take 256", cfg_.head_dim);
  if (cfg_.group() != 4)
    fatal("%d query heads per KV head (Hq %d / Hkv %d): the decode kernel is written for 4", cfg_.group(), cfg_.hq,
          cfg_.hkv);
  if (embed_.N != uint32_t(cfg_.vocab) || embed_.K != uint32_t(cfg_.hidden))
    fatal("model.embedding is [%u, %u], expected [%d, %d]", embed_.N, embed_.K, cfg_.vocab, cfg_.hidden);
  if (head_.N != uint32_t(cfg_.vocab) || head_.K != uint32_t(cfg_.hidden))
    fatal("%s is [%u, %u], expected [%d, %d]", cfg_.tie_embeddings ? "tied lm_head" : "lm_head", head_.N, head_.K,
          cfg_.vocab, cfg_.hidden);
  for (int i = 0; i < cfg_.layers; ++i) {
    const SparkLayer& L = layers_[size_t(i)];
    if (L.qkv.N != uint32_t(cfg_.qkv_dim()) || L.qkv.K != uint32_t(cfg_.hidden))
      fatal("layer %d: q_k_v_proj is [%u, %u], expected [%d, %d]", i, L.qkv.N, L.qkv.K, cfg_.qkv_dim(), cfg_.hidden);
    if (L.gate.N != uint32_t(cfg_.hq) || L.gate.K != uint32_t(cfg_.hidden))
      fatal("layer %d: g_proj is [%u, %u], expected [%d, %d] (one scalar per query head)", i, L.gate.N, L.gate.K,
            cfg_.hq, cfg_.hidden);
    if (L.o.N != uint32_t(cfg_.hidden) || L.o.K != uint32_t(cfg_.q_dim()))
      fatal("layer %d: out_proj is [%u, %u], expected [%d, %d]", i, L.o.N, L.o.K, cfg_.hidden, cfg_.q_dim());
    if (L.mg.N != uint32_t(cfg_.ffn) || L.mu.N != uint32_t(cfg_.ffn) || L.md.N != uint32_t(cfg_.hidden) ||
        L.md.K != uint32_t(cfg_.ffn))
      fatal("layer %d: MLP shapes do not match ffn %d / hidden %d", i, cfg_.ffn, cfg_.hidden);
    if (L.in_norm->dtype != DType::BF16 || L.post_norm->dtype != DType::BF16)
      fatal("layer %d: RMSNorm weights are not bf16", i);
  }
  if (final_norm_->dtype != DType::BF16) fatal("model.norm.weight: not bf16");
}

// ---------------------------------------------------------------------------------------------------------------
// memory
// ---------------------------------------------------------------------------------------------------------------

size_t Spark::kv_bytes_per_pos() const { return spark_kv_per_pos(cfg_); }

int Spark::cache_rows(const Seq& s, int li) const {
  // A sliding layer's cache is a ring of ring_rows_ (window + the prefill chunk, a power of two); a
  // full layer's grows with the conversation. The chunk term is required, not spare: prefill writes
  // the whole chunk before attention reads it.
  return ring_layer(li) ? ring_rows_ : s.cap;
}

std::unique_ptr<Seq> Spark::new_seq(int limit, int initial) {
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->cap = initial > 0 ? std::min(initial, limit) : std::min(limit, 1024);
  s->kc.resize(size_t(cfg_.layers));
  s->vc.resize(size_t(cfg_.layers));
  for (int i = 0; i < cfg_.layers; ++i) {
    const size_t bytes =
        size_t(cache_rows(*s, i)) * size_t(cfg_.hkv) * size_t(cfg_.head_dim) * sizeof(uint16_t);
    s->kc[size_t(i)] = Gpu::get().alloc(bytes, "spark_kc");
    s->vc[size_t(i)] = Gpu::get().alloc(bytes, "spark_vc");
  }
  s->tok = Gpu::get().alloc(size_t(limit + 8) * sizeof(int32_t), "spark_tok");
  return s;
}

bool Spark::reserve(Seq& s, int need) {
  if (need <= s.cap) return true;
  if (need > s.limit) return false;
  // Only the full-attention layers have growing caches; the sliding ones are fixed rings (`ring_layer`).
  const size_t per = size_t(cfg_.hkv) * size_t(cfg_.head_dim) * 2 * sizeof(uint16_t);
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  auto fits = [&](int c) {
    // Final size; a copy holds both the old and the new buffers for a moment (at most 256 MB, and at most the old cache).
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || Gpu::get().allocated_bytes() + size_t(c - s.cap) * per + copy <= mem_budget_;
  };
  // A quarter more than needed, so a conversation filling up does not reallocate at every step.
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;

  // The existing positions must be preserved. Dropping them silently loses the prompt mid-generation (the earlier
  // version allocated fresh buffers and copied nothing; nothing else re-reads those slots, so it produced a model that
  // had "forgotten" everything before the reallocation without any error). kv_regrow is the base library's kernel for
  // exactly this, and Qwen35::grow_kv is the pattern: allocate, copy, commit, release.
  Gpu& g = Gpu::get();
  const int old = s.cap;
  std::vector<std::pair<Buffer*, const char*>> bufs;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (ring_layer(i)) continue;
    bufs.push_back({&s.kc[size_t(i)], "spark_kc"});
    bufs.push_back({&s.vc[size_t(i)], "spark_kv"});
  }
  const size_t nb = size_t(target) * per;
  std::vector<Buffer> olds;
  std::unique_ptr<Encoder> e;
  for (auto& [b, why] : bufs) {
    Buffer nbuf = g.alloc(nb, why);
    if (old > 0 && *b) {
      if (!e) e = g.begin();
      const uint32_t n16 = uint32_t(cfg_.head_dim) * 2 / 16;  // 16-byte words per position
      // kv_regrow: p = (positions to copy in 16-byte words, source cap in words, destination cap in words), and the
      // grid is y = Hkv. Compare qwen35's enc_kv_regrow, which packs it the same way.
      const uint32_t p[4] = {uint32_t(old) * n16, uint32_t(old) * n16, uint32_t(target) * n16, 0};
      e->use(pipe("kv_regrow"));
      e->buf(0, *b);
      e->buf(1, nbuf);
      e->bytes(2, p, sizeof p);
      e->dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
    }
    olds.push_back(*b);
    *b = nbuf;
  }
  if (e) e->commit_and_wait();
  for (auto& b : olds) g.release(b);
  s.cap = target;
  return true;
}

void Spark::drop_kv(Seq& s) {
  s.kc.clear();
  s.vc.clear();
  s.cap = 0;
}

void Spark::reset(Seq& s) {
  // Nothing to clear: every pass writes the slots it reads before reading them, and a query never reads a key at a
  // position it has not written.
  (void)s;
}

Checkpoint Spark::new_checkpoint() { return Checkpoint{}; }

// A checkpoint holds copies of the sliding-window rings, as Dory's does: the full-attention layers append, so every
// position up to `pos` is still in their cache and a truncation to a shorter sequence reads the same bytes, while a
// ring with an older sequence in it has been overwritten and must be saved. Without this the engine could not cut a
// prefill chunk at a turn boundary (it fatal'd with "Spark has no snapshot"), which is what a second message in a
// conversation needs.
void Spark::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  c.pos = -1;
  if (c.rec.size() != size_t(2 * cfg_.layers)) c.rec.assign(size_t(2 * cfg_.layers), Buffer{});
  for (int i = 0; i < cfg_.layers; ++i) {
    if (!ring_layer(i)) {
      // no ring: nothing to save, and anything left from an earlier save at another position goes
      for (int kv = 0; kv < 2; ++kv) {
        Buffer& dst = c.rec[2 * size_t(i) + size_t(kv)];
        if (dst) Gpu::get().release(dst), dst = Buffer{};
      }
      continue;
    }
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = kv ? s.vc[size_t(i)] : s.kc[size_t(i)];
      Buffer& dst = c.rec[2 * size_t(i) + size_t(kv)];
      if (!src) continue;
      if (!dst) {
        if (mem_budget_ && Gpu::get().allocated_bytes() + src.size > mem_budget_) return;  // no room: no checkpoint
        dst = Gpu::get().alloc(src.size, "spark_ring_ckpt");
      }
      const uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
      e.use(pipe("spark_copy4"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
  }
  c.pos = pos;
}

void Spark::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  // Only the rings: a full layer's cache still holds everything up to the checkpoint's position.
  for (size_t i = 0; i < size_t(cfg_.layers) && 2 * i + 1 < c.rec.size(); ++i) {
    if (!ring_layer(int(i))) continue;
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = c.rec[2 * i + size_t(kv)];
      const Buffer& dst = kv ? s.vc[i] : s.kc[i];
      if (!src || !dst) continue;
      const uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
      e.use(pipe("spark_copy4"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
  }
}

std::string Spark::disk_format() const {
  char b[160];
  snprintf(b, sizeof b, "spark h%d l%d hq%d hkv%d d%d w%d r%d", cfg_.hidden, cfg_.layers, cfg_.hq, cfg_.hkv,
           cfg_.head_dim, cfg_.window, ring_rows_);
  return b;
}

// The full-attention caches as rows: they are linear, so position p is row p, and their stride is s.cap.
void Spark::disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const {
  (void)mtp;  // Spark has no MTP head
  for (int i = 0; i < cfg_.layers; ++i) {
    if (ring_layer(i)) continue;
    for (const Buffer* b : {&s.kc[size_t(i)], &s.vc[size_t(i)]})
      out.push_back({*b, cfg_.hkv, size_t(cfg_.head_dim) * 2});
  }
}

// The sliding-window rings, whole, in cache order: k then v per layer, the same order and sizes for a checkpoint as
// for the live state (so a stored state is found again at the position it was saved at).
void Spark::disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const {
  for (int i = 0; i < cfg_.layers; ++i) {
    if (!ring_layer(i)) continue;
    for (size_t kv = 0; kv < 2; ++kv) {
      const size_t idx = 2 * size_t(i) + kv;
      out.push_back(!c ? (kv ? s.vc[size_t(i)] : s.kc[size_t(i)])
                       : idx < c->rec.size() ? c->rec[idx] : Buffer{});
    }
  }
}

std::string Spark::formats_summary() const { return cfg_.formats(); }

// ---------------------------------------------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------------------------------------------

void Spark::alloc_state() {
  auto A = [](size_t bytes, const char* why) { return Gpu::get().alloc(bytes, why); };
  const size_t H = size_t(cfg_.hidden), Q = size_t(cfg_.q_dim()), KV = size_t(cfg_.kv_dim());
  x_ = A(H * sizeof(float), "spark_x");
  // Four distinct buffers, because each step's output is another step's input and none of them can overlap:
  //   stg_  the fused q_k_v_proj matvec's three row slices: q at +0, k at +q_dim, v at +q_dim + kv_dim (qkv_dim)
  //   qr_   the roped dense q, [Hq][D], what attn_decode_partial multiplies
  //   qg_   the interleaved [Hq][2][D] query|gate, what attn_decode_reduce takes the gate from (2*q_dim)
  // stg_ and qr_ must differ (RoPE reads a partner lane, so rotating in place races); qr_ and qg_ must differ (the
  // interleaved layout is twice as wide, so an in-place expansion writes what later heads still have to read).
  stg_ = A(size_t(cfg_.qkv_dim()) * sizeof(float), "spark_stg");
  kr_ = A(KV * sizeof(float), "spark_kr");
  vr_ = A(KV * sizeof(float), "spark_vr");
  qr_ = A(Q * sizeof(float), "spark_qr");
  qg_ = A(size_t(2 * cfg_.q_dim()) * sizeof(float), "spark_qg");
  ao_ = A(Q * sizeof(float), "spark_ao");
  gate_ = A(size_t(cfg_.hq) * sizeof(float), "spark_gate");
  gu_ = A(size_t(cfg_.ffn) * sizeof(float) * 2, "spark_gu");
  mh_ = A(size_t(cfg_.ffn) * sizeof(float), "spark_mh");
  logits_ = A(size_t(cfg_.vocab) * sizeof(float), "spark_logits");
  // Indexed by chunk, and enc_attn's nchunks is bounded by this same count (it sizes a chunk to at least span/cores),
  // so the buffers must be sized by the number that splits the work -- not a literal 32 beside a dynamic core count.
  const uint32_t cores = gpu_cores();
  part_o_ = A(size_t(cfg_.hq) * cores * size_t(cfg_.head_dim) * sizeof(float), "spark_part_o");
  part_ml_ = A(size_t(cfg_.hq) * cores * 2 * sizeof(float), "spark_part_ml");
  // The batched-prefill activation set, T = prefill_chunk rows wide. Sized here rather than at first use so the memory
  // plan can account for it (spark_activation_bytes) and so a model that does not fit fails at load, not mid-prompt.
  {
    const size_t T = size_t(tmax_);
    const size_t F = size_t(cfg_.ffn), Q = size_t(cfg_.q_dim()), HQ = size_t(cfg_.hq), H = size_t(cfg_.hidden);
    xb_ = A(T * H * sizeof(float), "spark_xb");          // the fp32 residual over T rows
    xh_ = A(T * H * 2, "spark_xh");                       // its normed half copy (the GEMMs' input)
    hh_ = A(T * F * 2, "spark_hh");                       // gate*up as half (down_proj's input)
    stgb_ = A(T * size_t(cfg_.qkv_dim()) * sizeof(float), "spark_stgb");
    qrb_ = A(T * Q * sizeof(float), "spark_qrb");
    qgb_ = A(T * 2 * Q * sizeof(float), "spark_qgb");
    aob_ = A(T * Q * sizeof(float), "spark_aob");
    gub_ = A(T * 2 * F * sizeof(float), "spark_gub");
    lgb_ = A(size_t(cfg_.vocab) * sizeof(float), "spark_lgb");
    (void)HQ;
  }
  cand_v_ = A(size_t(cfg_.vocab) * sizeof(float), "spark_cand_v");
  cand_i_ = A(size_t(cfg_.vocab) * sizeof(int32_t), "spark_cand_i");
  sprob_ = A(64 * sizeof(float), "spark_sprob");
}

// ---------------------------------------------------------------------------------------------------------------
// the pieces of a layer
// ---------------------------------------------------------------------------------------------------------------

// The matvec kernel for one matrix and how its result is consumed. One rule, one place: qmv2's b8 instantiations
// exist only with a folded norm, qmv2's b4 ones with every combination, qmv3 covers wide K, and qmvg (wfmt.metal)
// covers every other format. Getting a name wrong here is a fatal "no such pipeline" only at run time, so this is
// deliberately not copy-pasted per call site.
Spark::Matvec Spark::matvec_pipe(const SparkLinear& W, bool norm, bool add) {
  // The exact instantiation sets, read off the kernel sources. No _add_norm form exists anywhere, so `add` and `norm`
  // are mutually exclusive by construction (the one place both are wanted -- out_proj's residual -- passes add only,
  // because attention's output is already the normalized value it should add).
  //
  //   qmv.metal    qmv2_b4_g64_r{1,2,4}            plain, _add, _norm        (nothing for b8 plain/_add)
  //                qmv2_b8_g64_r{1,2,4}            _norm only
  //                qmv3_b4_r{2,4}                  plain, _norm (r2 _add)
  //                qmv3_b8_r2                      plain, _add, _norm
  //   wfmt.metal   qmvg_f16g64 | f{3,4,5,6,8}g64 | f4g32, each r2 plain, r2_add, r2_norm, r4 plain, r4_norm
  //                (NI_WF_DECODE over NI_WF_NO8 / NI_WF_ALL; note r4 has no _add)
  if (add && norm) fatal("Spark: no matvec instantiation combines a residual add with a folded norm");
  const bool huge = W.N >= 65536;
  const std::string rows4 = huge ? "4" : "2";
  std::string name;
  if (W.q4() && W.gs == 64) {
    if (W.K > 4096) {
      name = "qmv3_b4_r" + (add ? std::string("2") : rows4);
    } else {
      name = "qmv2_b4_g64_r" + rows4;
    }
  } else if (W.q8() && W.gs == 64) {
    if (!norm) {
      name = "qmv3_b8_r2";  // the only 8-bit form without a folded norm
      if (add) name += "_add";
    } else if (W.K > 4096) {
      name = "qmv3_b8_r2";
    } else {
      name = "qmv2_b8_g64_r" + rows4;
    }
  } else {
    // qmvg carries bf16, the 3/5/6-bit formats, and both group-32 formats. qmv2_b4_g64/qmv3_* and
    // qmv2_b8_g64 hard-code GS == 64; sending Spark-MLX-LLM's f4g32/f8g32 there reads a scale per 64
    // values from a table indexed per 32, which is silently wrong, not a slower path.
    name = "qmvg_" + W.fmt() + "_r" + (add ? std::string("2") : rows4);
  }
  if (add) name += "_add";
  else if (norm) name += "_norm";
  const uint32_t rows = name.find("_r4") != std::string::npos ? 4u : 2u;
  const uint32_t nsg = 8;
  return Matvec{pipe(name), rows, nsg};
}

void Spark::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out) {
  const uint32_t D = uint32_t(cfg_.hidden);
  kp::Embed ep{D, uint32_t(embed_.gs), uint32_t(tok_off)};
  e.use(pipe(embed_.q4() ? std::string("embed_q4") : "embed_" + embed_.fmt()));
  e.buf(0, embed_.w->buf);
  e.buf(1, embed_.s->buf);
  e.buf(2, embed_.b->buf);
  e.buf(3, tokens);
  e.buf(4, out);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, uint32_t(R), 1, 64);
}

void Spark::enc_head(Encoder& e, const Buffer& x, const Buffer& out) {
  // the tied output layer after the final norm (the RMSNorm folded into the matvec)
  const SparkLinear& L = head_;
  const Matvec m = matvec_pipe(L, /*norm=*/true, /*add=*/false);
  const uint32_t rows = m.rows, nsg = m.nsg;
  e.use(m.p);
  kp::Qmv p{L.N, L.K, L.K, L.N, cfg_.eps};
  e.buf(0, L.w->buf);
  e.buf(1, L.s->buf);
  e.buf(2, L.b->buf);
  e.buf(3, x);
  e.buf(4, out);
  e.val(5, p);
  e.buf(6, out);
  e.buf(7, final_norm_->buf);
  e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Spark::enc_mlp(Encoder& e, const SparkLayer& L) {
  // gate and up with the post-attention norm folded in (the norm is folded into the matvec, not applied separately),
  // exact-GELU(gate) * up folded by Spark's own epilogue, then down added into x_ as the residual.
  // `in` is the matvec's input row: x_ for gate/up (which normalize it on the fly), mh_ for down_proj (the GELU's
  // output). It was hard-coded to x_ for all three, so down_proj multiplied the *residual* instead of the activation --
  // every layer's MLP was then a linear map of x with no gate or activation in it at all, which produces fluent-looking
  // nonsense.
  auto matvec = [&](const SparkLinear& W, const Buffer& in, const Buffer& y, const Tensor* norm, const Buffer* resid) {
    const Matvec m = matvec_pipe(W, norm != nullptr, resid != nullptr);
    const uint32_t rows = m.rows, nsg = m.nsg;
    e.use(m.p);
    kp::Qmv p{W.N, W.K, W.K, W.N, cfg_.eps};
    e.buf(0, W.w->buf);
    e.buf(1, W.s->buf);
    e.buf(2, W.b->buf);
    e.buf(3, in);
    e.buf(4, y);
    e.val(5, p);
    // The _norm variants of qmv2/qmv3/qmvg take the RMSNorm weight in buffer 7 and normalize x on the fly
    // (rms_ctx<1>(x, K, K, eps, lane) in qmv.metal); they do not write the normalized row anywhere. With `add` the
    // residual comes in buffer 6. So the norm is folded for gate/up (which must not modify x_) and the add for down
    // (which must), with no extra buffer and no flags.
    e.buf(6, resid ? *resid : y);
    e.buf(7, norm ? norm->buf : y);
    e.dispatch((W.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
  };

  Buffer g = gu_, u = gu_.view(size_t(cfg_.ffn) * sizeof(float), size_t(cfg_.ffn) * sizeof(float));
  matvec(L.mg, x_, g, L.post_norm, nullptr);
  matvec(L.mu, x_, u, L.post_norm, nullptr);
  {
    kp::SparkGu gp{uint32_t(cfg_.ffn), uint32_t(cfg_.ffn)};
    e.use(pipe("spark_gelu_gu"));
    e.buf(0, g);
    e.buf(1, u);
    e.buf(2, mh_);
    e.val(3, gp);
    e.dispatch((uint32_t(cfg_.ffn) + 255) / 256, 1, 1, 256);
  }
  matvec(L.md, mh_, x_, nullptr, &x_);  // down_proj(act * up), added into the residual
}

void Spark::enc_attn(Encoder& e, Seq& s, int li, int pos) {
  const SparkLayer& L = layers_[size_t(li)];
  const uint32_t D = uint32_t(cfg_.head_dim), G = uint32_t(cfg_.group());
  const int qd = cfg_.q_dim(), kd = cfg_.kv_dim();

  // q | k | v out of the fused q_k_v_proj: three row ranges of ONE matrix, sharing one normalized input. Upstream fuses
  // exactly this case into a multi-segment matvec (qmv.metal's qmv2_multi / qmv3_multi, used by Qwen35::enc_qmv_multi
  // for q/k/v), which computes the RMSNorm once and issues one dispatch instead of three. Same here; when the geometry
  // is outside what those kernels cover (a format they do not have, an N past their four segments) it falls back to
  // three ordinary matvecs, which is what this did before.
  auto rows = [&](const Buffer& y, int n, int row0) {
    const Matvec m = matvec_pipe(L.qkv, /*norm=*/true, /*add=*/false);
    const uint32_t r = m.rows, nsg = m.nsg;
    const size_t w_row = size_t(L.qkv.K) * size_t(L.qkv.bits) / 8;
    const size_t s_row = size_t(L.qkv.K) / size_t(L.qkv.gs) * 2;
    // The row offset below is a byte offset into the packed weight/scales/biases. That needs a row to be a whole
    // number of 16-byte words, so the view stays 16-byte aligned (the kernels load it as vectors). True for both
    // models and every supported format -- hidden 2048/2560 and ffn 6656/10240 at 3/4/5/6/8/16 bits, group 32/64 --
    // but asserted here rather than assumed, because a future Spark-Coder with an odd ffn would break it silently.
    if (w_row % 16 || ((L.qkv.q() ? s_row : 0) % 16))
      fatal("layer's q_k_v_proj row stride (%zu bytes of w, %zu of scales) is not 16-byte aligned: the q/k/v row "
            "slices cannot be bound as byte offsets", w_row, s_row);
    e.use(m.p);
    kp::Qmv p{uint32_t(n), L.qkv.K, L.qkv.K, uint32_t(n), cfg_.eps};
    e.buf(0, L.qkv.w->buf, size_t(row0) * w_row);
    e.buf(1, L.qkv.s->buf, size_t(row0) * s_row);
    e.buf(2, L.qkv.b->buf, size_t(row0) * s_row);
    e.buf(3, x_);
    e.buf(4, y);
    e.val(5, p);
    e.buf(6, y);
    e.buf(7, L.in_norm->buf);
    e.dispatch((uint32_t(n) + r * nsg - 1) / (r * nsg), 1, 1, 32 * nsg);
  };
  // The multi-segment kernels address the whole matrix (segment i owns tg.x in [tg_begin[i], tg_begin[i+1])), so unlike
  // the row-sliced path above they need no byte-offset views: buffers 0..2 take q_k_v_proj in full.
  const bool multi = (L.qkv.q4() || L.qkv.q8()) && L.qkv.K <= 4096;
  if (multi) {
    constexpr uint32_t rows_per = 2, nsg = 8;
    kp::QmvMulti2 p{};
    p.K = L.qkv.K;
    p.nseg = 3;
    p.eps = cfg_.eps;
    p.norm = 1;
    p.write_xn = 0;  // the normalized row is not wanted as a separate buffer (attn_prep re-derives it)
    const uint32_t segs[3] = {uint32_t(qd), uint32_t(kd), uint32_t(kd)};
    const Buffer* ys[3] = {&stg_, &kr_, &vr_};
    uint32_t tg = 0;
    for (int i = 0; i < 3; ++i) {
      p.N[i] = segs[i];
      p.bits[i] = uint32_t(L.qkv.bits);
      p.tg_begin[i] = tg;
      tg += (segs[i] + rows_per * nsg - 1) / (rows_per * nsg);
      e.buf(i * 4 + 0, L.qkv.w->buf);
      e.buf(i * 4 + 1, L.qkv.s->buf);
      e.buf(i * 4 + 2, L.qkv.b->buf);
      e.buf(i * 4 + 3, *ys[i]);
    }
    p.tg_begin[3] = tg;
    e.use(pipe(L.qkv.K > 4096 ? "qmv3_multi_r2" : "qmv2_multi_r2"));
    e.buf(16, x_);
    e.val(17, p);
    e.buf(18, L.in_norm->buf);
    e.buf(19, stg_);
    e.dispatch(tg, 1, 1, 32 * nsg);
  } else {
    rows(stg_, qd, 0);
    rows(kr_, kd, qd);
    rows(vr_, kd, qd + kd);
  }
  // the per-head gate rows, which share the same normalized input
  {
    const Matvec m = matvec_pipe(L.gate, /*norm=*/true, /*add=*/false);
    const uint32_t r = m.rows, nsg = m.nsg;
    e.use(m.p);
    kp::Qmv p{L.gate.N, L.gate.K, L.gate.K, L.gate.N, cfg_.eps};
    e.buf(0, L.gate.w->buf);
    e.buf(1, L.gate.s->buf);
    e.buf(2, L.gate.b->buf);
    e.buf(3, x_);
    e.buf(4, gate_);
    e.val(5, p);
    e.buf(6, gate_);
    e.buf(7, L.in_norm->buf);
    e.dispatch((L.gate.N + r * nsg - 1) / (r * nsg), 1, 1, 32 * nsg);
  }

  // Spark's own prologue replaces attn_prep: that kernel always applies an RMSNorm to q and k (it is the Qwen3.5
  // prologue, whose checkpoints have q_norm/k_norm), and Spark has neither -- its reference ropes the raw q_k_v_proj
  // output. See spark_attn_prep in src/kernels/spark/spark.metal for the citation.
  const bool ring = ring_layer(li);
  const int rows_n = ring ? ring_rows_ : s.cap;
  {
    kp::SparkPrep pp{};
    pp.T = 1;
    pp.Hq = uint32_t(cfg_.hq);
    pp.Hkv = uint32_t(cfg_.hkv);
    pp.D = D;
    pp.rot = uint32_t(L.rot);
    pp.pos = uint32_t(pos);
    pp.max_seq = uint32_t(rows_n);
    pp.g_ld = uint32_t(cfg_.hq);
    pp.theta = L.theta;
    pp.slot_mask = ring ? uint32_t(ring_rows_ - 1) : 0u;
    // decode: q|k|v are the three contiguous slices of stg_, one row each
    pp.q_ld = uint32_t(qd);
    pp.k_ld = uint32_t(kd);
    pp.v_ld = uint32_t(kd);
    pp.q_off = pp.k_off = pp.v_off = 0;
    e.use(pipe("spark_attn_prep"));
    e.buf(0, stg_);  // dense q at the front, as the fused matvec left it
    e.buf(1, kr_);
    e.buf(2, vr_);
    e.buf(3, gate_);
    e.buf(4, qr_);  // the roped dense q, laid out [Hq][D] for attn_decode_partial
    e.buf(5, qg_);  // the interleaved q|gate buffer the reduce reads
    e.buf(6, s.kc[size_t(li)]);
    e.buf(7, s.vc[size_t(li)]);
    e.val(8, pp);
    e.dispatch(uint32_t(cfg_.hq) + 2 * uint32_t(cfg_.hkv), 1, 1, D);
  }

  // The key range: everything the cache holds up to `pos` for a full layer; the last `window` keys for a sliding one.
  const uint32_t lo = ring && pos + 1 > cfg_.window ? uint32_t(pos + 1 - cfg_.window) : 0u;
  const uint32_t hi = uint32_t(pos + 1);
  const uint32_t span = hi - lo;
  // Split-K: about one chunk per GPU core, chunks of at least 64 positions (as qwen35.cpp:enc_attn does). A fixed 32
  // over-split a 20-core M4 Pro by 1.6x and would under-split a 40-core M4 Max.
  const uint32_t cores = gpu_cores();
  uint32_t chunk = std::max<uint32_t>(64, (span + cores - 1) / cores);
  chunk = (chunk + 15) / 16 * 16;
  const uint32_t nch = (span + chunk - 1) / chunk;
  // Once a sliding layer's ring has wrapped, the live keys sit at (pos+1-lo) & (window-1) onwards, wrapping around: not
  // contiguous from slot 0 any more. attn_decode_partial reads such a cache `wrapped` slots further round (see
  // AttnDecParams), which is the same sum over the same keys in ring order -- attention is a weighted sum, so the
  // order comes from the softmax weights, not the slots. The write side stores at `pos & (ring_rows_-1)`
  // (spark_attn_prep's slot_mask), so the ring holds at least the window's keys already; nothing to move or copy.
  //
  // ring_rows_ is a power of two (window + the prefill chunk, rounded up), so `&` is the modulus, and
  // it is a multiple of the kernel's kTile (16); a tile crosses the seam at most once.
  const uint32_t wrapped = lo > 0 ? (lo & (uint32_t(ring_rows_) - 1)) : 0u;
  // L is the number of live keys: span, which is hi for a full layer (and for a sliding one before it wraps).
  kp::AttnDec dp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), span, chunk, uint32_t(rows_n),
                 1.0f / std::sqrt(float(cfg_.head_dim)), nch, wrapped, ring ? uint32_t(ring_rows_ - 1) : 0u};
  e.use(pipe("attn_decode_partial"));
  e.buf(0, qr_);
  e.buf(1, s.kc[size_t(li)]);
  e.buf(2, s.vc[size_t(li)]);
  e.buf(3, part_o_);
  e.buf(4, part_ml_);
  e.val(5, dp);
  e.dispatch(nch, uint32_t(cfg_.hkv), 1, 32 * G);
  e.use(pipe("attn_decode_reduce"));
  e.buf(0, part_o_);
  e.buf(1, part_ml_);
  e.buf(2, qg_);  // the gate half of the interleaved buffer
  e.buf(3, ao_);
  e.val(4, dp);
  e.dispatch(uint32_t(cfg_.hq), 1, 1, D);

  // out_proj, added into x_ (the residual)
  {
    const Matvec m = matvec_pipe(L.o, /*norm=*/false, /*add=*/true);
    const uint32_t r = m.rows, nsg = m.nsg;
    e.use(m.p);
    kp::Qmv p{L.o.N, L.o.K, L.o.K, L.o.N, cfg_.eps};
    e.buf(0, L.o.w->buf);
    e.buf(1, L.o.s->buf);
    e.buf(2, L.o.b->buf);
    e.buf(3, ao_);
    e.buf(4, x_);
    e.val(5, p);
    e.buf(6, x_);
    e.buf(7, x_);
    e.dispatch((L.o.N + r * nsg - 1) / (r * nsg), 1, 1, 32 * nsg);
  }
}

void Spark::enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp,
                       uint32_t step) {
  enc_head(e, xrow, logits_);
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  kp::TopkStage ts{uint32_t(cfg_.vocab), K, 2048};
  e.use(pipe("topk_stage1"));
  e.buf(0, logits_);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::Sample sp2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, s.tok);
  e.buf(3, sprob_);
  e.val(4, sp2);
  e.dispatch(1, 1, 1, 1024);
}

// ---------------------------------------------------------------------------------------------------------------
// passes
// ---------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------------------------
// batched prefill (docs/spark/SPARK.md §14)
// ---------------------------------------------------------------------------------------------------------------
//
// The decode path streams every weight once per row, which is optimal for one row and T times too expensive for a
// prompt. These three pieces run T rows through one pass each instead:
//   rows_to_half   a row's RMSNorm and its float->half conversion, in one pass (gdn.metal; four models already share it)
//   qrf_*          the format-generic prefill GEMM (wfmt.metal), one weight stream per T rows instead of per row
//   attn_prefill11r / qattn_t*   the batched causal attention the other models use, now ring-aware for the sliding layers
// Nothing here is a new kernel: the GEMM ladder is Qwen35::enc_qrf's and the attention choice is
// Qwen35::enc_attn_prefill's, so Spark's prefill moves on kernels that already have benchmark numbers and fp64 checks
// behind them.

void Spark::enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D) {
  e.use(pipe("rows_to_half"));
  kp::Rows p{uint32_t(D), norm ? 1u : 0u, cfg_.eps};
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 256);
}

// One batched projection. The tile ladder is Qwen35::enc_qrf's exactly (qwen35.cpp:2545), which is the point: the bf16
// instantiations that matter here are the ones that already exist and are already exercised
// (qrf_f16_m32_n128_mf / qrf_f16_m16_n64 / qrf_f16_m8_n64), and the M-tiles-along-grid-x form is what keeps bf16's 4x
// larger weights from being streamed once per M tile.
void Spark::qrf_proj(Encoder& e, const SparkLinear& W, const Buffer& xh, const Buffer& y, int T, bool add) {
  // On an M5 (Apple10, macOS 26) 4-bit group 32/64 prefill runs on the matrix units (qmt_m64_n64*),
  // the same tile Qwen35 uses: 21-23 TFLOP/s against qmr's 7.4-7.6, and the generic qrf tiles are
  // slower still. Group 32 is the reference converter's default for QKV/MLP; group 64 is OptiQ's.
  // qmt's 16-code weight chunks never cross a 32-value group, so one template instantiation covers both.
  // Below 64 rows the 64-row tile wastes more than it gains; qrf's m16/m8 tiles take those.
  if (W.bits == 4 && (W.gs == 32 || W.gs == 64) && T >= 64 && W.K % 32 == 0 && qmt_on()) {
    const std::string n = W.gs == 32 ? "qmt_m64_n64_g32" : "qmt_m64_n64";
    e.use(pipe(n));
    kp::Qmm p{uint32_t(T), W.N, W.K, W.K, W.N, add ? 1u : 0u, uint32_t(W.gs), 0, 0};
    e.buf(0, xh);
    e.buf(1, W.w->buf);
    e.buf(2, W.s->buf);
    e.buf(3, W.b->buf);
    e.buf(4, y);
    e.val(5, p);
    e.dispatch((W.N + 63) / 64, (uint32_t(T) + 63) / 64, 1, 128);
    return;
  }
  // bf16's large tile has a form that runs its M tiles along grid x (_mf), so the threadgroups executing together share a
  // weight tile through the cache instead of each streaming the whole matrix -- bf16 weights are 4x the bytes of 4-bit
  // ones, which is why only the bf16 tile is built that way. Qwen35 gates that on NI_QRF16_MF (default on, qwen35.cpp);
  // Spark has no env surface of its own, so it takes the same default without a switch.
  const bool mf = W.dense();
  std::string n = "qrf_" + W.fmt();
  uint32_t bm = 32, bn = 128, th = 128;
  bool mfast = false;
  if (T <= 8) {
    n += "_m8_n64", bm = 8, bn = 64, th = 64;
  } else if (T < 64 || W.N <= 64) {
    n += "_m16_n64", bm = 16, bn = 64;
  } else if (W.dense() && mf) {
    n += "_m32_n128_mf", mfast = true;
  } else {
    n += "_m32_n128";
  }
  e.use(pipe(n));
  kp::Qmm p{uint32_t(T), W.N, W.K, W.K, W.N, add ? 1u : 0u, uint32_t(W.gs), 0, 0};
  e.buf(0, xh);
  e.buf(1, W.w->buf);
  e.buf(2, W.s->buf);
  e.buf(3, W.b->buf);
  e.buf(4, y);
  e.val(5, p);
  const uint32_t nt = (W.N + bn - 1) / bn, mt = (uint32_t(T) + bm - 1) / bm;
  if (mfast) e.dispatch(mt, nt, 1, th);
  else e.dispatch(nt, mt, 1, th);
}

// One layer's MLP over T rows. The GELU keeps Spark's own epilogue (exact erf, D4): enc_qmm_gu fuses silu *and* the
// gate*up multiply inside the GEMM, which is not Spark's activation, and folding erf into it would recompile the hot
// path of every shipping model. Two GEMMs plus an elementwise pass over T*ffn is ~0.1% of the layer.
void Spark::enc_mlp_prefill(Encoder& e, const SparkLayer& L, int T) {
  const int F = cfg_.ffn;
  enc_rows_half(e, xb_, L.post_norm, xh_, T, cfg_.hidden);
  // Two contiguous [T][F] products, not [T][2F] rows split at F: qrf writes row-major with y_ld = F.
  // The earlier "u = gub_.view(F, F)" started up at float F and let its row t overlap gate row t+1, so
  // every gate row after the first was overwritten by an up row before spark_gelu_gu read it. The
  // buffer is still 2*T*F floats; it is partitioned once, after the pass's row count is known.
  Buffer gate = gub_.view(0, size_t(T) * size_t(F) * sizeof(float));
  Buffer up = gub_.view(size_t(T) * size_t(F) * sizeof(float), size_t(T) * size_t(F) * sizeof(float));
  qrf_proj(e, L.mg, xh_, gate, T, false);  // gate -> gate[t][c]
  qrf_proj(e, L.mu, xh_, up, T, false);    // up   -> up[t][c]
  // gate*up stays per row: the epilogue reads both buffers at [t][c] and writes the fp32 product into
  // the gate half, then down_proj reads it as half (hh_), the same rounding Qwen35's swiglu_half -> qrf
  // path has.
  {
    kp::SparkGu gp{uint32_t(F), uint32_t(F)};
    e.use(pipe("spark_gelu_gu"));
    e.buf(0, gate);
    e.buf(1, up);
    e.buf(2, gate);
    e.val(3, gp);
    e.dispatch((uint32_t(F) + 255u) / 256u, uint32_t(T), 1, 256);
  }
  enc_rows_half(e, gate, nullptr, hh_, T, F);
  qrf_proj(e, L.md, hh_, xb_, T, /*add=*/true);
}

// One layer's attention over T rows. The prologue is Spark's own (spark_attn_prep: RoPE, the q|gate packing, the cache
// append) and is already batched over T; the attention itself is the shared kernel, whose params now carry the ring.
void Spark::enc_attn_prefill(Encoder& e, Seq& s, int li, int pos0, int T) {
  const SparkLayer& L = layers_[size_t(li)];
  const uint32_t D = uint32_t(cfg_.head_dim), Hq = uint32_t(cfg_.hq), Hkv = uint32_t(cfg_.hkv);
  const int qd = cfg_.q_dim(), kd = cfg_.kv_dim(), qkvd = cfg_.qkv_dim();
  const bool ring = ring_layer(li);
  // the fused q_k_v_proj (Q|K|V) and g_proj, over the shared normalized input
  enc_rows_half(e, xb_, L.in_norm, xh_, T, cfg_.hidden);
  qrf_proj(e, L.qkv, xh_, stgb_, T, false);
  qrf_proj(e, L.gate, xh_, gub_, T, false);  // the per-head scalar gate (T x Hq), packed by spark_attn_prep below
  // qrf writes a row-major [T][qkvd] matrix. spark_attn_prep therefore reads q, k and v from the same
  // buffer with per-row strides qkvd and offsets 0 / qd / qd + kd. The earlier version assumed the
  // projection had written three contiguous T-row blocks (q at [0, T*qd), then k, then v), which is not
  // what any GEMM here emits: every row after the first then read another row's q as k.
  // spark_attn_prep: RoPE (per-layer rot/theta), the interleaved q|gate, and the cache append at the ring slot
  {
    const int rows_n = ring ? ring_rows_ : s.cap;
    kp::SparkPrep pp{};
    pp.T = uint32_t(T);
    pp.Hq = Hq;
    pp.Hkv = Hkv;
    pp.D = D;
    pp.rot = uint32_t(L.rot);
    pp.pos = uint32_t(pos0);
    pp.max_seq = uint32_t(rows_n);
    pp.g_ld = Hq;
    pp.theta = L.theta;
    pp.slot_mask = ring ? uint32_t(ring_rows_ - 1) : 0u;
    pp.q_ld = pp.k_ld = pp.v_ld = uint32_t(qkvd);
    pp.q_off = 0;
    pp.k_off = uint32_t(qd);
    pp.v_off = uint32_t(qd + kd);
    e.use(pipe("spark_attn_prep"));
    e.buf(0, stgb_);  // q, k and v: one fused [T][qkvd] matrix, read with the strides above
    e.buf(1, stgb_);
    e.buf(2, stgb_);
    // buffer 3 is read (T x Hq gate scalars) while buffer 5 is written (the packed [T][Hq][2][D]), so they must be
    // different buffers -- in one dispatch an in-place pack would race the gate reads of the rows still to come.
    e.buf(3, gub_);
    e.buf(4, qrb_);
    e.buf(5, qgb_);
    e.buf(6, s.kc[size_t(li)]);
    e.buf(7, s.vc[size_t(li)]);
    e.val(8, pp);
    e.dispatch(Hq + 2 * Hkv, uint32_t(T), 1, D);
  }
  // The attention. Window/slot_mask are the ring (0 / 0 for a full-attention layer, which is also what every other
  // model passes, so the shared kernel reads exactly as before for them).
  {
    const int rows_n = ring ? ring_rows_ : s.cap;
    kp::AttnPref fp{uint32_t(T), Hq, Hkv, uint32_t(pos0), uint32_t(rows_n), 1.0f / std::sqrt(float(cfg_.head_dim)),
                    ring ? uint32_t(cfg_.window) : 0u, ring ? uint32_t(ring_rows_ - 1) : 0u};
    // On the M5, G = 4 runs qattn_t_g4d2s8: 16 tokens x one KV head's 4 heads per threadgroup, the head
    // dim split over a pair of simdgroups. Qwen35's measurement of exactly these heads (T 2048 against
    // 4096/8192/16384 keys): 3.09 / 7.34 / 15.95 ms against attn_prefill11r's 5.15 / 11.48 / 24.21.
    // NI_ATTN (any value) forces 11r, as Qwen35's does; NI_QATTN=0 and NI_QATTN_G4D2=0 turn it off.
    // The ring parameters are the same AttnPref layout both kernels share.
    auto env = [](const char* n, int def) {
      const char* v = getenv(n);
      return v && *v ? atoi(v) : def;
    };
    static const bool attn_env = getenv("NI_ATTN") && *getenv("NI_ATTN");
    const bool qattn = !attn_env && qmt_on() && cfg_.head_dim == 256 && cfg_.hq == 4 * cfg_.hkv &&
                       env("NI_QATTN", 1) != 0 && env("NI_QATTN_G4D2", 1) != 0;
    e.use(pipe(qattn ? "qattn_t_g4d2s8" : "attn_prefill11r"));
    e.buf(0, qrb_);
    e.buf(1, s.kc[size_t(li)]);
    e.buf(2, s.vc[size_t(li)]);
    e.buf(3, qgb_);
    e.buf(4, aob_);
    e.val(5, fp);
    if (qattn) e.dispatch((uint32_t(T) * 4u + 63u) / 64u, Hkv, 1, 256);  // 4 pairs x 16 rows
    else e.dispatch((uint32_t(T) + 31) / 32, Hq, 1, 256);
  }
  // out_proj, added into the fp32 residual
  enc_rows_half(e, aob_, nullptr, xh_, T, qd);
  qrf_proj(e, L.o, xh_, xb_, T, /*add=*/true);
}

void Spark::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  // `pos + 2`: this step reads position `pos` and, when it samples, writes the token at `pos + 1`, so the sequence needs
  // that slot to exist (tokens is int32 [limit + 8], but the KV caches must cover the row too). Qwen35's guard is the
  // same comparison (qwen35.cpp:2419).
  if (pos + 2 > s.cap && !reserve(s, pos + 2)) fatal("context full (%d)", s.cap);
  enc_embed(e, s.tok, pos, 1, x_);
  for (int i = 0; i < cfg_.layers; ++i) enc_attn(e, s, i, pos), enc_mlp(e, layers_[size_t(i)]);
  // The sampled token goes to pos + 1, not pos: a step at `pos` reads the token at `pos` and produces the one after it,
  // which is what the next pass embeds and what the engine reads back as s.tokens()[pos + 1]. Writing at `pos`
  // overwrote the token just embedded -- so the engine read that token back as the model's output and the next step
  // re-embedded a slot nothing had written. The symptom was perfectly repetitive output (the same token forever),
  // because after the first step the input never changed again. Qwen35::encode_step does the same pos + 1
  // (qwen35.cpp:2439); this is the difference that mattered and the reason the model appeared to "work" and then
  // emit only BOS.
  if (sample) enc_sample(e, s, x_, pos + 1, sp, step);
}

void Spark::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                           Checkpoint* snap, int snap_row) {
  if (snap && !(snap_row > 0 && snap_row < T)) fatal("prefill snapshot at row %d of %d", snap_row, T);
  // +1: a sampling pass writes the token at pos0 + T (the row after its last), the same slot requirement encode_step
  // encodes as pos + 2. Without it the write lands one past the cache the reserve guaranteed.
  if (pos0 + T + 1 > s.limit) fatal("context full (%d + %d > %d)", pos0, T, s.limit);
  if (!reserve(s, pos0 + T + 1)) fatal("context full (%d)", s.limit);
  // Batched: each weight is streamed once per pass instead of once per row. The former loop ran a row at a time through
  // the decode path, which is correct but sits on the bandwidth floor -- 3.42 GB of bf16 weights x T, so ~0.96 s even for
  // a 77-token prompt (measured 831.7 ms; docs/spark/SPARK.md §14).
  //
  // A snapshot in the middle of a pass is the one thing the batched form cannot do for free: the rows are processed
  // together, so there is no encoder point after exactly snap_row rows. Qwen35 captures mid-pass; Spark does the simple
  // thing -- the rows before snap_row go through as their own short pass, then the checkpoint, then the rest. Two passes
  // instead of one, and only when a snapshot is asked for.
  const int split = (snap && snap_row > 0 && snap_row < T) ? snap_row : -1;
  auto pass = [&](int p0, int n, bool want_sample) {
    if (n <= 0) return;
    enc_embed(e, s.tok, p0, n, xb_);  // the residual starts as the embeddings; every later write into xb_ is `add`
    for (int i = 0; i < cfg_.layers; ++i) enc_attn_prefill(e, s, i, p0, n), enc_mlp_prefill(e, layers_[size_t(i)], n);
    if (want_sample) {
      // The pass's last row produces the token after it (pos0 + n), as encode_step's does. enc_head folds
      // the final norm into its matvec and reads a fp32 row; the earlier code passed the half xh_ after
      // rows_to_half had already applied the final norm, so the head kernel read half data as floats and
      // normalized it a second time. Use the residual row exactly as encode_step does (encode_step passes
      // x_ with no separate norm pass).
      enc_sample(e, s,
                 xb_.view(size_t(n - 1) * size_t(cfg_.hidden) * sizeof(float),
                          size_t(cfg_.hidden) * sizeof(float)),
                 p0 + n, sp, step);
    }
  };
  if (split < 0) {
    pass(pos0, T, sample);
  } else {
    pass(pos0, split, false);
    save(e, s, *snap, pos0 + split);
    pass(pos0 + split, T - split, sample);
  }
}

void Spark::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  if (!reserve(s, pos0 + T)) fatal("context full (%d)", s.limit);
  for (int t = 0; t < T; ++t) {
    const int pos = pos0 + t;
    enc_embed(e, s.tok, pos, 1, x_);
    for (int i = 0; i < cfg_.layers; ++i) enc_attn(e, s, i, pos), enc_mlp(e, layers_[size_t(i)]);
    enc_head(e, x_, out.view(size_t(t) * size_t(cfg_.vocab) * sizeof(float), size_t(cfg_.vocab) * sizeof(float)));
  }
}

MemPlan Spark::plan(const std::string& path, int prefill_chunk) {
  const SparkConfig c = read_config(path);
  MemPlan m;
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = c.ffn;
  m.dense = true;
  m.ffn_ane = false;
  m.ffn_mma = false;
  m.formats = c.formats();
  m.kv_per_pos = spark_kv_per_pos(c);
  m.slot_state = spark_ring_bytes(c, prefill_chunk);
  // One row's activations, and exactly alloc_state()'s buffers. Two errors lived here, both fixed:
  //   - it multiplied one row by `prefill_chunk`, reserving ~310 MB on the 4B that nothing allocates. The number
  //     reaches the engine's fit checks and doctor's report, so Spark looked far larger than it is;
  //   - the formula then did not match alloc_state even for one row (it counted ffn twice, missed part_o_/part_ml_).
  // The arithmetic now lives in spark_activation_bytes, which alloc_state-independent test code calls directly; a test
  // that re-implements a formula cannot catch a bug in it, which is how the first version of this check passed while
  // the number was wrong.
  m.activations = spark_activation_bytes(c, prefill_chunk, std::max<uint32_t>(1, gpu_cores()));
  return m;
}

}  // namespace ni
