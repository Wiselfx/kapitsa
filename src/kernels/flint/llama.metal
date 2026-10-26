// Dense Llama (src/model/llama.cpp): the one attention kernel it does not share with Flint. It runs in Flint's library
// (this file sorts after flint.metal, whose FlintPrepP, flint_kv_index and kFD it uses) because its attention is
// Flint's: flint_attn_prefill / flint_attn_partial / flint_attn_reduce / flint_kv_store over the K / V this writes.
//  - llama_prep: RoPE on q and k (NeoX halves over all 128 dims, HF's rotate_half; no q / k norm), k and v into the
//    layer's linear cache at pos0 + t, or a prefill chunk's own K / V buffer [Hkv][R][128] (p.chunk). p.ring, p.rope,
//    p.eps and p.tree are not read: every layer rotates and none is a ring.
// grid: (Hq + 2 Hkv, R) x 128 threads. qkv: fp32 [R][qkv_ld] (q heads | k heads | v heads); q_out: fp32 [R][Hq * 128];
// inv: fp32 [64], RoPE's inverse frequencies.
[[kernel]] void llama_prep(const device float* qkv [[buffer(0)]], device float* q_out [[buffer(1)]],
                           const device float* inv [[buffer(2)]], device half* kc [[buffer(3)]],
                           device half* vc [[buffer(4)]], constant FlintPrepP& p [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  const uint h = tg.x, t = tg.y;
  const device float* src = qkv + size_t(t) * p.qkv_ld;
  const uint QD = p.Hq * kFD, KD = p.Hkv * kFD;
  if (h >= p.Hq + p.Hkv) {  // a value head: into the cache as it is
    const uint vh = h - p.Hq - p.Hkv;
    vc[flint_kv_index(p, vh, t, d)] = half(src[QD + KD + vh * kFD + d]);
    return;
  }
  const bool is_q = h < p.Hq;
  const device float* hs = src + (is_q ? 0 : QD) + (is_q ? h : h - p.Hq) * kFD;
  const float ang = float(p.pos0 + t) * inv[d % (kFD / 2)];
  const float cs = metal::precise::cos(ang), sn = metal::precise::sin(ang);
  const float x = d < kFD / 2 ? hs[d] * cs - hs[d + kFD / 2] * sn : hs[d] * cs + hs[d - kFD / 2] * sn;
  if (is_q) q_out[(size_t(t) * p.Hq + h) * kFD + d] = x;
  else kc[flint_kv_index(p, h - p.Hq, t, d)] = half(x);
}
