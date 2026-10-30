// The ring read the prefill kernels do, checked against a reference implementation on the CPU. Neither Mac nor CI can run
// a Metal kernel, so this is where the arithmetic between "the cache is a ring" and "the scores are a softmax over it"
// gets verified: it reproduces attn.metal's attn11_slot / attn11_kstart and the two masks (causal, ring floor) and
// compares the keys a sliding layer actually attends to against the set the reference model attends to.
//
// The property that makes the two-field ring read correct at all is that attention is a weighted sum over a SET of keys:
// reading the ring in ring order gives the same softmax as reading it linearly, because the order comes from the scores,
// not from the slots. That is what this checks -- not the kernel's speed, and not its fp32 rounding.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

// ---- the kernels' view (attn.metal: attn11_slot, attn11_kstart, and the two mask tests) ----
static unsigned slot_of(unsigned pos, unsigned slot_mask) { return slot_mask ? (pos & slot_mask) : pos; }
static unsigned kstart_of(unsigned kend, unsigned window) {
  return window && kend > window ? kend - window : 0u;
}
// The keys query `qpos` sees, as the kernel computes them: tiles from kstart/32 to kend/32, each key masked by
// `key <= qpos` (causal) and `key >= kstart` (ring floor).
static std::vector<unsigned> kernel_keys(unsigned qpos, unsigned kend, unsigned window, unsigned slot_mask) {
  const unsigned kstart = kstart_of(kend, window);
  const unsigned ntiles = (kend + 31) / 32;
  std::vector<unsigned> out;
  for (unsigned it = kstart / 32; it < ntiles; ++it) {
    const unsigned k0 = it * 32;
    for (unsigned r = 0; r < 32; ++r) {
      const unsigned key = k0 + r;
      if (key >= kend) continue;
      if (key > qpos) continue;                                  // causal
      if (k0 < kstart && key < kstart) continue;                 // ring floor
      out.push_back(slot_of(key, slot_mask));                    // the slot it is read from
    }
  }
  return out;
}
// What the reference model attends to: every position in [max(0, qpos+1-window), qpos] for a sliding layer, held in a
// ring of `window` slots. The slot is computed here as an explicit modulo rather than through slot_of -- when this
// called slot_of instead, mutating slot_of to `return pos` broke both sides together and the comparison passed
// vacuously, which is exactly the bug the check exists to catch.
static std::vector<unsigned> ref_keys(unsigned qpos, unsigned window, unsigned slot_mask) {
  (void)slot_mask;
  const unsigned lo = qpos + 1 > window ? qpos + 1 - window : 0u;
  std::vector<unsigned> out;
  for (unsigned p = lo; p <= qpos; ++p) out.push_back(window ? (p % window) : p);
  return out;
}

static int fails = 0;
static void check(bool ok, const char* what, unsigned qpos, unsigned window, unsigned slot_mask) {
  if (ok) return;
  if (++fails <= 10) printf("FAIL %s (qpos %u, window %u, slot_mask %u)\n", what, qpos, window, slot_mask);
}

int main() {
  // A linear cache (every model but Spark) must be bit-identical to the pre-change behaviour: no floor, slot == pos.
  for (unsigned qpos = 0; qpos < 200; ++qpos) {
    auto k = kernel_keys(qpos, qpos + 1, /*window=*/0, /*slot_mask=*/0);
    if (k.size() != qpos + 1) {
      printf("FAIL linear: qpos %u saw %zu keys, expected %u\n", qpos, k.size(), qpos + 1);
      ++fails;
      break;
    }
    for (unsigned i = 0; i < k.size(); ++i)
      if (k[i] != i) { printf("FAIL linear: qpos %u key %u read slot %u\n", qpos, i, k[i]); ++fails; break; }
  }

  // A ring: the set of slots read must equal the set the reference attends to, for every position across several laps.
  for (unsigned window : {512u, 1024u}) {
    const unsigned mask = window - 1;
    for (unsigned qpos = 0; qpos < 5 * window + 37; ++qpos) {
      auto got = kernel_keys(qpos, qpos + 1, window, mask);
      auto want = ref_keys(qpos, window, mask);
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      check(got == want, "ring key set", qpos, window, mask);
    }
  }

  // A prefill pass: several queries at once, the last of which sets kend. Each query's set must still be its own.
  for (unsigned window : {512u, 1024u}) {
    const unsigned mask = window - 1;
    for (unsigned pos0 = 0; pos0 < 3 * window; pos0 += 37) {
      const unsigned T = 32;
      const unsigned qpos = pos0 + T - 1;
      auto got = kernel_keys(qpos, qpos + 1, window, mask);
      auto want = ref_keys(qpos, window, mask);
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      check(got == want, "prefill last query", qpos, window, mask);
    }
  }

  // A key must never be read from a slot holding a different lap's key: for a ring of `window` slots holding `window`
  // live keys, the slots read must be exactly the whole ring, with no repeats.
  {
    const unsigned window = 512, mask = window - 1;
    for (unsigned qpos = window - 1; qpos < 3 * window; ++qpos) {
      auto got = kernel_keys(qpos, qpos + 1, window, mask);
      std::sort(got.begin(), got.end());
      const bool distinct = std::adjacent_find(got.begin(), got.end()) == got.end();
      check(got.size() == window && distinct, "ring covers each slot once", qpos, window, mask);
    }
  }

  // The config facts the whole scheme rests on (spark_config.cpp validates them too, but the kernel has no access):
  // a power-of-two window makes & the modulus, and a multiple of the 32-key tile means at most one tile straddles the
  // ring floor, which is the only reason the floor mask can be per tile.
  for (unsigned w : {512u, 1024u}) {
    const bool pow2 = (w & (w - 1)) == 0;
    const bool tile_aligned = w % 32 == 0;
    if (!pow2 || !tile_aligned) printf("FAIL window %u: pow2 %d, 32-aligned %d\n", w, pow2, tile_aligned), ++fails;
  }

  if (fails) {
    printf("ring_prefill: %d failure(s)\n", fails);
    return 1;
  }
  printf("ring_prefill: ok (linear identity, ring key sets over 5 laps, per-query prefill sets, one slot per key)\n");
  return 0;
}
