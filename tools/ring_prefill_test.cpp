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
// The keys query `qpos` sees, as the kernels compute them. The loop starts at the block's *first*
// query's window floor (`qfirst`, not the block's last key), and each row masks its own floor
// (`qpos + 1 - window`). The ring holds window + a whole prefill chunk, so those keys are present.
// `ring` is the cache's slot count (0 = linear), and slots are `key & (ring - 1)`.
static std::vector<unsigned> kernel_keys(unsigned qpos, unsigned kend, unsigned qfirst, unsigned window,
                                         unsigned ring) {
  const unsigned kstart = kstart_of(qfirst + 1, window);  // earliest row's live floor
  const unsigned qfloor = kstart_of(qpos + 1, window);    // this row's live floor
  const unsigned mask = ring ? ring - 1 : 0;
  const unsigned ntiles = (kend + 31) / 32;
  std::vector<unsigned> out;
  for (unsigned it = kstart / 32; it < ntiles; ++it) {
    const unsigned k0 = it * 32;
    for (unsigned r = 0; r < 32; ++r) {
      const unsigned key = k0 + r;
      if (key >= kend) continue;
      if (key > qpos) continue;                    // causal
      if (key < kstart) continue;                  // tile rounded below the block floor
      if (key < qfloor) continue;                  // this row's own window floor
      out.push_back(slot_of(key, mask));           // the slot it is read from
    }
  }
  return out;
}
// What the reference model attends to: every position in [max(0, qpos+1-window), qpos] for a sliding layer, held in a
// ring of `ring` slots. The slot is computed here as an explicit modulo rather than through slot_of -- when this
// called slot_of instead, mutating slot_of to `return pos` broke both sides together and the comparison passed
// vacuously, which is exactly the bug the check exists to catch.
static std::vector<unsigned> ref_keys(unsigned qpos, unsigned window, unsigned ring) {
  const unsigned lo = qpos + 1 > window ? qpos + 1 - window : 0u;
  std::vector<unsigned> out;
  for (unsigned p = lo; p <= qpos; ++p) out.push_back(ring ? (p % ring) : p);
  return out;
}

static int fails = 0;
static void check(bool ok, const char* what, unsigned qpos, unsigned window, unsigned ring) {
  if (ok) return;
  if (++fails <= 10) printf("FAIL %s (qpos %u, window %u, ring %u)\n", what, qpos, window, ring);
}
static unsigned ring_rows(unsigned window, unsigned chunk) {
  unsigned r = 1;
  while (r < window + chunk) r <<= 1;
  return r;
}
static void compare(unsigned qpos, unsigned kend, unsigned qfirst, unsigned window, unsigned ring, const char* what) {
  auto got = kernel_keys(qpos, kend, qfirst, window, ring);
  auto want = ref_keys(qpos, window, ring);
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  check(got == want, what, qpos, window, ring);
}

int main() {
  // A linear cache (every model but Spark) must be bit-identical to the pre-change behaviour: no floor, slot == pos.
  for (unsigned qpos = 0; qpos < 200; ++qpos) {
    auto k = kernel_keys(qpos, qpos + 1, qpos, /*window=*/0, /*ring=*/0);
    if (k.size() != qpos + 1) {
      printf("FAIL linear: qpos %u saw %zu keys, expected %u\n", qpos, k.size(), qpos + 1);
      ++fails;
      break;
    }
    for (unsigned i = 0; i < k.size(); ++i)
      if (k[i] != i) { printf("FAIL linear: qpos %u key %u read slot %u\n", qpos, i, k[i]); ++fails; break; }
  }

  constexpr unsigned chunk = 2048;  // Spark's default prefill_chunk
  // A ring: the set of slots read must equal the set the reference attends to, over several laps.
  for (unsigned window : {512u, 1024u}) {
    const unsigned ring = ring_rows(window, chunk);
    for (unsigned qpos = 0; qpos < 3 * ring + 37; ++qpos) compare(qpos, qpos + 1, qpos, window, ring, "ring key set");
  }

  // A prefill pass: the whole chunk is written before attention, so the block's floor is its FIRST
  // query's window; each row's own floor is masked separately. Check samples from start to end of T.
  for (unsigned window : {512u, 1024u}) {
    const unsigned ring = ring_rows(window, chunk);
    for (unsigned pos0 = 0; pos0 < 3 * ring; pos0 += 137) {
      const unsigned qs[] = {pos0, pos0 + 1, pos0 + 31, pos0 + 32, pos0 + chunk / 2, pos0 + chunk - 1};
      for (unsigned qpos : qs) compare(qpos, pos0 + chunk, pos0, window, ring, "prefill query");
      // The old code started at the last key's floor; show that this skips a key the first query needs.
      if (pos0 > window) {
        const unsigned old_start = kstart_of(pos0 + chunk, window);
        const unsigned need = kstart_of(pos0 + 1, window);
        check(need < old_start, "block-last floor would skip the first query's oldest key", pos0, window, ring);
      }
    }
  }

  // The boundary a ring of exactly `window` slots got wrong: the first query of a full chunk at pos0
  // = window sees keys [1, pos0], but the chunk's last key overwrote them. With window + chunk slots
  // they are present and the per-query floor keeps them.
  {
    const unsigned window = 512, ring = ring_rows(window, chunk), pos0 = window;
    for (unsigned q = pos0; q < pos0 + chunk; ++q) compare(q, pos0 + chunk, pos0, window, ring, "full-chunk query");
  }

  // A key must never be read from a slot holding a different lap's key: for a ring holding `window`
  // live keys, the slots read must be exactly the live window, with no repeats.
  {
    const unsigned window = 512, ring = ring_rows(window, chunk);
    for (unsigned qpos = window - 1; qpos < 2 * ring; qpos += 17) {
      auto got = kernel_keys(qpos, qpos + 1, qpos, window, ring);
      std::sort(got.begin(), got.end());
      const bool distinct = std::adjacent_find(got.begin(), got.end()) == got.end();
      check(got.size() == window && distinct, "ring covers each live key once", qpos, window, ring);
    }
  }

  // The config facts the whole scheme rests on (spark_config.cpp validates them too, but the kernel has no access):
  // the ring is a power of two and holds window + a whole chunk; window is a multiple of the 32-key tile.
  for (unsigned w : {512u, 1024u}) {
    const unsigned ring = ring_rows(w, chunk);
    const bool pow2 = (ring & (ring - 1)) == 0;
    const bool holds_chunk = ring >= w + chunk;
    const bool tile_aligned = w % 32 == 0;
    if (!pow2 || !holds_chunk || !tile_aligned)
      printf("FAIL window %u: ring %u pow2 %d holds-chunk %d, 32-aligned %d\n", w, ring, pow2, holds_chunk,
             tile_aligned),
          ++fails;
  }

  if (fails) {
    printf("ring_prefill: %d failure(s)\n", fails);
    return 1;
  }
  printf("ring_prefill: ok (linear identity, ring sets over laps, per-query chunk sets, one slot per key)\n");
  return 0;
}
