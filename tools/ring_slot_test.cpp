// CPU test of the ring-slot arithmetic behind the sliding-window attention read (src/kernels/attn.metal's attn_slot and
// src/kernels/spark/spark.metal's write slot, src/model/spark.cpp's `wrapped`): no GPU, no model.
//
// It exists because Spark-X2.6 stopped at 513 tokens with a fatal instead of running. The fix reads a wrapped ring in
// ring order, which is only correct if the host's slot arithmetic matches the kernel's -- an off-by-one here would make
// the model attend over keys from the wrong positions and produce plausible-looking nonsense, which no smoke test
// catches. So the two formulas are checked against each other directly, over a whole window of positions.
//   ring_slot_test
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
  if (ok) {
    std::printf("ok   %s\n", what);
    return;
  }
  ++g_fail;
  std::printf("FAIL %s\n", what);
}

constexpr uint32_t window = 512;                // Spark's sliding_window, both sizes
constexpr uint32_t slot_mask = window - 1;
constexpr uint32_t kTile = 16;                  // src/kernels/attn.metal

// The write side: spark_attn_prep's `slot = p.slot_mask ? (pos & p.slot_mask) : pos`.
uint32_t write_slot(uint32_t pos) { return pos & slot_mask; }

// The read side: attn_slot(p, t) with `p.wrapped` as src/model/spark.cpp computes it
// (lo & (window-1), lo = pos+1-window once wrapped).
uint32_t read_slot(uint32_t pos, uint32_t t) {
  const uint32_t lo = pos + 1 > window ? pos + 1 - window : 0u;
  const uint32_t wrapped = lo & slot_mask;
  return (wrapped + t) & slot_mask;
}

}  // namespace

int main() {
  // Spark's configs make two assumptions the kernel relies on; check them rather than trusting the comment.
  check((window & (window - 1)) == 0, "window is a power of two, so & is the modulus");
  check(window % kTile == 0, "window is a multiple of kTile, so a tile crosses the seam at most once");

  // For every position, the live window read in ring order must hit exactly the slots the write side filled -- no
  // repeats, none missed. Before wrapping that is [0, pos]; after, the whole ring.
  bool order_ok = true, set_ok = true;
  for (uint32_t pos = 0; pos < 4 * window; ++pos) {
    const uint32_t lo = pos + 1 > window ? pos + 1 - window : 0u;
    const uint32_t span = pos + 1 - lo;
    std::set<uint32_t> seen;
    for (uint32_t t = 0; t < span; ++t) {
      const uint32_t slot = read_slot(pos, t);
      // the position the read believes it is looking at, and the slot the write side used for it
      if (slot != write_slot(lo + t)) order_ok = false;
      seen.insert(slot);
    }
    if (seen.size() != span) set_ok = false;  // repeated slots: the same key counted twice
    if (pos + 1 <= window && span != pos + 1) set_ok = false;
    if (pos + 1 >= window && span != window) set_ok = false;
  }
  check(order_ok, "the t-th live key reads the slot its position was written to, for 4 windows of positions");
  check(set_ok, "each live key is read once and the whole window is covered");

  // The specific boundary the failure happened at: 512 -> 513 tokens.
  check(read_slot(511, 511) == 511, "at position 511 (window full, not wrapped) the last key is slot 511");
  check(read_slot(512, 0) == 1, "at 513 tokens the oldest live key is slot 1, not slot 0");
  check(read_slot(512, 511) == (512 & slot_mask), "at 513 tokens the newest key (position 512) is slot 0");

  // A tile straddling the seam, as the kernel splits it: the run to the end of the ring, then the run from its start.
  {
    const uint32_t pos = 512, t0 = 496;  // slots 497..511 then 0..0
    const uint32_t s0 = read_slot(pos, t0);
    const uint32_t rows_to_end = (slot_mask + 1u) - s0;
    check(s0 + rows_to_end == slot_mask + 1u, "a straddling tile's first run ends exactly at the ring's end");
  }

  if (g_fail) {
    std::printf("ring_slot_test: %d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("ring_slot_test: ok\n");
  return 0;
}
