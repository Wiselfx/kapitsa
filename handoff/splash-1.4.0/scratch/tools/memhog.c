// memhog: bounded memory pressure for measurements (scratch tool, not part of Kapitsa).
//   memhog <max_gb> <hold_s> [swap_guard_mb=256] [step_mb=512] [stop_level=2]
// Allocates up to max_gb of incompressible anonymous memory in steps. It stops growing, and gives back the last step,
// when swap use has grown by more than swap_guard_mb since it started or the kernel's memory pressure level reaches
// stop_level (2 = warning, 4 = critical). It then holds until hold_s have passed since it started, its parent exits,
// or it gets SIGINT/SIGTERM, and frees everything. While holding it re-reads 1/16 of its pages per second (so they stay
// in use) and keeps checking swap: past the guard it frees everything and exits at once.
// Every line it prints starts with "memhog"; one line per step and one every 5 s while holding.
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int s) { (void)s; g_stop = 1; }

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}
// the host clock of Metal's GPUStartTime and Kapitsa's Encoder::host_ms (mach time), in ms
static double host_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1e6; }
static double swap_used_mb(void) {
  struct xsw_usage x;
  size_t len = sizeof(x);
  if (sysctlbyname("vm.swapusage", &x, &len, NULL, 0)) return -1;
  return x.xsu_used / 1048576.0;
}
static int pressure_level(void) {
  int l = 0;
  size_t len = sizeof(l);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &l, &len, NULL, 0)) return -1;
  return l;
}
static void vm(double* free_gb, double* comp_gb, double* wired_gb) {
  vm_statistics64_data_t v;
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n);
  const double pg = (double)vm_page_size / 1e9;
  *free_gb = (v.free_count + v.speculative_count) * pg;
  *comp_gb = v.compressor_page_count * pg;
  *wired_gb = v.wire_count * pg;
}

#define MAXSTEPS 256
static void* g_blk[MAXSTEPS];
static size_t g_len[MAXSTEPS];
static int g_n = 0;

static void fill(void* p, size_t len, uint64_t seed) {  // xorshift: the compressor cannot shrink it
  uint64_t* q = (uint64_t*)p;
  uint64_t x = seed | 1;
  for (size_t i = 0; i < len / 8; ++i) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    q[i] = x;
  }
}
static void free_all(void) {
  for (int i = 0; i < g_n; ++i) munmap(g_blk[i], g_len[i]);
  g_n = 0;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: memhog <max_gb> <hold_s> [swap_guard_mb=256] [step_mb=512] [stop_level=2]\n");
    return 2;
  }
  const double max_gb = atof(argv[1]), hold_s = atof(argv[2]);
  const double guard = argc > 3 ? atof(argv[3]) : 256, step_mb = argc > 4 ? atof(argv[4]) : 512;
  const int stop_level = argc > 5 ? atoi(argv[5]) : 2;
  if (max_gb <= 0 || max_gb > 16 || hold_s <= 0 || hold_s > 900 || step_mb < 64 || guard > 1024) {
    fprintf(stderr, "memhog: refusing (max_gb in (0,16], hold_s in (0,900], step_mb >= 64, guard <= 1024)\n");
    return 2;
  }
  signal(SIGINT, on_sig);
  signal(SIGTERM, on_sig);
  signal(SIGHUP, on_sig);
  const pid_t parent = getppid();
  const double t0 = now_s(), swap0 = swap_used_mb();
  double fr, cp, wd;
  vm(&fr, &cp, &wd);
  fprintf(stderr, "memhog start: host_ms %.1f, target %.1f GB, hold %.0f s, swap %.0f MB (guard +%.0f), pressure %d, "
          "free %.2f GB, compressor %.2f GB, wired %.2f GB\n", host_ms(), max_gb, hold_s, swap0, guard, pressure_level(),
          fr, cp, wd);
  const size_t step = (size_t)(step_mb * 1048576.0);
  double have = 0;
  const char* why = "target reached";
  while (have + step_mb / 1024.0 <= max_gb + 1e-9 && g_n < MAXSTEPS) {
    if (g_stop || getppid() != parent) { why = "stopped"; break; }
    void* p = mmap(NULL, step, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p == MAP_FAILED) { why = "mmap failed"; break; }
    fill(p, step, (uint64_t)(g_n + 1) * 0x9E3779B97F4A7C15ull);
    g_blk[g_n] = p;
    g_len[g_n] = step;
    ++g_n;
    have += step_mb / 1024.0;
    const double sw = swap_used_mb();
    const int lvl = pressure_level();
    vm(&fr, &cp, &wd);
    fprintf(stderr, "memhog step: %.2f GB held, swap %.0f MB (%+.0f), pressure %d, free %.2f GB, compressor %.2f GB\n",
            have, sw, sw - swap0, lvl, fr, cp);
    if (sw - swap0 > guard || lvl >= stop_level) {
      --g_n;
      munmap(g_blk[g_n], g_len[g_n]);
      have -= step_mb / 1024.0;
      why = sw - swap0 > guard ? "swap guard" : "pressure level";
      break;
    }
  }
  fprintf(stderr, "memhog holding %.2f GB (%s), host_ms %.1f\n", have, why, host_ms());
  size_t pos = 0;  // re-read cursor over all blocks, in pages
  double last_print = now_s();
  while (!g_stop && getppid() == parent && now_s() - t0 < hold_s) {
    const double sw = swap_used_mb();
    if (sw - swap0 > guard) {
      fprintf(stderr, "memhog: swap grew %.0f MB (guard %.0f): freeing everything\n", sw - swap0, guard);
      break;
    }
    // re-read 1/16 of the pages (cycling): the hold stays in use without much memory traffic
    size_t total_pages = 0;
    for (int i = 0; i < g_n; ++i) total_pages += g_len[i] / 16384;
    size_t todo = total_pages / 16 + 1;
    volatile uint64_t sink = 0;
    while (todo-- && total_pages) {
      size_t pg = pos % total_pages, i = 0;
      while (pg >= g_len[i] / 16384) pg -= g_len[i++] / 16384;
      sink += ((volatile uint64_t*)((char*)g_blk[i] + pg * 16384))[0];
      ++pos;
    }
    (void)sink;
    if (now_s() - last_print >= 5) {
      vm(&fr, &cp, &wd);
      fprintf(stderr, "memhog hold: %.0f s, %.2f GB, swap %.0f MB (%+.0f), pressure %d, free %.2f GB, compressor %.2f GB, "
              "wired %.2f GB\n", now_s() - t0, have, sw, sw - swap0, pressure_level(), fr, cp, wd);
      last_print = now_s();
    }
    usleep(1000000);
  }
  const double t_free = host_ms();
  free_all();
  vm(&fr, &cp, &wd);
  fprintf(stderr, "memhog done after %.0f s (freed at host_ms %.1f): swap %.0f MB (%+.0f), pressure %d, free %.2f GB, "
          "compressor %.2f GB\n", now_s() - t0, t_free, swap_used_mb(), swap_used_mb() - swap0, pressure_level(), fr, cp);
  return 0;
}
