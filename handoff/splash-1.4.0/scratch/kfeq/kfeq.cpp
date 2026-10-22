// KfBudget before/after the KfCosts extraction: the same cycles, the same numbers (printed in full)
#include "model/kflash_budget.h"
#include "util/log.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using namespace ni;
int main(int argc, char** argv) {
  setenv("KAPITSA_HOME", argv[1], 1);
  log_level() = 0;
  std::vector<int32_t> tree(kf::kWords, 0);
  std::mt19937_64 r(5);
  auto cycle = [&](KfBudget& b, int n, double ms, int a) {
    tree[kf::kObs] = n;
    const float e = 1.0f + float(n) * 0.4f;
    memcpy(&tree[kf::kObs + 1], &e, 4);
    tree[kf::kN] = 8;
    tree[kf::kA] = a;
    tree[kf::kNrec] = 3;
    for (int i = 0; i < 3; ++i) {
      tree[kf::kRecDep + i] = 1 + i;
      const float lq = -0.05f - 0.3f * float(r() % 7);
      memcpy(&tree[kf::kRecLq + i], &lq, 4);
      tree[kf::kRecHit + i] = int(r() % 3 != 0);
    }
    b.observe(tree.data(), ms);
  };
  for (int pass = 0; pass < 2; ++pass) {
    KfBudget b("eq");
    b.set_rows(0xffu, 8);
    for (int i = 0; i < 600; ++i) {
      KfParams p{};
      b.write(p);
      const int n = p.force_n ? p.force_n : 1 + int(r() % 8);
      cycle(b, n, 14.0 + 1.1 * n + 0.01 * i + 0.3 * double(r() % 5), int(r() % 4));
      if (i % 50 == 0 || i == 599) {
        printf("pass %d i %d force %d rate %.9g", pass, i, p.force_n, p.rate);
        for (int k = 1; k <= 8; ++k) printf(" c%d=%.9g", k, p.cost[k]);
        for (int d = 1; d <= 3; ++d) printf(" a%d=%.9g b%d=%.9g", d, p.alpha[d], d, p.beta[d]);
        printf("\n");
      }
    }
    printf("%s\n", b.summary().c_str());
  }
}
