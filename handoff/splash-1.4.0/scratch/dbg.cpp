#include "engine/grammar.h"
#include "tokenizer/tokenizer.h"
#include <chrono>
#include <cstdio>
using namespace ni;
static double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
int main(int argc, char** argv) {
  Tokenizer tok;
  tok.load(std::string(getenv("HOME")) + "/.cache/kapitsa/models/qwen3.5-4b-4bit/tokenizer.json");
  auto tt = std::make_shared<TokenTable>(tok, 248320, std::vector<int>{tok.find("<|im_end|>")}, tok.find("<tool_call>"),
                                         tok.find("</tool_call>"));
  const ojson tools = ojson::parse(R"([{"type":"function","function":{"name":"run_shell","parameters":{"type":"object","properties":{"command":{"type":"string"}},
      "required":["command"]}}}])");
  OutputSpec s;
  s.tools = OutputSpec::Tools::kRequired;
  auto g = output_grammar(s, tools);
  GrammarMatcher m(g, tt);
  const char* texts[] = {"\n<function=run_shell>\n<parameter=command", "\n<function=run_shell>\n<parameter=command>",
                         "\n<function=run_shell>\n<parameter=command>\n", "\n<function=run_shell>\n<parameter=command>\nls"};
  for (const char* t : texts) {
    GrammarState st = m.start();
    m.advance_tool_open(st);
    m.advance_text(st, t);
    std::vector<uint32_t> w(size_t(tt->words()));
    for (int rep = 0; rep < 3; ++rep) {
      const double t0 = now_us();
      m.fill_mask(st, w.data());
      int n = 0;
      for (uint32_t x : w) n += __builtin_popcount(x);
      printf("%-50s rep %d: %.0f us, %d allowed, frames %zu top kind %d st %d\n", t, rep, now_us() - t0, n,
             st.threads[0].size(), st.threads[0].back().kind, st.threads[0].back().st);
    }
  }
  (void)argc, (void)argv;
}
