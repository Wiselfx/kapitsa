// prompt_ids <model_dir> <bodies.jsonl>
// Each line: a chat request body ({"messages", "tools", "chat_template_kwargs", "reasoning_effort"}), optionally with
// "reply" (the answer's text). Prints one line of comma-separated token ids per body: the prompt as kapitsa serve renders
// and encodes it (server main.cpp parse_request + Engine::render + Tokenizer::encode), then the reply's text encoded
// after it (an approximation of the generated ids: the text re-encoded).
#include "api/common.h"
#include "engine/engine.h"
#include "tokenizer/tokenizer.h"

#include <minja/chat-template.hpp>
#include <minja/minja.hpp>

#include <cstdio>
#include <fstream>
#include <string>

using namespace ni;

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: prompt_ids <model_dir> <bodies.jsonl>\n");
    return 1;
  }
  const std::string dir = argv[1];
  Tokenizer tok;
  tok.load(dir + "/tokenizer.json");
  const std::string src = chat_template_source(dir);
  if (src.empty()) {
    fprintf(stderr, "no chat template in %s\n", dir.c_str());
    return 1;
  }
  minja::chat_template tmpl(src, "", "<|im_end|>");
  std::ifstream f(argv[2]);
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const ojson b = ojson::parse(line);
    ChatRequest r;
    r.messages = b.at("messages");
    if (b.contains("tools") && b["tools"].is_array()) r.tools = b["tools"];
    if (b.contains("chat_template_kwargs") && b["chat_template_kwargs"].is_object())
      r.template_kwargs = b["chat_template_kwargs"];
    if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string()) {
      const std::string e = b["reasoning_effort"];
      if (e == "none" || e == "minimal" || !r.template_kwargs.contains("reasoning_effort"))
        api::apply_effort(e, r.template_kwargs);
    }
    std::vector<int> ids = tok.encode(render_chat(tmpl, r, true));
    if (b.contains("reply") && b["reply"].is_string()) {
      const std::vector<int> more = tok.encode(b["reply"].get<std::string>());
      ids.insert(ids.end(), more.begin(), more.end());
    }
    for (size_t i = 0; i < ids.size(); ++i) printf(i ? ",%d" : "%d", ids[i]);
    printf("\n");
  }
  return 0;
}
