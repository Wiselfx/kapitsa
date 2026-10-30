// CPU test of the chat-template engine (third_party/minja): no GPU, no model.
//
// It exists because a real request failed here. Spark-X2.6's template renders tool schemas with
// `tojson(separators=(',', ':'))`, and minja's tojson accepted only (value, indent), so it raised "Unknown argument
// separators for function tojson" and the request never reached the model. That is fixed in minja.hpp, and this pins
// the behaviour so it cannot regress: the compact form, the default Python spacing, the single-argument form, indenting,
// and -- the part easiest to get wrong when implementing separators by rewriting text instead of serializing -- that a
// string's own contents are never treated as separators.
//
// The second half renders Spark's published template itself (tests/fixtures/spark/chat_template.jinja, fetched from
// XHToken/Spark-X2.6-4B) with and without tools, which is the request shape that failed.
//   minja_template_test
#include "minja/chat-template.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

using nlohmann::ordered_json;
using minja::chat_template;
using minja::chat_template_inputs;

namespace {

int g_fail = 0;

void check(bool ok, const char* what, const std::string& got = "") {
  if (ok) {
    std::printf("ok   %s\n", what);
    return;
  }
  ++g_fail;
  std::printf("FAIL %s\n", what);
  if (!got.empty()) std::printf("     got: [%s]\n", got.c_str());
}

// Render `tpl` with `ctx` bound as extra template variables. No messages: these cases test the filter, not a template.
std::string render(const std::string& tpl, const ordered_json& ctx) {
  chat_template t(tpl, "", "");
  chat_template_inputs in;
  in.messages = ordered_json::array();
  in.tools = ordered_json::array();
  in.add_generation_prompt = false;
  in.extra_context = ctx;
  // A template error must be reported as a failed check, not an uncaught throw: the case this test exists for is
  // minja rejecting `separators` with "Unknown argument separators for function tojson".
  try {
    return t.apply(in);
  } catch (const std::exception& e) {
    return std::string("<<template error: ") + e.what() + ">>";
  }
}

void check_tojson_separators() {
  // The exact value shape the failing template feeds it, with a description holding both separators and an escape.
  ordered_json v = ordered_json::parse(
      R"({"type":"function","function":{"name":"get_weather","description":"Cold, then warm: 20C","note":"said \"hi\""}})");
  ordered_json ctx;
  ctx["v"] = v;

  const std::string want_compact =
      R"({"type":"function","function":{"name":"get_weather","description":"Cold, then warm: 20C","note":"said \"hi\""}})";
  check(render("{{ v | tojson(separators=(',', ':')) }}", ctx) == want_compact,
        "tojson(separators=(',', ':')) renders the compact form");

  // Python's default is (', ', ': '), and that must stay what a bare tojson does.
  const std::string got_default = render("{{ v | tojson }}", ctx);
  check(got_default.find(R"("type": "function")") != std::string::npos &&
            got_default.find(R"("Cold, then warm: 20C")") != std::string::npos,
        "tojson without separators keeps Python's spacing and string contents", got_default);

  // A single separator sets both, as in Python.
  const std::string got_one = render("{{ v | tojson(separators='|') }}", ctx);
  check(got_one.find("|\"function\"|") != std::string::npos, "a single separator sets item and key", got_one);

  // indent still indents (separators are the non-pretty form; Python ignores them when indent is set, and so does this).
  const std::string got_indent = render("{{ v | tojson(2) }}", ctx);
  check(got_indent.find("\n  \"type\": \"function\"") != std::string::npos, "tojson(2) still indents", got_indent);

  // The failure mode of a text-rewriting implementation: the separators inside a string must survive untouched.
  ordered_json s;
  s["only"] = "a, b: c";
  ordered_json sctx;
  sctx["v"] = s;
  check(render("{{ v | tojson(separators=(',', ':')) }}", sctx) == R"({"only":"a, b: c"})",
        "separators inside a string value are not changed");

  // Scalars keep their JSON spelling rather than becoming Python's (True/None).
  ordered_json b;
  b["v"] = true;
  ordered_json n;
  n["v"] = nullptr;
  ordered_json i;
  i["v"] = 42;
  check(render("{{ v | tojson }}", b) == "true", "a boolean renders as JSON true");
  check(render("{{ v | tojson }}", n) == "null", "null renders as JSON null");
  check(render("{{ v | tojson(separators=(',', ':')) }}", i) == "42", "a scalar passes through the separators form");
}

void check_spark_template(const char* path) {
  std::ifstream f(path);
  if (!f) {
    check(false, "tests/fixtures/spark/chat_template.jinja is readable (run from the repository root)");
    return;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string src = ss.str();
  chat_template tpl(src, "", "");

  ordered_json msgs = ordered_json::array();
  msgs.push_back({{"role", "user"}, {"content", "What is the weather in Paris?"}});
  chat_template_inputs in;
  in.messages = msgs;
  in.tools = ordered_json::array();
  in.add_generation_prompt = true;
  in.extra_context = {{"enable_thinking", false}};

  // Without tools -- the request that produced the error report.
  std::string plain;
  try {
    plain = tpl.apply(in);
  } catch (const std::exception& e) {
    plain = std::string("<<template error: ") + e.what() + ">>";
  }
  check(plain.find("<|User|>What is the weather in Paris?") != std::string::npos,
        "Spark template renders a user turn without tools", plain);
  check(plain.find("<|Bot|>") != std::string::npos, "Spark template ends with the assistant header", plain);

  // With tools, which is the branch that calls tojson(separators=...).
  ordered_json tools = ordered_json::array();
  tools.push_back({{"type", "function"},
                   {"function",
                    {{"name", "get_weather"},
                     {"description", "Get the current weather"},
                     {"parameters",
                      {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}, {"required", {"city"}}}}}}});
  chat_template_inputs with_tools = in;
  with_tools.tools = tools;
  std::string got;
  try {
    got = tpl.apply(with_tools);
  } catch (const std::exception& e) {
    got = std::string("<<template error: ") + e.what() + ">>";
  }
  check(got.find(R"({"name":"get_weather","description":"Get the current weather","parameters":{"type":"object")") !=
            std::string::npos,
        "Spark template renders the tool schema compactly (the failing branch)", got);
  check(got.find("## Tools") != std::string::npos, "Spark template emits the tools preamble", got);
}

}  // namespace

int main(int argc, char** argv) {
  check_tojson_separators();
  check_spark_template(argc > 1 ? argv[1] : "tests/fixtures/spark/chat_template.jinja");
  if (g_fail) {
    std::printf("minja_template_test: %d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("minja_template_test: ok\n");
  return 0;
}
