#include "splash/discover.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

namespace ni::splash {
namespace {

bool is_file(const fs::path& p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec);
}

fs::path models_dir(const std::string& home) { return fs::path(home) / "Library/Application Support/Splash/models/.resolved"; }

struct Model {
  std::string name, family, source;  // source: model.json's "model" (e.g. "nace-local/Qwen3.6-35B-A3B-4bit"), "" when absent
  fs::file_time_type written;
};

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::vector<Model> models(const std::string& home) {
  std::vector<Model> out;
  std::error_code ec;
  for (fs::directory_iterator it(models_dir(home), ec), end; !ec && it != end; it.increment(ec)) {
    const fs::path& d = it->path();
    if (!is_file(d / "tokenizer/tokenizer.json")) continue;
    Model m{d.filename().string(), "?", "", {}};
    std::error_code tec;
    m.written = fs::last_write_time(d, tec);
    std::ifstream f(d / "model.json");
    auto j = nlohmann::json::parse(f, nullptr, false);
    if (j.is_object() && j.contains("family") && j["family"].is_string()) m.family = j["family"].get<std::string>();
    if (j.is_object() && j.contains("model") && j["model"].is_string()) m.source = j["model"].get<std::string>();
    out.push_back(m);
  }
  std::sort(out.begin(), out.end(), [](const Model& a, const Model& b) { return a.name < b.name; });
  return out;
}

}  // namespace

Found find_binary(const std::string& home, const std::string& splash_home, const std::string& path_env, const std::string& last) {
  std::vector<fs::path> roots;
  if (!splash_home.empty()) roots.push_back(splash_home);
  std::stringstream ss(path_env);
  for (std::string dir; std::getline(ss, dir, ':');) {
    if (dir.empty()) continue;
    std::error_code ec;
    fs::path exe = fs::canonical(fs::path(dir) / "splash", ec);  // through a symlink to where the install is
    if (ec) continue;
    roots.push_back(exe.parent_path().parent_path());  // <root>/engine/splash or <root>/bin/splash
  }
  roots.push_back(fs::path(home) / "Applications/Splash");
  roots.push_back("/Applications/Splash");
  roots.push_back(fs::path(home) / "Splash");
  std::string tried;
  for (const fs::path& r : roots) {
    fs::path bin = r / "engine/splash";
    if (is_file(bin)) return {bin.string(), ""};
    tried += "\n  " + bin.string();
  }
  if (!last.empty()) {
    if (is_file(last)) return {last, ""};
    tried += "\n  " + last + "  (the last run's)";
  }
  return {"", "no Splash engine binary found; looked for:" + tried +
                  "\nInstall Splash, set SPLASH_HOME to its folder, or give --splash-binary <path to engine/splash>."};
}

std::vector<std::string> list_models(const std::string& home) {
  std::vector<std::string> out;
  for (const Model& m : models(home)) out.push_back(m.family + "  " + m.name);
  return out;
}

Found find_model(const std::string& home, const std::string& want, const std::string& last) {
  std::vector<Model> all = models(home);
  const std::string dir = models_dir(home).string();
  if (all.empty())
    return {"", "no Splash model found: " + dir + " has no folder with a tokenizer. Run a model in Splash once, or give --splash-model <folder>."};
  if (want.empty()) {
    if (all.size() == 1) return {dir + "/" + all[0].name, ""};
    // several: the one the last run used, else the newest resolved (Splash writes a folder when it resolves a model)
    const Model* pick = &all[0];
    std::string why = "the newest of " + std::to_string(all.size()) + " resolved models";
    for (const Model& m : all)
      if (m.written > pick->written) pick = &m;
    for (const Model& m : all)
      if (!last.empty() && last == dir + "/" + m.name) pick = &m, why = "the one the last run used";
    return {dir + "/" + pick->name, "", "Splash model: " + pick->family + " " + pick->name + " (" + why + "; --splash-model <family or folder name> picks another)"};
  }
  std::vector<const Model*> hit;
  if (want.size() >= 6)
    for (const Model& m : all)
      if (m.name.compare(0, want.size(), want) == 0) hit.push_back(&m);
  if (hit.size() == 1) return {dir + "/" + hit[0]->name, ""};
  if (hit.size() > 1) return {"", "--splash-model " + want + " matches " + std::to_string(hit.size()) + " models; give more characters"};
  // not a folder prefix: a family or source name, e.g. "flash-next", "27b", "Q2_0" (any case, anywhere in it)
  const std::string w = lower(want);
  for (const Model& m : all)
    if (lower(m.family).find(w) != std::string::npos || lower(m.source).find(w) != std::string::npos) hit.push_back(&m);
  if (hit.empty()) {
    std::string e = "--splash-model " + want + ": no resolved model in " + dir + " has a folder name starting with it or a family or source name containing it:";
    for (const Model& m : all) e += "\n  " + m.family + "  " + m.name;
    return {"", e};
  }
  // one model, or re-resolutions of the same source (Splash keeps each): the newest
  const Model* best = hit[0];
  for (const Model* m : hit) {
    if (m != best && (m->family != best->family || m->source != best->source || m->source.empty())) {
      std::string e = "--splash-model " + want + " matches " + std::to_string(hit.size()) + " different models; say which with more of its name or its folder name:";
      for (const Model* x : hit) e += "\n  " + x->family + "  " + x->name + (x->source.empty() ? "" : "  (" + x->source + ")");
      return {"", e};
    }
    if (m->written > best->written) best = m;
  }
  return {dir + "/" + best->name, ""};
}

Last read_last(const std::string& file) {
  Last out;
  std::ifstream f(file);
  auto j = nlohmann::json::parse(f, nullptr, false);
  if (!j.is_object()) return out;
  if (j.contains("binary") && j["binary"].is_string()) out.binary = j["binary"].get<std::string>();
  if (j.contains("model") && j["model"].is_string()) out.model = j["model"].get<std::string>();
  return out;
}

void write_last(const std::string& file, const Last& last) {
  if (last.binary.empty() && last.model.empty()) return;
  const Last old = read_last(file);
  if (old.binary == last.binary && old.model == last.model) return;
  std::error_code ec;
  fs::create_directories(fs::path(file).parent_path(), ec);
  const std::string tmp = file + ".tmp";
  {
    std::ofstream f(tmp);
    f << nlohmann::json{{"binary", last.binary}, {"model", last.model}}.dump(2) << "\n";
    if (!f) return;
  }
  fs::rename(tmp, file, ec);
  if (ec) fs::remove(tmp, ec);
}

Check check(const std::string& home, const std::string& splash_home, const std::string& path_env, const std::string& binary,
            const std::string& model, const Last& last) {
  Check out;
  Found b = binary.empty() ? find_binary(home, splash_home, path_env, last.binary) : Found{binary, ""};
  if (!binary.empty() && !is_file(binary)) b = {"", "--splash-binary " + binary + ": no such file"};
  out.text = b.ok() ? "Splash engine: " + b.path + "\n" : "Splash engine: NOT FOUND\n" + b.error + "\n";
  const std::vector<Model> all = models(home);
  out.text += "\nSplash models in " + models_dir(home).string() + ": " + (all.empty() ? "none" : std::to_string(all.size())) + "\n";
  for (const Model& m : all) out.text += "  " + m.family + "  " + m.name + (m.source.empty() ? "" : "  (" + m.source + ")") + "\n";
  Found m = model.empty() || (model.find('/') == std::string::npos && !fs::exists(model)) ? find_model(home, model, last.model)
                                                                                           : Found{model, ""};
  out.text += m.ok() ? "\nModel to run: " + m.path + "\n" + (m.note.empty() ? "" : m.note + "\n") : "\nModel to run: NOT DECIDED\n" + m.error + "\n";
  if (!m.ok() && !all.empty() && model.empty()) {
    std::string f = all[0].family;
    std::transform(f.begin(), f.end(), f.begin(), [](unsigned char c) { return std::tolower(c); });
    out.text += "e.g. kapitsa serve --engine splash --splash-model " + f + "\n";
  }
  out.ok = b.ok() && m.ok();
  if (out.ok) out.text += "\nReady: run the same command without --check.\n";
  return out;
}

}  // namespace ni::splash
