import sys
p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new, 1)


rep("""        for (const ojson& a : alts) n.alts.push_back(plain ? value(a, depth + 1) : value(merge(base, deref(a, depth)), depth + 1));""",
    """        for (const ojson& a : alts)
          n.alts.push_back(plain ? value(a, depth + 1) : value(merge(base, deref(a, depth)), depth + 1));""")
rep("""          (t == "integer" && (v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>()))))""",
    """          (t == "integer" &&
           (v.is_number_integer() || (v.is_number_float() && std::floor(v.get<double>()) == v.get<double>()))))""")
rep("""          if (std::find(y.begin(), y.end(), t) != y.end() || (t == "integer" && std::find(y.begin(), y.end(), "number") != y.end()))""",
    """          if (std::find(y.begin(), y.end(), t) != y.end() ||
              (t == "integer" && std::find(y.begin(), y.end(), "number") != y.end()))""")
rep("""      if (s.contains("minLength") && s["minLength"].is_number()) lo = std::max<int64_t>(0, int64_t(s["minLength"].get<double>()));
      if (s.contains("maxLength") && s["maxLength"].is_number()) hi = std::max<int64_t>(0, int64_t(s["maxLength"].get<double>()));""",
    """      if (s.contains("minLength") && s["minLength"].is_number())
        lo = std::max<int64_t>(0, int64_t(s["minLength"].get<double>()));
      if (s.contains("maxLength") && s["maxLength"].is_number())
        hi = std::max<int64_t>(0, int64_t(s["maxLength"].get<double>()));""")
rep("""          const ojson* vals_of = v.contains("const") ? nullptr : v.contains("enum") && v["enum"].is_array() ? &v["enum"] : nullptr;""",
    """          const ojson* vals_of =
              v.contains("const") ? nullptr : v.contains("enum") && v["enum"].is_array() ? &v["enum"] : nullptr;""")
rep("""    if (s.contains("minItems") && s["minItems"].is_number()) n.min_items = std::max<int64_t>(0, int64_t(s["minItems"].get<double>()));
    if (s.contains("maxItems") && s["maxItems"].is_number()) n.max_items = std::max<int64_t>(0, int64_t(s["maxItems"].get<double>()));""",
    """    if (s.contains("minItems") && s["minItems"].is_number())
      n.min_items = std::max<int64_t>(0, int64_t(s["minItems"].get<double>()));
    if (s.contains("maxItems") && s["maxItems"].is_number())
      n.max_items = std::max<int64_t>(0, int64_t(s["maxItems"].get<double>()));""")
rep("""    if (!callable(o->name)) throw GrammarError("tool \\"" + o->name + "\\" has a name a call cannot write (<, > or a line break)");""",
    """    if (!callable(o->name))
      throw GrammarError("tool \\"" + o->name + "\\" has a name a call cannot write (<, > or a line break)");""")
rep("""  if (g->root >= 0 && g->nodes[size_t(g->root)].kind == NK::Never) throw GrammarError("response format: the schema allows no value");""",
    """  if (g->root >= 0 && g->nodes[size_t(g->root)].kind == NK::Never)
    throw GrammarError("response format: the schema allows no value");""")
rep("""      byte_.push_back(uint8_t(s[k])), depth_.push_back(uint8_t(std::min<size_t>(k + 1, 255))), tok_.push_back(-1), end_.push_back(0);""",
    """      byte_.push_back(uint8_t(s[k])), depth_.push_back(uint8_t(std::min<size_t>(k + 1, 255)));
      tok_.push_back(-1), end_.push_back(0);""")
open(p, "w").write(s)
print("wrapped")
