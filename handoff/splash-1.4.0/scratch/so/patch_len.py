import sys
p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new, 1)


# node fields
rep("""  int64_t min_items = 0, max_items = INT64_MAX;
  // Alt""", """  int64_t min_items = 0, max_items = INT64_MAX;
  // Str: the characters a string holds (minLength, maxLength; -1: no most)
  int64_t min_len = 0, max_len = -1;
  bool limited() const { return min_len > 0 || max_len >= 0; }
  // Alt""")

# the compiler: string lengths
rep("""    if (t == "string") return Grammar::kAnyStr;  // (minLength, maxLength, pattern, format: not enforced)""",
    """    if (t == "string") {  // (pattern, format: not enforced)
      int64_t lo = 0, hi = -1;
      if (s.contains("minLength") && s["minLength"].is_number()) lo = std::max<int64_t>(0, int64_t(s["minLength"].get<double>()));
      if (s.contains("maxLength") && s["maxLength"].is_number()) hi = std::max<int64_t>(0, int64_t(s["maxLength"].get<double>()));
      if (lo == 0 && hi < 0) return Grammar::kAnyStr;
      if (hi >= 0 && lo > hi) return Grammar::kNever;
      GNode n;
      n.kind = NK::Str;
      n.min_len = lo;
      n.max_len = hi;
      return add(std::move(n));
    }""")

# the frame: count a limited string's characters (one starts at each byte read at a boundary, outside an escape)
rep("""      case F_STR: {
        if (f.st == 0) {
          if (sym == '"') f.st = 1, f.a = 0;
          else dead();
          return;
        }
        if (sym > 255) {
          dead();
          return;
        }
        const int r = lex_str(int(f.a), sym);
        if (r == kLexDead) dead();
        else if (r == kLexExit) w.pop_back();
        else f.a = uint32_t(r);
        return;
      }""", """      case F_STR: {
        if (f.st == 0) {
          if (sym == '"') f.st = 1, f.a = 0, f.x = 0;
          else dead();
          return;
        }
        if (sym > 255) {
          dead();
          return;
        }
        const int r = lex_str(int(f.a), sym);
        if (r == kLexDead) {
          dead();
        } else if (r == kLexExit) {
          if (int64_t(f.x) < nd.min_len) dead();
          else w.pop_back();
        } else {
          // a limited string counts its characters (x): one starts at each byte read at a boundary (f.a 0: no
          // escape, no character open)
          if (nd.limited() && f.a == 0 && ++f.x > uint64_t(nd.max_len) && nd.max_len >= 0) {
            dead();
            return;
          }
          f.a = uint32_t(r);
        }
        return;
      }""")

# pure_text: the characters still needed and allowed
rep("""bool pure_text(const Grammar& g, const Thread& w, int* lexer, int* sigma) {
  if (w.empty()) return false;
  const Frame& f = w.back();
  switch (f.kind) {
    case F_STR:
      if (f.st != 1) return false;
      *lexer = kLexStr, *sigma = int(f.a);
      return true;""", """// (a limited string: lo the characters it still needs, hi the most it may still take; else 0 and -1)
bool pure_text(const Grammar& g, const Thread& w, int* lexer, int* sigma, int64_t* lo, int64_t* hi) {
  *lo = 0, *hi = -1;
  if (w.empty()) return false;
  const Frame& f = w.back();
  switch (f.kind) {
    case F_STR: {
      if (f.st != 1) return false;
      *lexer = kLexStr, *sigma = int(f.a);
      const GNode& nd = g.nodes[size_t(f.node)];
      if (nd.limited()) {
        *lo = std::max<int64_t>(0, nd.min_len - int64_t(f.x));
        *hi = nd.max_len >= 0 ? nd.max_len - int64_t(f.x) : -1;
      }
      return true;
    }""")

# the bulk entry: each token's characters (JSON strings)
rep("""struct TokenTable::Bulk {
  std::vector<uint32_t> bits;  // the tokens that stay inside the text: all mask words, or
  std::vector<int32_t> stay;   // their ids
  // those that leave it: the bytes after the end of the text, in a trie (preorder; node 0 the empty rest), each node's
  // tokens toks[off[i], off[i + 1])
  std::vector<uint8_t> b, d;
  std::vector<uint32_t> e, off;
  std::vector<int32_t> toks;
  int max_depth = 0;
};""", """struct TokenTable::Bulk {
  std::vector<uint32_t> bits;  // the tokens that stay inside the text: all mask words, and/or
  std::vector<int32_t> stay;   // their ids (kept for a JSON string's content, with the characters each adds)
  std::vector<uint16_t> stay_n;
  // those that leave it: the bytes after the end of the text, in a trie (preorder; node 0 the empty rest), each node's
  // tokens toks[off[i], off[i + 1]) (with the characters before the end, a JSON string's)
  std::vector<uint8_t> b, d;
  std::vector<uint32_t> e, off;
  std::vector<int32_t> toks;
  std::vector<uint16_t> toks_n;
  int max_depth = 0;
};""")

rep("""  auto b = std::make_unique<Bulk>();
  const int base = depth_[node];
  std::vector<std::pair<std::string, int>> exits;""", """  auto b = std::make_unique<Bulk>();
  const int base = depth_[node];
  struct Exit {
    std::string rest;
    int id, chars;
    bool operator<(const Exit& o) const { return rest < o.rest || (rest == o.rest && id < o.id); }
  };
  std::vector<Exit> exits;""")

rep("""    int st = sigma;
    size_t k = size_t(base);
    bool alive = true, left = false;
    for (; k < s.size(); ++k) {
      const int r = lex(lexer, st, uint8_t(s[k]));
      if (r == kLexDead) {
        alive = false;
        break;
      }
      if (r == kLexExit) {
        left = true;
        break;
      }
      st = r;
    }
    if (!alive) continue;
    if (left) exits.push_back({s.substr(k + 1), id});
    else b->stay.push_back(id);
  }
  if (b->stay.size() > 4096) {
    b->bits.assign(size_t(words()), 0);
    for (int id : b->stay) b->bits[size_t(id) >> 5] |= 1u << (id & 31);
    std::vector<int32_t>().swap(b->stay);
  }""", """    int st = sigma, chars = 0;
    size_t k = size_t(base);
    bool alive = true, left = false;
    for (; k < s.size(); ++k) {
      const int r = lex(lexer, st, uint8_t(s[k]));
      if (r == kLexDead) {
        alive = false;
        break;
      }
      if (r == kLexExit) {
        left = true;
        break;
      }
      if (st == 0) ++chars;  // (a JSON string's: a character starts here)
      st = r;
    }
    if (!alive) continue;
    if (left) {
      exits.push_back({s.substr(k + 1), id, chars});
    } else {
      b->stay.push_back(id);
      b->stay_n.push_back(uint16_t(chars));
    }
  }
  if (b->stay.size() > 4096) {
    b->bits.assign(size_t(words()), 0);
    for (int id : b->stay) b->bits[size_t(id) >> 5] |= 1u << (id & 31);
    if (lexer != kLexStr) std::vector<int32_t>().swap(b->stay), std::vector<uint16_t>().swap(b->stay_n);
  }""")

rep("""  std::vector<std::vector<int32_t>> at(1);
  std::vector<uint32_t> path{0};
  const std::string* prev = nullptr;
  for (const auto& [rest, id] : exits) {""", """  std::vector<std::vector<std::pair<int32_t, uint16_t>>> at(1);
  std::vector<uint32_t> path{0};
  const std::string* prev = nullptr;
  for (const auto& [rest, id, chars] : exits) {""")
rep("""    at[path.back()].push_back(id);
    b->max_depth = std::max(b->max_depth, int(rest.size()));""", """    at[path.back()].push_back({id, uint16_t(chars)});
    b->max_depth = std::max(b->max_depth, int(rest.size()));""")
rep("""    for (int id : at[i]) b->toks.push_back(id);""", """    for (const auto& [id, chars] : at[i]) b->toks.push_back(id), b->toks_n.push_back(chars);""")
open(p, "w").write(s)
print("patched")
