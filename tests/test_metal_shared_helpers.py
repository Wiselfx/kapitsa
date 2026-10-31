#!/usr/bin/env python3
"""Helpers defined in one kernel file and called from another must keep the signature the callers use.

The Metal sources are concatenated into ONE translation unit per library (cmake/embed_kernels*.cmake), so every
NI_INLINE helper is visible to every other file in its library. Changing one helper's signature therefore silently
breaks callers in files the author never opened -- which is exactly what happened when the ring read added a params
argument to attn11_fetch_k / attn11_fetch_v: they live in attn.metal, and verify8.metal (speculative verification, a
different file) kept calling the 5-argument form. That is a hard compile error, but only on a Mac, only at model load.

This is deliberately narrow. A general C++-in-Python arity checker was tried first and produced ~20 false positives
(default arguments, macro-generated bodies, calls whose arguments contain commas inside ternaries), so it would have
been ignored within a week. Instead it checks only the helpers that actually cross a file boundary -- the ones where
an edit in one file can break another -- and it parses no expressions: it just counts top-level commas.

When this fails, either update the callers or (as here) keep an overload for the old arity and say why.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
KERNELS = ROOT / "src" / "kernels"


def strip_comments(src):
    out, i, n = [], 0, len(src)
    while i < n:
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in src[i:j]))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


DEF_RE = re.compile(r"NI_INLINE\s+[\w:<>&\s\*]+?\b(\w+)\s*\(([^)]*)\)\s*\{")


def split_params(s):
    """Split an argument list on its top-level commas.

    Tracks ()[]{} as brackets, and treats <...> as a template argument list only when it really is one. The naive
    version ("every < opens, every > closes") miscounts `a << 16` as two opens, which is how `wfmt.metal`'s
    `v8_mma(dot, as_type<half2>(c[i0] | (c[i0 + 4] << 16) | 0x64006400u), b[j])` came out as 2 arguments instead of 3
    and the first run of this test reported a bug that did not exist. Shift operators are skipped as a unit, and a '<'
    only opens a template when an identifier immediately precedes it and the next character is neither '<' nor '='.
    """
    s = s.strip()
    if not s or s == "void":
        return []
    depth = angle = 0
    cur, out = "", []
    i, n = 0, len(s)
    prev_sig = ""  # last non-space character emitted
    while i < n:
        ch = s[i]
        two = s[i:i + 2]
        if two in ("<<", ">>"):
            # a shift operator, not brackets -- unless we are inside a template already and this closes one
            if two == ">>" and angle >= 2:
                angle -= 2
            elif two == ">>" and angle == 1:
                angle -= 1
            cur += two
            i += 2
            prev_sig = ">"
            continue
        if ch == "<":
            if (prev_sig.isalnum() or prev_sig == "_") and s[i + 1:i + 2] not in ("<", "=") and angle == 0 and depth == 0:
                angle += 1
            elif angle:
                angle += 1
            cur += ch
            prev_sig = ch
            i += 1
            continue
        if ch == ">":
            if angle:
                angle -= 1
            cur += ch
            prev_sig = ch
            i += 1
            continue
        if ch in "([{":
            depth += 1 if angle == 0 else 0
            angle += 0
            cur += ch
            prev_sig = ch
            i += 1
            continue
        if ch in ")]}":
            depth -= 1 if angle == 0 else 0
            cur += ch
            prev_sig = ch
            i += 1
            continue
        if ch == "," and depth == 0 and angle == 0:
            out.append(cur)
            cur = ""
            prev_sig = ch
            i += 1
            continue
        cur += ch
        if not ch.isspace():
            prev_sig = ch
        i += 1
    out.append(cur)
    return [p for p in out if p.strip()]


def main():
    # library -> [(path, stripped source)]
    libs = {}
    for f in sorted(KERNELS.rglob("*.metal")):
        lib = "tensor" if f.parent.name == "tensor" else "base"
        libs.setdefault(lib, []).append((f, strip_comments(f.read_text())))

    fails, checked = [], 0
    for lib, files in libs.items():
        defs = {}      # name -> (set of arities, files defining it)
        callers = {}   # name -> set of files calling it
        for path, src in files:
            for m in DEF_RE.finditer(src):
                ps = split_params(m.group(2))
                lo = sum(1 for p in ps if "=" not in p)
                defs.setdefault(m.group(1), (set(), set()))[0].add((lo, len(ps)))
                defs[m.group(1)][1].add(path.name)
            for name in list(defs) + re.findall(r"\b(\w+)\s*(?:<[^;()]*?>)?\s*\(", src):
                if name in defs and name not in callers.setdefault(name, set()):
                    # count this file as a caller only if it has a call-shaped occurrence beyond the definition
                    body = re.sub(DEF_RE, "", src)
                    if re.search(r"\b" + re.escape(name) + r"\s*(?:<[^;()]*?>)?\s*\(", body):
                        callers[name].add(path.name)

        for name, (arities, definers) in defs.items():
            if not callers.get(name):
                continue
            # the risk is only real when a DIFFERENT file calls it than defines it
            other = callers[name] - definers
            if not other:
                continue
            checked += 1
            lo = min(a[0] for a in arities)
            hi = max(a[1] for a in arities)
            for path, src in files:
                if path.name not in other:
                    continue
                body = re.sub(DEF_RE, "", src)
                for m in re.finditer(r"\b" + re.escape(name) + r"\s*(?:<[^;()]*?>)?\s*\(", body):
                    start = m.end() - 1
                    depth, i = 0, start
                    while i < len(body):
                        if body[i] in "([{":
                            depth += 1
                        elif body[i] in ")]}":
                            depth -= 1
                            if depth == 0:
                                break
                        i += 1
                    argc = len(split_params(body[start + 1:i])) if body[start + 1:i].strip() else 0
                    if not (lo <= argc <= hi):
                        fails.append("%s calls %s with %d argument(s); %s defines %s"
                                     % (path.name, name, argc, "/".join(sorted(definers)),
                                        "%d" % lo if lo == hi else "%d..%d" % (lo, hi)))

    if fails:
        for f in sorted(set(fails)):
            print("FAIL", f)
        print("\ntest_metal_shared_helpers: %d failure(s)" % len(set(fails)))
        return 1
    print("test_metal_shared_helpers: ok (%d helper(s) called across a file boundary, all arities match)" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
