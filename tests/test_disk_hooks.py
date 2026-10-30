#!/usr/bin/env python3
"""Every model that advertises an SSD prefix cache must implement the hooks that describe it.

The bug this exists for: Spark::disk_format() returned a non-empty string from the day the loader landed, while
disk_rows/disk_state fell through to the base class's no-ops. Nothing failed. The engine saw a format, enabled the
cache (the server defaults --disk-cache to "auto"), stored states that hashed as empty, and on a later request
disk_restore matched those empty layouts, restored nothing and returned success -- so a conversation would have
continued from an uninitialised cache. Silent wrongness, not a crash, which is the worst kind.

There is now a runtime guard (Model::disk_hooks_live, checked in Engine::disk_start) and this static check, which
catches the omission at review time rather than in a user's conversation.

Inheritance matters here: Clover (Qwen35), DSpark and LiliCorr (NhDrafter) override only disk_format and get the hooks
from their base, which is correct. So this resolves both members through the class hierarchy rather than looking at one
header in isolation -- the earlier version of this test reported five false positives for exactly that reason.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src/model"

fails = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def class_bodies():
    """class name -> (base, header text, body text)."""
    out = {}
    for h in sorted(SRC.glob("*.h")):
        text = h.read_text()
        for m in re.finditer(r"class (\w+)\s*:\s*public (\w+)\s*\{", text):
            body = text[m.end():]
            depth, i = 1, 0
            while i < len(body) and depth:
                if body[i] == "{":
                    depth += 1
                elif body[i] == "}":
                    depth -= 1
                i += 1
            out[m.group(1)] = (m.group(2), h, body[: i - 1])
    return out


def resolves(cls, member, classes, _seen=None):
    """The declaration of `member` that `cls` ends up with, following bases. None if nothing declares it."""
    _seen = _seen or set()
    if cls in _seen or cls not in classes:
        return None
    _seen.add(cls)
    base, _h, body = classes[cls]
    m = re.search(rf"\b{member}\(", body)
    if m:
        return body[max(0, m.start() - 60): m.start() + 200]
    return resolves(base, member, classes, _seen)


def body_of(cls, member, classes):
    """Whether `cls`'s own body declares `member`."""
    _base, _h, body = classes[cls]
    return re.search(rf"\b{member}\(", body) is not None


def main():
    src = (SRC / "model.h").read_text()
    classes = class_bodies()

    check(re.search(r'virtual std::string disk_format\(\) const \{ return ""; \}', src) is not None,
          "Model::disk_format defaults to empty, so silence means 'no SSD cache', not 'broken'")
    for hook in ("disk_rows", "disk_state"):
        check(re.search(rf"virtual void {hook}\(", src) is not None and
              re.search(rf"virtual void {hook}\(.*?(void\)s|\(void\)).*?\n  \}}", src, re.S) is not None,
              f"Model::{hook} defaults to a no-op")

    # Every model that ends up with a non-empty disk_format must also end up with both hooks declared somewhere up its
    # chain. A format is only "non-empty" if it is not the base's `return ""`.
    advertising, dead = [], []
    for cls in sorted(classes):
        fmt = resolves(cls, "disk_format", classes)
        if fmt is None or 'return ""' in fmt:
            continue
        rows = resolves(cls, "disk_rows", classes)
        state = resolves(cls, "disk_state", classes)
        # a resolved hook that came from Model itself is the no-op
        rows_live = rows is not None and re.search(r"\(void\)s, \(void\)mtp", rows) is None
        state_live = state is not None and re.search(r"\(void\)s, \(void\)c", state) is None
        advertising.append(cls)
        if not (rows_live and state_live):
            dead.append((cls, rows_live, state_live))

    check(bool(advertising), "at least one model advertises an SSD state format (else this test proves nothing)")
    for cls in advertising:
        rows_live = not any(d[0] == cls for d in dead)
        check(rows_live, f"{cls} advertises a disk format and has real disk_rows + disk_state")

    # the runtime guard, in case a future model reaches the engine anyway (a plugin, a split model)
    disk_start = (ROOT / "src/engine/engine_disk.cpp").read_text()
    check("disk_hooks_live" in disk_start, "disk_start checks disk_hooks_live before storing anything")
    check("bool disk_hooks_live(" in src, "Model::disk_hooks_live exists")

    print()
    if fails:
        print(f"test_disk_hooks: {len(fails)} check(s) failed")
        return 1
    print(f"test_disk_hooks: ok ({len(advertising)} model(s) advertising an SSD state format: "
          f"{', '.join(advertising)})")
    return 0


def check_pointer_stable_ownership():
    """A model that returns &owned_.back() must not keep owned_ in a std::vector.

    Spark's norm_copy handed out &owned_.back() for every norm it converted (73 of them on the 4B) while owned_ was a
    std::vector<Tensor>: each push could reallocate, and every pointer the loader was still holding -- L.in_norm,
    L.post_norm, final_norm_ -- dangled. Qwen35 keeps the same data in a std::deque, which is pointer-stable; the port
    copied the pattern and changed the container. Nothing crashed, so this is a static check.
    """
    spark_cpp = (SRC / "spark.cpp").read_text()
    if "&owned_.back()" not in spark_cpp:
        check(True, "spark.cpp no longer hands out &owned_.back(), so this check does not apply")
        return
    text = (SRC / "spark.h").read_text()
    check(re.search(r"std::vector<Tensor>\s+owned_;", text) is None,
          "spark.h: owned_ is not a std::vector (norm_copy returns &owned_.back())")
    check(re.search(r"std::deque<Tensor>\s+owned_;", text) is not None,
          "spark.h: owned_ is a std::deque, whose element addresses stay put")


check_pointer_stable_ownership()


if __name__ == "__main__":
    sys.exit(main())
