#!/usr/bin/env python3
"""Write ksx_h1.metal, the kernel file patch 0002 reads at run time (SPLASH_KSX_KERNELS), for the Splash checkout given.

  gen_kernels.py SPLASH_SRC OUT.metal

Splash compiles the file with newLibraryWithSource, so it cannot include Splash's headers: this flattens the helpers of
Splash's runtime/metal/kernels/shared/gguf_repack.metal (everything before its gguf_repack kernel) into it, turns every
pointer those helpers read staging through into a system-coherent one, and appends kernels/ksx_h1_kernels.metal (the H1
kernels, written by hand). The coherent loads let ksx_moe_expert_load_c read staging the CPU wrote during the same
command buffer; its arithmetic is Splash's moe_expert_load's. The output follows the pinned Splash source, so it is made
at build time and never committed.

Exit 1, with what changed, when the helpers no longer have the 8 staging pointers this transform expects: an Inco release
that changed gguf_repack.metal needs the transform reviewed before it ships."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HAND = os.path.join(HERE, "kernels", "ksx_h1_kernels.metal")
REPACK = "metal/kernels/shared/gguf_repack.metal"
POINTERS = 8


def flatten(root, path, seen, out):
    with open(path) as f:
        for line in f:
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if m:
                inc = os.path.join(root, m.group(1))
                if not os.path.exists(inc):
                    inc = os.path.join(os.path.dirname(path), m.group(1))
                inc = os.path.realpath(inc)
                if inc not in seen:
                    seen.add(inc)
                    flatten(root, inc, seen, out)
                continue
            if line.strip() == "#pragma once":
                continue
            out.append(line)


def generate(splash_src):
    root = os.path.join(splash_src, "runtime")
    repack = os.path.join(root, REPACK)
    lines = []
    flatten(root, repack, {os.path.realpath(repack)}, lines)
    text = "".join(lines)
    cut = text.find("kernel void gguf_repack(")
    if cut < 0:
        raise ValueError(f"{REPACK} has no gguf_repack kernel: Splash changed, review the transform")
    helpers, count = re.subn(r"\bdevice const (uchar|ushort) \*", r"KSXC \1 *", text[:cut])
    helpers = helpers.replace("(device const ushort *)", "(KSXC ushort *)")
    head, sep, body = helpers.partition("using namespace metal;")
    if not sep or count != POINTERS or "device const" in body:
        raise ValueError(f"{REPACK}'s helpers have {count} staging pointers, not {POINTERS}: Splash changed, review the transform")
    with open(HAND) as f:
        hand = f.read()
    return ("#pragma METAL internals : enable\n" + head + "using namespace metal;\n"
            "#define KSXC volatile coherent(system) device const\n" + body + hand)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        text = generate(sys.argv[1])
    except (OSError, ValueError) as e:
        sys.exit(f"gen_kernels: {e}")
    os.makedirs(os.path.dirname(os.path.abspath(sys.argv[2])), exist_ok=True)
    with open(sys.argv[2], "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
