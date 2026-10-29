#!/usr/bin/env python3
"""Build the KFlash Splash engine: Splash at the commit in pin.json with patches/*.patch applied in order.

  build.py --release SPLASH_RELEASE_DIR [--out DIR] [--cache DIR]   clone, check, patch, build, write the kernels
  build.py --check [--cache DIR]                                     clone, check and apply the patches only (any OS)

SPLASH_RELEASE_DIR is Splash's unpacked release archive for the pinned tag (engine/splash.metallib,
engine/host-step.metallib, release.json): the build takes its metallibs, checked against pin.json, and compiles only the
host code, so the KFlash binary differs from Splash's release by the patches alone. The output goes to
DIR (default <kapitsa home>/engines/splash-kflash-<tag>-<id>, id from pin.json, the patches and the kernels), with
engine/splash, engine/*.metallib, kernels/ksx_h1.metal and kflash.json (what was built from what). The engine runs
KFlash with SPLASH_KSX_H1=3 SPLASH_KSX_KERNELS=<DIR>/kernels; unset, it runs as Splash.

Fails, naming the step, when the clone is not the pinned commit, a patch stops applying, the release's metallibs are
not the pinned ones, or the build fails. Nothing is reused from a failed build: DIR is rebuilt from scratch. DIR is
deleted first only when it is empty or an earlier build.py output (MARKER or kflash.json); any other DIR is refused."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_kernels  # noqa: E402


class BuildError(Exception):
    pass


MARKER = ".kflash-build"  # written first into DIR, so a failed build's leftovers are recognised as this script's


def clear_out(out):
    """Make out an empty directory carrying MARKER. Deletes only an empty directory or an earlier output of this script (MARKER, or a finished build's kflash.json)."""
    if os.path.lexists(out):
        if os.path.islink(out) or not os.path.isdir(out):
            raise BuildError(f"--out {out} exists and is not a directory")
        if os.listdir(out) and not any(os.path.isfile(os.path.join(out, n)) for n in (MARKER, "kflash.json")):
            raise BuildError(f"--out {out} exists and is not a KFlash build: not deleting it; pass a new or empty directory")
        shutil.rmtree(out)
    os.makedirs(out)
    open(os.path.join(out, MARKER), "w").close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd, cwd=None, step=""):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(f"{step or cmd[0]} failed ({' '.join(cmd)}):\n{r.stdout[-2000:]}{r.stderr[-4000:]}")
    return r.stdout


def load_pin():
    with open(os.path.join(HERE, "pin.json")) as f:
        return json.load(f)


def patches():
    d = os.path.join(HERE, "patches")
    return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith(".patch")]


def build_id(pin):
    """Changes with anything that changes the build: the pin, a patch, the hand-written kernels, this script."""
    h = hashlib.sha256()
    h.update(pin["commit"].encode())
    for p in patches() + [gen_kernels.HAND, os.path.abspath(__file__), gen_kernels.__file__]:
        h.update(os.path.basename(p).encode())
        h.update(sha256(p).encode())
    return h.hexdigest()[:12]


def kapitsa_home():
    h = os.environ.get("KAPITSA_HOME")
    return os.path.expanduser(h).rstrip("/") if h else os.path.expanduser("~/.cache/kapitsa")


def pinned_source(pin, cache, dest):
    """A fresh checkout of the pinned commit at dest, from a clone kept in cache (fetched when it lacks the commit)."""
    if not os.path.isdir(os.path.join(cache, ".git")):
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        run(["git", "clone", "--quiet", "--no-checkout", pin["repo"], cache], step="clone")
    have = subprocess.run(["git", "cat-file", "-e", pin["commit"] + "^{commit}"], cwd=cache, capture_output=True)
    if have.returncode != 0:
        run(["git", "fetch", "--quiet", "--tags", "origin"], cwd=cache, step="fetch")
    tag = run(["git", "rev-parse", f"refs/tags/{pin['tag']}^{{commit}}"], cwd=cache, step="tag lookup").strip()
    if tag != pin["commit"]:
        raise BuildError(f"Splash tag {pin['tag']} is {tag}, pin.json says {pin['commit']}: the tag moved")
    run(["git", "clone", "--quiet", "--shared", "--no-checkout", cache, dest], step="checkout")
    run(["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", pin["commit"]], cwd=dest, step="checkout")
    head = run(["git", "rev-parse", "HEAD"], cwd=dest).strip()
    if head != pin["commit"]:
        raise BuildError(f"checkout is {head}, not the pinned {pin['commit']}")


def apply_patches(src):
    for p in patches():
        name = os.path.basename(p)
        r = subprocess.run(["git", "apply", "--check", p], cwd=src, capture_output=True, text=True)
        if r.returncode != 0:
            raise BuildError(f"patch {name} no longer applies to the pinned Splash:\n{r.stderr}")
        run(["git", "apply", p], cwd=src, step=f"apply {name}")


def check(pin, cache):
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "splash")
        pinned_source(pin, cache, src)
        apply_patches(src)
        gen_kernels.generate(src)
    print(f"ok: {len(patches())} patches apply to Splash {pin['tag']} ({pin['commit'][:8]}), kernels generate")


def build(pin, cache, release, out):
    with open(os.path.join(release, "release.json")) as f:
        rel = json.load(f)
    if rel.get("version") != pin["tag"]:
        raise BuildError(f"{release} is Splash {rel.get('version')}, the pin is {pin['tag']}")
    for m, key in (("splash.metallib", "release_metallib_sha256"), ("host-step.metallib", "release_host_step_metallib_sha256")):
        path = os.path.join(release, "engine", m)
        if sha256(path) != pin[key]:
            raise BuildError(f"{path} is not the pinned release's {m}")
    clear_out(out)
    src = os.path.join(out, "src")
    pinned_source(pin, cache, src)
    apply_patches(src)
    os.makedirs(os.path.join(src, "build"))
    for m in ("splash.metallib", "host-step.metallib"):
        shutil.copy2(os.path.join(release, "engine", m), os.path.join(src, "build", m))
    run(["make", f"-j{os.cpu_count() or 4}", "-o", "build/splash.metallib", "-o", "build/host-step.metallib",
         "build/splash"], cwd=src, step="make")
    engine = os.path.join(out, "engine")
    os.makedirs(engine)
    for m in ("splash", "splash.metallib", "host-step.metallib"):
        shutil.copy2(os.path.join(src, "build", m), os.path.join(engine, m))
    kernels = os.path.join(out, "kernels")
    os.makedirs(kernels)
    with open(os.path.join(kernels, "ksx_h1.metal"), "w") as f:
        f.write(gen_kernels.generate(src))
    binary = sha256(os.path.join(engine, "splash"))
    want = pin.get("kflash_binary_sha256")
    record = {
        "splash_tag": pin["tag"], "splash_commit": pin["commit"],
        "patches": {os.path.basename(p): sha256(p) for p in patches()},
        "binary_sha256": binary, "pinned_binary_sha256": want or None,
        "kernels_sha256": sha256(os.path.join(kernels, "ksx_h1.metal")),
        "env": {"SPLASH_KSX_H1": "3", "SPLASH_KSX_KERNELS": kernels},
    }
    with open(os.path.join(out, "kflash.json"), "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    shutil.rmtree(src)
    print(f"built {engine}/splash ({binary[:16]})")
    if want and binary != want:
        # The toolchain differs from the measured build's: the binary is not the one the A/B measured.
        print(f"warning: the binary is not the measured one ({want[:16]}); a different Xcode SDK builds different code",
              file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="clone and apply the patches only")
    ap.add_argument("--release", help="Splash's unpacked release archive for the pinned tag")
    ap.add_argument("--out", help="output directory (default <kapitsa home>/engines/splash-kflash-<tag>-<id>)")
    ap.add_argument("--cache", help="where the Splash clone is kept (default <kapitsa home>/src/splash)")
    a = ap.parse_args()
    pin = load_pin()
    cache = a.cache or os.path.join(kapitsa_home(), "src", "splash")
    try:
        if a.check:
            check(pin, cache)
            return
        if not a.release:
            ap.error("--release is required unless --check")
        out = a.out or os.path.join(kapitsa_home(), "engines", f"splash-kflash-{pin['tag']}-{build_id(pin)}")
        build(pin, cache, a.release, out)
    except (BuildError, OSError, ValueError) as e:
        sys.exit(f"splash build: {e}")


if __name__ == "__main__":
    main()
