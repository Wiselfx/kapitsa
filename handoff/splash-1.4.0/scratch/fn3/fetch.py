#!/usr/bin/env python3
"""fetch.py CACHE REPO REVISION PATH [PATH ...] [--workers N] [--min-free-gb G]: the named files of one pinned Hub
revision into a Hugging Face cache layout under CACHE (models--ORG--REPO/blobs/<sha256 or git oid>, snapshots/REV/PATH
as relative symlinks, refs/main = REV), by parallel HTTP range requests of 64 MiB. Resumable: the finished pieces of
each file are kept in <blob>.incomplete.pieces. Stops (exit 3) when the data volume's free space falls below
--min-free-gb. Each finished LFS file is checked against its SHA-256 before it is renamed to its blob. Stdlib only."""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "kapitsa-bench-fetch/0.1"}
PIECE = 64 << 20


def api_tree(repo, rev, folder):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}" + (f"/{folder}" if folder else "")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


class Stop(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("repo")
    ap.add_argument("rev")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=11.0)
    a = ap.parse_args()
    base = os.path.join(a.cache, "models--" + a.repo.replace("/", "--"))
    blobs, snap = os.path.join(base, "blobs"), os.path.join(base, "snapshots", a.rev)
    os.makedirs(blobs, exist_ok=True)
    os.makedirs(os.path.join(base, "refs"), exist_ok=True)
    with open(os.path.join(base, "refs", "main"), "w") as f:
        f.write(a.rev)
    meta = {}
    for folder in sorted({os.path.dirname(p) for p in a.paths}):
        for e in api_tree(a.repo, a.rev, folder):
            meta[e["path"]] = e
    lock = threading.Lock()
    stop = threading.Event()
    got = [0]
    t0 = time.time()
    for path in a.paths:
        e = meta[path]
        size = e["size"]
        lfs = e.get("lfs") or {}
        oid = lfs.get("oid") or e["oid"]
        blob = os.path.join(blobs, oid)
        link = os.path.join(snap, path)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not (os.path.exists(blob) and os.path.getsize(blob) == size):
            url = f"https://huggingface.co/{a.repo}/resolve/{a.rev}/{path}"
            tmp = blob + ".incomplete"
            pfile = tmp + ".pieces"
            done = set(json.load(open(pfile))) if os.path.exists(pfile) and os.path.exists(tmp) else set()
            left = max(0, size - len(done) * PIECE)
            if free_gb(a.cache) - left / 1e9 < a.min_free_gb:
                print(f"{path}: the {left / 1e9:.2f} GB still to fetch would leave less than {a.min_free_gb} GB free "
                      f"({free_gb(a.cache):.1f} GB free now)", flush=True)
                sys.exit(3)
            if not os.path.exists(tmp):
                open(tmp, "wb").close()
            n = (size + PIECE - 1) // PIECE
            todo = [i for i in range(n) if i not in done]
            print(f"{path}: {size / 1e9:.2f} GB, {len(done)} of {n} pieces already here, {a.workers} connections",
                  flush=True)

            def piece(i, url=url, tmp=tmp, size=size, done=done, pfile=pfile):
                st, en = i * PIECE, min((i + 1) * PIECE, size) - 1
                for t in range(10):
                    if stop.is_set():
                        raise Stop()
                    try:
                        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={st}-{en}"})
                        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "r+b") as f:
                            if r.status != 206:
                                raise IOError(f"HTTP {r.status} for a range request")
                            f.seek(st)
                            k = 0
                            while True:
                                b = r.read(1 << 20)
                                if not b:
                                    break
                                f.write(b)
                                k += len(b)
                                with lock:
                                    got[0] += len(b)
                            if k != en - st + 1:
                                raise IOError(f"short piece {k} != {en - st + 1}")
                        with lock:
                            done.add(i)
                            json.dump(sorted(done), open(pfile + ".tmp", "w"))
                            os.replace(pfile + ".tmp", pfile)
                        return
                    except Stop:
                        raise
                    except Exception as ex:  # retry
                        print(f"  piece {i}: {ex!r} (try {t + 1})", flush=True)
                        time.sleep(2 + 3 * t)
                raise RuntimeError(f"piece {i} of {url} failed")

            def guard():
                while not stop.is_set() and not all_done.is_set():
                    if free_gb(a.cache) < a.min_free_gb:
                        print(f"disk guard: {free_gb(a.cache):.1f} GB free, stopping", flush=True)
                        stop.set()
                    el = time.time() - t0
                    with lock:
                        g = got[0]
                    print(f"  {time.strftime('%H:%M:%S')} {path}: {len(done)}/{n} pieces, {g / 1e9:.2f} GB this run, "
                          f"{g / 1e6 / max(el, 1):.1f} MB/s, {free_gb(a.cache):.1f} GB free", flush=True)
                    all_done.wait(30)

            all_done = threading.Event()
            threading.Thread(target=guard, daemon=True).start()
            with ThreadPoolExecutor(a.workers) as ex:
                futs = [ex.submit(piece, i) for i in todo]
                err = None
                for f in futs:
                    try:
                        f.result()
                    except Exception as x:
                        err = err or x
                        stop.set()
            all_done.set()
            if err:
                print(f"{path}: stopped ({err!r})", flush=True)
                sys.exit(3 if isinstance(err, Stop) else 1)
            if os.path.getsize(tmp) != size:
                print(f"{path}: size {os.path.getsize(tmp)} != {size}", flush=True)
                sys.exit(1)
            if lfs.get("oid"):
                h = hashlib.sha256()
                with open(tmp, "rb") as f:
                    while True:
                        b = f.read(64 << 20)
                        if not b:
                            break
                        h.update(b)
                if h.hexdigest() != lfs["oid"]:
                    print(f"{path}: SHA-256 {h.hexdigest()} != {lfs['oid']}: kept as {tmp}", flush=True)
                    sys.exit(1)
                print(f"{path}: SHA-256 ok", flush=True)
            os.replace(tmp, blob)
            os.remove(pfile)
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.relpath(blob, os.path.dirname(link)), link)
        print(f"{path}: done -> {link}", flush=True)
    print(f"all done in {time.time() - t0:.0f} s, {got[0] / 1e9:.2f} GB fetched", flush=True)


if __name__ == "__main__":
    main()
