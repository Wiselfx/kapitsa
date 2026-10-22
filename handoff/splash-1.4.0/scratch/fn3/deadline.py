#!/usr/bin/env python3
"""deadline.py SECONDS LOG cmd ...: runs cmd in a new session (its own process group, servers it starts included);
at SECONDS it sends TERM to the whole group, 15 s later KILL. When cmd ends first, whatever is left of its group gets
TERM, then KILL 5 s later. Appends start / end / signal lines to LOG."""
import os
import signal
import subprocess
import sys
import time


def note(log, msg):
    with open(log, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} deadline: {msg}\n")


def group_alive(pg):
    try:
        os.killpg(pg, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_group(pg, log, grace):
    if not group_alive(pg):
        return
    note(log, f"TERM to process group {pg}")
    os.killpg(pg, signal.SIGTERM)
    t = time.time()
    while time.time() - t < grace and group_alive(pg):
        time.sleep(0.25)
    if group_alive(pg):
        note(log, f"KILL to process group {pg}")
        os.killpg(pg, signal.SIGKILL)


def main():
    lim, log, cmd = float(sys.argv[1]), sys.argv[2], sys.argv[3:]
    t0 = time.time()
    note(log, f"start (limit {lim:.0f} s): {' '.join(cmd)}")
    p = subprocess.Popen(cmd, start_new_session=True)
    try:
        rc = p.wait(timeout=lim)
    except subprocess.TimeoutExpired:
        note(log, f"limit of {lim:.0f} s reached")
        stop_group(p.pid, log, 15)
        rc = p.wait()
    stop_group(p.pid, log, 5)
    # processes that left the group (Strata's engine runs in its own group): DEADLINE_PKILL names their command line
    pat = os.environ.get("DEADLINE_PKILL")
    if pat:
        def alive():
            return subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
        left = alive()
        if left:
            note(log, f"TERM to {left} (left the group: {pat})")
            subprocess.run(["pkill", "-TERM", "-f", pat])
            t = time.time()
            while time.time() - t < 15 and alive():
                time.sleep(0.5)
            if alive():
                note(log, f"KILL to {alive()}")
                subprocess.run(["pkill", "-KILL", "-f", pat])
    note(log, f"end: exit {rc} after {time.time() - t0:.0f} s")
    sys.exit(rc if rc >= 0 else 128 - rc)


if __name__ == "__main__":
    main()
