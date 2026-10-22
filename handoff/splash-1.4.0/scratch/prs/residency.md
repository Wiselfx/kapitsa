## What changed

Metal unwires a residency set about 2 s after the process's last command buffer, whether or not it was requested
(M4 Pro, macOS 27.0.1: a requested 8 GB set was unwired 2–3 s after one command buffer, even one that bound nothing in
it). Kapitsa attached its one residency set, holding every buffer, to the command queue. So the first command buffer
after any pause of a few seconds waited until the whole model was wired again. Coding agents pause like that between
turns all the time.

Kapitsa binds every buffer a kernel reads to its dispatches: no `gpuAddress`, argument buffers or `useResource` in
`src/` or `tools/`. So the queue doesn't need the set.

- The set is no longer attached to the queue.
- A heartbeat (a GCD timer, every 500 ms) requests the set while command buffers are committed and for
  `NI_RESIDENCY_KEEP_S` seconds after the last one (default 600; 0 = forever). The first commit after that restarts it.
- Under memory pressure (`DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` at warning or critical, with
  `kern.memorystatus_vm_pressure_level` re-read each beat), the heartbeat stops. One small blit then lets Metal unwire
  the set, so an idle server gives its memory back as on main. Each level change is logged in one line.
- `NI_RESIDENCY=queue` brings back the old behaviour, and `NI_RESIDENCY=0` runs without a set.
- Two diagnostics: `NI_CB_LAT=<file>` writes each command buffer's commit, GPU start and GPU end times.
  `NI_RESIDENCY_PRESSURE_TEST=<file>` takes the pressure level from a file.

**Behaviour change:** at normal memory pressure, the weights now stay wired between requests for up to 10 minutes
after the last command buffer. Before, they were unwired about 2 s after it.

## Measured

Runs: MacBook Pro 16-inch 2024, M4 Pro (20-core GPU), 48 GB, macOS 27.0.1, Qwen3.6-35B-A3B 4-bit (DFlash2, 1 slot,
32K context). Base: main 0553f55. Builds interleaved under the GPU lock.

Time to the first streamed token for a 2,170-token cached prompt:

| | Right after a request | After 10 s idle | After 20 s idle with a 6 GB memory hog |
| --- | ---: | ---: | ---: |
| main | 27, 27 ms | 432, 535 ms | 431, 456 ms |
| `NI_RESIDENCY=queue` | 27, 27 ms | 457, 459 ms | 477, 450 ms |
| this PR | 27, 28 ms | 49–118 ms | 46–57 ms |

- **Commit to GPU start:** main 394–495 ms, this PR 12–82 ms.
- **First request after load:** 1,969–1,991 ms in every build.
- **Decode** (`ni_run`, plain, 2,047 tokens):
  - main 108.5 / 103.5 tok/s, this PR 108.9 / 107.2 tok/s.
  - With a hog at the warning level: main 100.0, this PR 100.9 / 98.4 tok/s.
- **Synthetic test** (a 20 GB set, the first command buffer after 10 s idle): on the queue 162–197 ms; off the queue
  with the heartbeat 1.7–8.1 ms.
- **Simulated pressure warning** (`NI_RESIDENCY_PRESSURE_TEST`):
  - The set dropped from 24.4 to 3.5 GB within 3 s.
  - Requests after idle went back to main's 556–567 ms.
  - After "normal", the next request restarted the heartbeat (450 ms), and the one 10 s later was fast again
    (50.4 ms; 24.6 GB wired).

## Checks

- `api_test`, `test_server.py --cli`, `test_sidecar.py`, `test_update.py`, `test_reasoning.py`, `test_pr_size.py`.
- `test_bench.py` passes everything except its jsonschema case, which needs jsonschema, not installed on this Mac.

## Not verified

- Real (not simulated) memory-pressure warning and critical events.
- Decode under pressure with the final build.
- Other chips and macOS versions.
- Nemotron-H, Dory and vision models (same code path, not run).
- The heartbeat's power cost.
- Idle periods longer than the 600 s window.
- `test_community.py` (needs jsonschema).

The approach follows Splash 1.4.0's residency change (Apache-2.0); no code was taken.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
