#!/usr/bin/env bash
# Local, Linux-only syntax check of the C++ side. This is NOT a build and proves nothing about Metal or runtime
# behaviour: it catches typos, wrong signatures, missing includes and type errors in the code we write, which is the
# one class of mistake that is otherwise only caught by CI. Uses a scratch stub for macOS-only headers (never committed).
#
#   tools/lx_syntax_check.sh                 # the Spark files + the files they touch
#   tools/lx_syntax_check.sh src/model/x.cpp # specific ones
set -uo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
shim="${TMPDIR:-/tmp}/lxshim"
mkdir -p "$shim/sys"
# macOS-only headers a Linux g++ lacks. Local scratch only: this file is not used by CI or by any real build.
# struct stat's st_mtimespec (macOS) in place of st_mtim (Linux): same value, different name in the C library's header.
cat > "$shim/lx_stat.h" <<'EOF'
#pragma once
#include <sys/stat.h>
// macOS spells the nanosecond field st_mtimespec; define the name rather than the struct, so <sys/stat.h> still applies
#define st_mtimespec st_mtim
EOF
cat > "$shim/sys/sysctl.h" <<'EOF'
#pragma once
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif
int sysctlbyname(const char*, void*, size_t*, void*, size_t);
int sysctl(int*, unsigned, void*, size_t*, void*, size_t);
#ifdef __cplusplus
}
#endif
EOF

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  files=(
    "src/model/spark_config.cpp"
    "src/model/spark.cpp"
    "src/engine/engine.cpp"
    "src/engine/engine_disk.cpp"  # the SSD prefix cache, where Spark's disk_format/disk_rows/disk_state are checked
    "tools/spark_config_test.cpp"
    "tools/ring_slot_test.cpp"
  )
fi

fail=0
for f in "${files[@]}"; do
  [ -f "$root/$f" ] || { echo "missing: $f"; fail=1; continue; }
  # spark_config_test includes the generated kernel-name list (tools/spark_kernel_names.py); generate it first
  mkdir -p "$shim/gen" && python3 "$root/tools/spark_kernel_names.py" "$root/src/kernels" "$shim/gen/spark_kernel_names.h" >/dev/null 2>&1
  if out=$(g++ -std=c++20 -fsyntax-only -Isrc -Ithird_party -I"$shim" -I"$shim/gen" -include lx_stat.h \
             -D'__builtin_available(a,b)=1' "$f" 2>&1); then
    printf 'ok    %s\n' "$f"
  else
    printf 'FAIL  %s\n' "$f"
    printf '%s\n' "$out" | grep -E "error:" | head -20
    fail=1
  fi
done
exit "$fail"
