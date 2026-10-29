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
    "tools/spark_config_test.cpp"
  )
fi

fail=0
for f in "${files[@]}"; do
  [ -f "$root/$f" ] || { echo "missing: $f"; fail=1; continue; }
  if out=$(g++ -std=c++20 -fsyntax-only -Isrc -Ithird_party -I"$shim" \
             -D'__builtin_available(a,b)=1' "$f" 2>&1); then
    printf 'ok    %s\n' "$f"
  else
    printf 'FAIL  %s\n' "$f"
    printf '%s\n' "$out" | grep -E "error:" | head -20
    fail=1
  fi
done
exit "$fail"
