# Concatenates src/kernels/common.h + src/kernels/spark/*.metal into one C++ string literal: Spark's kernels, a library
# of their own compiled when a Spark model loads (Gpu::compile_extra). Kept apart from the base library so that adding
# the Spark family recompiles nothing the existing models run: their kernels are byte-identical to before.
file(READ ${SRC_DIR}/common.h COMMON)
set(ALL "${COMMON}\n")
file(GLOB METALS ${SRC_DIR}/spark/*.metal)
list(SORT METALS)
foreach(m ${METALS})
  file(READ ${m} BODY)
  string(REGEX REPLACE "#include \"[a-z_]+\\.h\"" "" BODY "${BODY}")
  get_filename_component(n ${m} NAME)
  string(APPEND ALL "\n// ---- spark/${n} ----\n${BODY}\n")
endforeach()
file(WRITE ${OUT} "// generated\nnamespace ni { extern const char* const kSparkKernelSource; const char* const kSparkKernelSource = R\"NIKERNEL(${ALL})NIKERNEL\"; }\n")
