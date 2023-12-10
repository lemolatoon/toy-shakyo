add_mlir_dialect(ToyOps toy)

set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(dialect.h.inc -gen-dialect-decls)
mlir_tablegen(dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyOpsIncGen)