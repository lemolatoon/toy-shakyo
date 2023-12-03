add_library(mlir STATIC
  dialect.cpp
)
##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(mlir PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(mlir LLVM-16)

##=== MLIR ===##
include_directories(${MLIR_INCLUDE_DIRS})
target_link_directories(mlir PRIVATE ${MLIR_LIBRARY_DIRS})
target_link_libraries(mlir MLIR)

target_include_directories(mlir PUBLIC ${PROJECT_SOURCE_DIR}/include)