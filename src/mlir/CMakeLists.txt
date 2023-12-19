add_library(mlir STATIC
  dialect.cpp
  mlirGen.cpp
  toyCombine.cpp
  shapeInferencePass.cpp
  lowerToAffineLoops.cpp
  lowerToLLVM.cpp
  putOutArithConstantPass.cpp
  replaceIndexToI64.cpp
  gpuEraseIndexArgPass.cpp
)
##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(mlir PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(mlir ${llvm_libs})

##=== MLIR ===##
include_directories(${MLIR_INCLUDE_DIRS})
target_link_directories(mlir PRIVATE ${MLIR_LIBRARY_DIRS})
target_link_libraries(mlir ${mlir_libs})

target_include_directories(mlir PUBLIC ${PROJECT_SOURCE_DIR}/include)