cmake_minimum_required(VERSION 3.16)

add_library(parser STATIC
  AST.cpp
)
##=== LLVM ===##
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(parser PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(parser LLVM-16)

##=== MLIR ===##
message(STATUS "Trying to search MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
target_include_directories(parser PUBLIC ${PROJECT_SOURCE_DIR}/include)
