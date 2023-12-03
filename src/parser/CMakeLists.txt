cmake_minimum_required(VERSION 3.16)

add_library(parser STATIC
  AST.cpp
)
##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(parser PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(parser LLVM-16)

##=== MLIR ===##

target_include_directories(parser PUBLIC ${PROJECT_SOURCE_DIR}/include)
