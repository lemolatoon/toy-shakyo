cmake_minimum_required(VERSION 3.16)

add_subdirectory(parser)

add_executable(a.out toyc.cpp)
target_link_libraries(a.out parser)

##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(a.out PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(a.out LLVM-16)

##=== MLIR ===##

target_include_directories(a.out PUBLIC ${PROJECT_SOURCE_DIR}/include)