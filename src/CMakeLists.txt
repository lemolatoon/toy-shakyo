cmake_minimum_required(VERSION 3.16)

add_subdirectory(parser)
add_subdirectory(mlir)

add_executable(a.out toyc.cpp)
target_link_libraries(a.out parser mlir)

##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(a.out PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(a.out LLVM-16)

##=== MLIR ===##
include_directories(${MLIR_INCLUDE_DIRS})
target_link_directories(a.out PRIVATE ${MLIR_LIBRARY_DIRS})

target_include_directories(a.out PUBLIC ${PROJECT_SOURCE_DIR}/include)