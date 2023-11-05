cmake_minimum_required(VERSION 3.16)


find_package(GTest CONFIG REQUIRED)

add_executable(googleTest 
  lib_test.cpp
  lexer_test.cpp
  parser_test.cpp
)

target_link_libraries(googleTest GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

# internal
target_link_libraries(googleTest mylib)

target_include_directories(googleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)

# external libraries

##=== LLVM ===##
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(googleTest PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(googleTest LLVM-16)

##=== MLIR ===##
message(STATUS "Trying to search MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

enable_testing()