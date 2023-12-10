cmake_minimum_required(VERSION 3.16)


find_package(GTest CONFIG REQUIRED)

add_executable(googleTest 
  lexer_test.cpp
  parser_test.cpp
  ast_snap_test.cpp
  rewrite_test.cpp
  test_util.cpp
)

target_link_libraries(googleTest GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

# internal
target_link_libraries(googleTest parser mlir)

target_include_directories(googleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)

# external libraries

##=== LLVM ===##
target_include_directories(googleTest PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_directories(googleTest PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(googleTest LLVM-16)

##=== MLIR ===##
# linking with internal `mlir` target already links with `MLIR`

enable_testing()