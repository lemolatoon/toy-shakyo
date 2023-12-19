cmake_minimum_required(VERSION 3.16)


find_package(GTest CONFIG REQUIRED)

add_executable(googleTest 
  lexer_test.cpp
  parser_test.cpp
  ast_snap_test.cpp
  rewrite_test.cpp
  interface_test.cpp
  lowering_test.cpp
  test_util.cpp
)

target_link_libraries(googleTest GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

# internal
target_link_libraries(googleTest parser mlir)

target_include_directories(googleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)

# external libraries

##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(googleTest PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(googleTest ${llvm_libs})

##=== MLIR ===##
include_directories(${MLIR_INCLUDE_DIRS})
target_link_directories(googleTest PRIVATE ${MLIR_LIBRARY_DIRS})
target_link_libraries(googleTest ${mlir_libs})

enable_testing()