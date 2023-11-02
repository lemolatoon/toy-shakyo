cmake_minimum_required(VERSION 3.16)

find_package(GTest CONFIG REQUIRED)

add_executable(googleTest 
  lib_test.cpp
)

target_link_libraries(googleTest GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)

# internal
target_link_libraries(googleTest mylib)

target_include_directories(googleTest PUBLIC ${PROJECT_SOURCE_DIR}/include)

enable_testing()