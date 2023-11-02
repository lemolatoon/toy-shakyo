cmake_minimum_required(VERSION 3.16)

add_library(mylib STATIC
  hello.cpp
)
target_include_directories(mylib PUBLIC ${PROJECT_SOURCE_DIR}/include)
