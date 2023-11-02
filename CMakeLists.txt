cmake_minimum_required(VERSION 3.16)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_CLANG_TIDY clang-tidy;checks=*,readability*,-header-filter=.*)

# CMAKE_TOOLCHAIN_FILE が無指定なら、環境変数 VCPKG_ROOT を元に設定.
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

# VCPKG_TARGET_TRIPLET が無指定なら、環境変数 VCPKG_DEFAULT_TRIPLET で設定.
if(NOT DEFINED VCPKG_TARGET_TRIPLET AND DEFINED ENV{VCPKG_DEFAULT_TRIPLET})
    set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(my-cpp-project CXX C)

add_subdirectory(test)
add_subdirectory(src)
