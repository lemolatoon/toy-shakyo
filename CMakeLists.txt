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

# LLVM_DIR が無指定なら、llvm-config-16 で設定.
if(NOT DEFINED OVERWRITE_LLVM_DIR)
    find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-16 llvm-config)
    if(LLVM_CONFIG_EXECUTABLE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --cmakedir OUTPUT_VARIABLE LLVM_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
        message(STATUS "Set LLVM_DIR: ${LLVM_DIR}")
    else()
        message(FATAL_ERROR "llvm-config-16 not found")
    endif()
else()
    set(LLVM_DIR "${OVERWRITE_LLVM_DIR}")
    message(STATUS "Overwritten LLVM_DIR: ${LLVM_DIR}")
endif()

# MLIR_DIR が無指定なら、llvm-config-16 で設定.
if(NOT DEFINED OVERWRITE_MLIR_DIR)
    find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config-16 llvm-config)
    if(LLVM_CONFIG_EXECUTABLE)
        execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE MLIR_DIR_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(MLIR_DIR "${MLIR_DIR_PREFIX}/lib/cmake/mlir")
        message(STATUS "Set MLIR_DIR: ${MLIR_DIR}")
    else()
        message(FATAL_ERROR "llvm-config-16 not found")
    endif()
else()
    set(MLIR_DIR "${OVERWRITE_MLIR_DIR}")
    message(STATUS "Overwritten MLIR_DIR: ${MLIR_DIR}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(my-cpp-project CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(test)
add_subdirectory(include)
add_subdirectory(src)
