cmake_minimum_required(VERSION 3.16)

add_subdirectory(parser)
add_subdirectory(mlir)

target_sources(toyc PRIVATE toyc.cpp)
target_link_libraries(toyc parser mlir)

##=== LLVM ===##
include_directories(${LLVM_INCLUDE_DIRS})
target_link_directories(toyc PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(toyc ${llvm_libs})

##=== MLIR ===##
include_directories(${MLIR_INCLUDE_DIRS})
target_link_directories(toyc PRIVATE ${MLIR_LIBRARY_DIRS})
target_link_libraries(toyc ${mlir_libs})


target_include_directories(toyc PUBLIC ${PROJECT_SOURCE_DIR}/include)