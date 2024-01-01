.PHONY: configure build run test clean FORCE
	
CLANG_FORMAT=clang-format-14
MLIR_TBLGEN=mlir-tblgen-16

INCLUDE=$(shell pwd)/include/

LLVM_DIR=$(shell llvm-config-16 --prefix)
include .env

MLIR_INCLUDE_DIR=$(LLVM_DIR)/include/
LLVM_CMAKE_DIR=$(LLVM_DIR)/lib/cmake/llvm
MLIR_CMAKE_DIR=$(LLVM_DIR)/lib/cmake/mlir
CMAKE_ARGS+=-DCMAKE_TOOLCHAIN_FILE=$(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake \
-DOVERWRITE_LLVM_DIR=$(LLVM_CMAKE_DIR) \
-DOVERWRITE_MLIR_DIR=$(MLIR_CMAKE_DIR)

build:
	ninja -C build

configure: FORCE
	cmake -GNinja -S . -B build $(CMAKE_ARGS) $(CMAKE_EXTRA_ARGS)

gen: FORCE
	$(MLIR_TBLGEN) -gen-dialect-decls include/toy/ops.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/dialect.h.inc
	$(MLIR_TBLGEN) -gen-dialect-defs include/toy/ops.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/dialect.cpp.inc
	$(MLIR_TBLGEN) -gen-op-decls include/toy/ops.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/ops.h.inc
	$(MLIR_TBLGEN) -gen-op-defs include/toy/ops.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/ops.cpp.inc
	$(MLIR_TBLGEN) -gen-rewriters src/mlir/toyCombine.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > src/mlir/toyCombine.cpp.inc
	$(MLIR_TBLGEN) -gen-op-interface-decls include/toy/shapeInferenceInterface.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/shapeInferenceInterface.h.inc
	$(MLIR_TBLGEN) -gen-op-interface-defs include/toy/shapeInferenceInterface.td -I $(MLIR_INCLUDE_DIR) -I $(INCLUDE) > include/toy/shapeInferenceInterface.cpp.inc

TOYC=./build/toyc
TEST=./build/googleTest
# example: 
#  make ARGS="sampels/smaple.toy --emit=ast" 
#  make ARGS="sampels/smaple.toy --emit=mlir"
run: build
	$(TOYC) $(ARGS)

SRC=samples/sample.toy
ARG=-opt
ast: build
	$(TOYC) $(SRC) --emit=ast $(ARG)
mlir: build
	$(TOYC) $(SRC) --emit=mlir $(ARG)
mlir-affine: build
	$(TOYC) $(SRC) --emit=mlir-affine $(ARG)
mlir-llvm: build
	$(TOYC) $(SRC) --emit=mlir-llvm $(ARG)
llvm: build
	$(TOYC) $(SRC) --emit=llvm $(ARG)

test: build
	$(TEST)

clean: FORCE
	rm -rf build
	rm -rf vcpkg_installed
	rm -rf .cache

fmt: FORCE
	find src -name "*.cpp" | xargs $(CLANG_FORMAT) -i
	find test -name "*.cpp" | xargs $(CLANG_FORMAT) -i
	find include -name "*.h" | xargs $(CLANG_FORMAT) -i
