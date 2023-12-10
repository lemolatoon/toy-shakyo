.PHONY: configure build run test clean FORCE
	
CLANG_FORMAT=clang-format-14
MLIR_TBLGEN=mlir-tblgen-16

LLVM_DIR=$(shell llvm-config-16 --prefix)
MLIR_INCLUDE_DIR=$(LLVM_DIR)/include/
LLVM_CMAKE_DIR=$(shell llvm-config-16 --cmakedir)
MLIR_CMAKE_DIR=$(shell llvm-config-16 --prefix)/lib/cmake/mlir
CMAKE_ARGS+=-DCMAKE_TOOLCHAIN_FILE=$(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake \
-DOVERWRITE_LLVM_DIR=$(LLVM_CMAKE_DIR) \
-DOVERWRITE_MLIR_DIR=$(MLIR_CMAKE_DIR)

build:
	ninja -C build

configure: FORCE
	cmake -GNinja -S . -B build $(CMAKE_ARGS) $(CMAKE_EXTRA_ARGS)

gen: FORCE
	$(MLIR_TBLGEN) -gen-dialect-decls include/toy/ops.td -I $(MLIR_INCLUDE_DIR) > include/toy/dialect.h.inc
	$(MLIR_TBLGEN) -gen-dialect-defs include/toy/ops.td -I $(MLIR_INCLUDE_DIR) > include/toy/dialect.cpp.inc
	$(MLIR_TBLGEN) -gen-op-decls include/toy/ops.td -I $(MLIR_INCLUDE_DIR) > include/toy/ops.h.inc
	$(MLIR_TBLGEN) -gen-op-defs include/toy/ops.td -I $(MLIR_INCLUDE_DIR) > include/toy/ops.cpp.inc

# example: 
#  make ARGS="sampels/smaple.toy --emit=ast" 
#  make ARGS="sampels/smaple.toy --emit=mlir"
run: build
	./build/src/a.out $(ARGS)

SRC=samples/sample.toy
ast: build
	./build/src/a.out $(SRC) --emit=ast
mlir: build
	./build/src/a.out $(SRC) --emit=mlir

test: build
	./build/test/googleTest

clean: FORCE
	rm -rf build
	rm -rf vcpkg_installed
	rm -rf .cache

fmt: FORCE
	find src -name "*.cpp" | xargs $(CLANG_FORMAT) -i
	find test -name "*.cpp" | xargs $(CLANG_FORMAT) -i
	find include -name "*.h" | xargs $(CLANG_FORMAT) -i