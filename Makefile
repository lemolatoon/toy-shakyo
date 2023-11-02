.PHONY: configure build run test clean FORCE
	
CLANG_FORMAT=clang-format-14

LLVM_CMAKE_DIR=$(shell llvm-config-16 --cmakedir)
CMAKE_ARGS+=-DCMAKE_TOOLCHAIN_FILE=$(VCPKG_ROOT)/scripts/buildsystems/vcpkg.cmake \
-DCMAKE_PREFIX_PATH=$(LLVM_CMAKE_DIR) \
-DLLVM_DIR=$(LLVM_CMAKE_DIR)

configure: FORCE
	cmake -GNinja -S . -B build $(CMAKE_ARGS) $(CMAKE_EXTRA_ARGS)

build:
	ninja -C build

run: build
	./build/src/a.out

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