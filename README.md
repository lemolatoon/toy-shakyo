# toy-shakyo
C++20環境を vcpkg + CMake + make (+ google test)で構築したい人向けのテンプレートリポジトリです。

必要な環境構築やトラブルシューティングは[こちら](https://github.com/lemolatoon/cpp-json-parser#%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89)

## 使いかた。
.envにbuildしたLLVMをinstallしたディレクトリを書く。

.envの例
```.env
LLVM_DIR=/opt/llvm/16.0.6/
```
toy-shakyoのbuild
```bash
# TableGenから.cppファイルと.hファイルを生成
$ make gen
# configure
$ make configure
# build
$ make build
```
これで、`./build/toyc`と`./build/googleTest`の２つの実行バイナリができる。

### toycの使い方
```bash
USAGE: toyc [options] <input toy file>
```
例
```bash
$ ./build/toyc samples/sample3.toy --emit=llvm -opt -gpu 2&> sample3.ir
$ ./build/toyc --help
# MLIR の遷移を見る
$ ./build/toyc samples/sample3.toy --emit=llvm -opt -gpu --mlir-print-ir-after-all 2&> all.mlir
```
mlir標準で追加されているオプションがたくさんあるので、とりあえず`--emit=<value>`と、`-opt`と`-gpu`を使えば良い。

### LLVM IRのコンパイルと実行
`sample3.ir`にLLVM IRを出力した場合
```bash
# JITによる実行
$ lli sample3.ir
# 実行ファイルにして実行
$ clang sample3.ir -o sample3
$ ./sample3
# gpuを使う場合。-L以降のパスは、LLVMをビルドしてインストールしたパスによって変わる。
# clangは少なくともバージョン16は必要。
$ clang-16 gpu.ll -lmlir_cuda_runtime -L/opt/llvm/16.0.6/lib -o sample3
$ LD_LIBRARY_PATH=/opt/llvm/16.0.6/lib ./sample3
```

## LLVMのinstall

### Ubuntu
https://apt.llvm.org/ を見てinstallする。
```bash
$ wget https://apt.llvm.org/llvm.sh
$ chmod +x llvm.sh
$ sudo ./llvm.sh 16
$ apt list | grep llvm-16
llvm-16/unknown,now 1:16.0.6~++20230710042046+7cbf1a259152-1~exp1~20230710162136.105 amd64 [installed,automatic]
$ sudo apt-get install llvm-16
$ llvm-config-16 --version
16.0.6
```

## mlir
```bash
$ apt list | grep mlir
libmlir-16/unknown 1:16.0.6~++20230710042046+7cbf1a259152-1~exp1~20230710162136.105 amd64
mlir-16-tools/unknown 1:16.0.6~++20230710042046+7cbf1a259152-1~exp1~20230710162136.105 amd64
$ sudo apt install libmlir-16-dev mlir-16-tools
```

### buildする場合
```
cmake -G Ninja ../llvm \
	-DLLVM_ENABLE_PROJECTS=mlir \
	-DLLVM_BUILD_EXAMPLES=ON \
	-DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
	-DCMAKE_BUILD_TYPE=Release \
	-DLLVM_ENABLE_LLD=ON \
	-DMLIR_INCLUDE_INTEGRATION_TESTS=ON \
	-DMLIR_ENABLE_CUDA_RUNNER=ON \
	-DBUILD_SHARED_LIBS=OFF \
	-DLLVM_LINK_LLVM_DYLIB=ON \
	-DLLVM_BUILD_LLVM_DYLIB=ON \
	-DCMAKE_C_COMPILER=clang-16 -DCMAKE_CXX_COMPILER=clang++-16 \
	-DCMAKE_INSTALL_PREFIX="/opt/llvm/16.0.6" \
	-DLLVM_ENABLE_RTTI=ON
```
```
cmake --build . --config RELEASE --target install
```

cuda関連で、undefined referenceが出るとき
https://discourse.llvm.org/t/mlir-build-error-when-enabling-cuda-runner/65880