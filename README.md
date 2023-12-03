# cpp-template
C++20環境を vcpkg + CMake + make (+ google test)で構築したい人向けのテンプレートリポジトリです。

必要な環境構築やトラブルシューティングは[こちら](https://github.com/lemolatoon/cpp-json-parser#%E7%92%B0%E5%A2%83%E6%A7%8B%E7%AF%89)

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