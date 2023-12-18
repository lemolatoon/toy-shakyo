#include "test_util.h"

#include "gtest/gtest.h"

TEST(LOWERING, ToyToAffine) {
  std::string_view toySource = R"(
	def mul_transpose(a, b) {
		return transpose(a) * transpose(b);
	}

	def main() {
		var a = [[1, 2, 3], [4, 5, 6]];
		var b = [[100, 200, 300], [400, 500, 600]];
		var c = mul_transpose(a, b);
		print(c + a);
	}
)";
  auto ir = toySource2mlir(toySource, true, LowerTo::Affine);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  func.func @main() {
    %cst = arith.constant 6.000000e+02 : f64
    %cst_0 = arith.constant 5.000000e+02 : f64
    %cst_1 = arith.constant 4.000000e+02 : f64
    %cst_2 = arith.constant 3.000000e+02 : f64
    %cst_3 = arith.constant 2.000000e+02 : f64
    %cst_4 = arith.constant 1.000000e+02 : f64
    %cst_5 = arith.constant 6.000000e+00 : f64
    %cst_6 = arith.constant 5.000000e+00 : f64
    %cst_7 = arith.constant 4.000000e+00 : f64
    %cst_8 = arith.constant 3.000000e+00 : f64
    %cst_9 = arith.constant 2.000000e+00 : f64
    %cst_10 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<3x2xf64>
    %alloc_11 = memref.alloc() : memref<2x3xf64>
    %alloc_12 = memref.alloc() : memref<2x3xf64>
    affine.store %cst_10, %alloc_12[0, 0] : memref<2x3xf64>
    affine.store %cst_9, %alloc_12[0, 1] : memref<2x3xf64>
    affine.store %cst_8, %alloc_12[0, 2] : memref<2x3xf64>
    affine.store %cst_7, %alloc_12[1, 0] : memref<2x3xf64>
    affine.store %cst_6, %alloc_12[1, 1] : memref<2x3xf64>
    affine.store %cst_5, %alloc_12[1, 2] : memref<2x3xf64>
    affine.store %cst_4, %alloc_11[0, 0] : memref<2x3xf64>
    affine.store %cst_3, %alloc_11[0, 1] : memref<2x3xf64>
    affine.store %cst_2, %alloc_11[0, 2] : memref<2x3xf64>
    affine.store %cst_1, %alloc_11[1, 0] : memref<2x3xf64>
    affine.store %cst_0, %alloc_11[1, 1] : memref<2x3xf64>
    affine.store %cst, %alloc_11[1, 2] : memref<2x3xf64>
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 2 {
        %0 = affine.load %alloc_12[%arg1, %arg0] : memref<2x3xf64>
        %1 = affine.load %alloc_11[%arg1, %arg0] : memref<2x3xf64>
        %2 = arith.mulf %0, %1 : f64
        %3 = affine.load %alloc_12[%arg0, %arg1] : memref<2x3xf64>
        %4 = arith.addf %2, %3 : f64
        affine.store %4, %alloc[%arg0, %arg1] : memref<3x2xf64>
      }
    }
    toy.print %alloc : memref<3x2xf64>
    memref.dealloc %alloc_12 : memref<2x3xf64>
    memref.dealloc %alloc_11 : memref<2x3xf64>
    memref.dealloc %alloc : memref<3x2xf64>
    return
  }
}
)");
}

TEST(Lowering, GPU) {
  std::string_view toySource = R"(
  def main() {
    print([[[1]]] + [[[2]]]);
  }
)";
  auto ir = toySource2mlir(toySource, true, LowerTo::Affine, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(),
            R"(#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
module {
  func.func @main() {
    %cst = arith.constant 2.000000e+00 : f64
    %cst_0 = arith.constant 1.000000e+00 : f64
    %alloc = memref.alloc() : memref<1x1x1xf64>
    %alloc_1 = memref.alloc() : memref<1x1x1xf64>
    %alloc_2 = memref.alloc() : memref<1x1x1xf64>
    %c0 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    memref.store %cst_0, %alloc_2[%c0, %c0_3, %c0_4] : memref<1x1x1xf64>
    %c0_5 = arith.constant 0 : index
    %c0_6 = arith.constant 0 : index
    %c0_7 = arith.constant 0 : index
    memref.store %cst, %alloc_1[%c0_5, %c0_6, %c0_7] : memref<1x1x1xf64>
    %c0_8 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_9 = arith.constant 1 : index
    %c0_10 = arith.constant 0 : index
    %c1_11 = arith.constant 1 : index
    %c1_12 = arith.constant 1 : index
    %c0_13 = arith.constant 0 : index
    %c1_14 = arith.constant 1 : index
    %c1_15 = arith.constant 1 : index
    %c1_16 = arith.constant 1 : index
    %0 = affine.apply #map(%c1)[%c0_8, %c1_9]
    %1 = affine.apply #map(%c1_11)[%c0_10, %c1_12]
    gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %0, %arg7 = %c1_16, %arg8 = %c1_16) threads(%arg3, %arg4, %arg5) in (%arg9 = %1, %arg10 = %c1_16, %arg11 = %c1_16) {
      %2 = affine.apply #map1(%arg0)[%c1_9, %c0_8]
      %3 = affine.apply #map1(%arg3)[%c1_12, %c0_10]
      scf.for %arg12 = %c0_13 to %c1_14 step %c1_15 {
        %4 = memref.load %alloc_2[%2, %3, %arg12] : memref<1x1x1xf64>
        %5 = memref.load %alloc_1[%2, %3, %arg12] : memref<1x1x1xf64>
        %6 = arith.addf %4, %5 : f64
        memref.store %6, %alloc[%2, %3, %arg12] : memref<1x1x1xf64>
      }
      gpu.terminator
    } {SCFToGPU_visited}
    toy.print %alloc : memref<1x1x1xf64>
    memref.dealloc %alloc_2 : memref<1x1x1xf64>
    memref.dealloc %alloc_1 : memref<1x1x1xf64>
    memref.dealloc %alloc : memref<1x1x1xf64>
    return
  }
}
)");
}

TEST(Lowering, LLVM) {
  std::string_view toySource = R"(
  def main() {
    print([[1, 2], [3, 4], [5, 6]]);
  }
)";
  auto ir = toySource2mlir(toySource, true, LowerTo::LLVM);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  llvm.func @free(!llvm.ptr<i8>)
  llvm.mlir.global internal constant @newLine("\0A\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @formatSpecifier("%f \00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @main() {
    %0 = llvm.mlir.constant(6.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.mlir.constant(4.000000e+00 : f64) : f64
    %3 = llvm.mlir.constant(3.000000e+00 : f64) : f64
    %4 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64
    %6 = llvm.mlir.constant(3 : index) : i64
    %7 = llvm.mlir.constant(2 : index) : i64
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.mlir.constant(6 : index) : i64
    %10 = llvm.mlir.null : !llvm.ptr<f64>
    %11 = llvm.getelementptr %10[%9] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %12 = llvm.ptrtoint %11 : !llvm.ptr<f64> to i64
    %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr<i8>
    %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<f64>
    %15 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
    %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(0 : index) : i64
    %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.insertvalue %6, %19[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %21 = llvm.insertvalue %7, %20[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.insertvalue %7, %21[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.insertvalue %8, %22[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(0 : index) : i64
    %25 = llvm.mlir.constant(0 : index) : i64
    %26 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.mlir.constant(2 : index) : i64
    %28 = llvm.mul %24, %27  : i64
    %29 = llvm.add %28, %25  : i64
    %30 = llvm.getelementptr %26[%29] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %5, %30 : !llvm.ptr<f64>
    %31 = llvm.mlir.constant(0 : index) : i64
    %32 = llvm.mlir.constant(1 : index) : i64
    %33 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(2 : index) : i64
    %35 = llvm.mul %31, %34  : i64
    %36 = llvm.add %35, %32  : i64
    %37 = llvm.getelementptr %33[%36] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %4, %37 : !llvm.ptr<f64>
    %38 = llvm.mlir.constant(1 : index) : i64
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.mlir.constant(2 : index) : i64
    %42 = llvm.mul %38, %41  : i64
    %43 = llvm.add %42, %39  : i64
    %44 = llvm.getelementptr %40[%43] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %3, %44 : !llvm.ptr<f64>
    %45 = llvm.mlir.constant(1 : index) : i64
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(2 : index) : i64
    %49 = llvm.mul %45, %48  : i64
    %50 = llvm.add %49, %46  : i64
    %51 = llvm.getelementptr %47[%50] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %2, %51 : !llvm.ptr<f64>
    %52 = llvm.mlir.constant(2 : index) : i64
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mlir.constant(2 : index) : i64
    %56 = llvm.mul %52, %55  : i64
    %57 = llvm.add %56, %53  : i64
    %58 = llvm.getelementptr %54[%57] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %1, %58 : !llvm.ptr<f64>
    %59 = llvm.mlir.constant(2 : index) : i64
    %60 = llvm.mlir.constant(1 : index) : i64
    %61 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %62 = llvm.mlir.constant(2 : index) : i64
    %63 = llvm.mul %59, %62  : i64
    %64 = llvm.add %63, %60  : i64
    %65 = llvm.getelementptr %61[%64] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    llvm.store %0, %65 : !llvm.ptr<f64>
    %66 = llvm.mlir.addressof @formatSpecifier : !llvm.ptr<array<4 x i8>>
    %67 = llvm.mlir.constant(0 : index) : i64
    %68 = llvm.getelementptr %66[%67, %67] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr, i8
    %69 = llvm.mlir.addressof @newLine : !llvm.ptr<array<2 x i8>>
    %70 = llvm.mlir.constant(0 : index) : i64
    %71 = llvm.getelementptr %69[%70, %70] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr, i8
    %72 = llvm.mlir.constant(0 : index) : i64
    %73 = llvm.mlir.constant(3 : index) : i64
    %74 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb1(%72 : i64)
  ^bb1(%75: i64):  // 2 preds: ^bb0, ^bb5
    %76 = llvm.icmp "slt" %75, %73 : i64
    llvm.cond_br %76, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %77 = llvm.mlir.constant(0 : index) : i64
    %78 = llvm.mlir.constant(2 : index) : i64
    %79 = llvm.mlir.constant(1 : index) : i64
    llvm.br ^bb3(%77 : i64)
  ^bb3(%80: i64):  // 2 preds: ^bb2, ^bb4
    %81 = llvm.icmp "slt" %80, %78 : i64
    llvm.cond_br %81, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %82 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %83 = llvm.mlir.constant(2 : index) : i64
    %84 = llvm.mul %75, %83  : i64
    %85 = llvm.add %84, %80  : i64
    %86 = llvm.getelementptr %82[%85] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
    %87 = llvm.load %86 : !llvm.ptr<f64>
    %88 = llvm.call @printf(%68, %87) : (!llvm.ptr, f64) -> i32
    %89 = llvm.add %80, %79  : i64
    llvm.br ^bb3(%89 : i64)
  ^bb5:  // pred: ^bb3
    %90 = llvm.call @printf(%71) : (!llvm.ptr) -> i32
    %91 = llvm.add %75, %74  : i64
    llvm.br ^bb1(%91 : i64)
  ^bb6:  // pred: ^bb1
    %92 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = llvm.bitcast %92 : !llvm.ptr<f64> to !llvm.ptr<i8>
    llvm.call @free(%93) : (!llvm.ptr<i8>) -> ()
    llvm.return
  }
}
)");
}