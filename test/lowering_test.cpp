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