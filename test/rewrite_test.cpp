#include "test_util.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <iostream>

TEST(REWRITE, TransposeTranspose) {
  std::string_view toySource = R"(
def transpose_transpose(a) {
	return transpose(transpose(a));
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
}
)");
}

TEST(REWRITE, ReshapeReshape) {
  std::string_view toySource = R"(
def reshape_reshape(a) {
  var b<2, 3> = a;
  var c<3, 2> = b;
  return c;
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @reshape_reshape(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.reshape(%arg0 : tensor<*xf64>) to tensor<3x2xf64>
    toy.return %0 : tensor<3x2xf64>
  }
}
)");
}

TEST(REWRITE, ReshapeConstantFold) {
  std::string_view toySource = R"(
def reshape_constant_fold() {
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  return b;
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @reshape_constant_fold() -> tensor<*xf64> {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    toy.return %0 : tensor<2x3xf64>
  }
}
)");
}