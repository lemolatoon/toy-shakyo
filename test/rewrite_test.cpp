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