#include "test_util.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <iostream>

TEST(REWRITE, TransposeTranspose) {
  std::string_view toySource = R"(
def main(a) {
	return transpose(transpose(a));
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    toy.return %arg0 : tensor<*xf64>
  }
}
)");
}

TEST(REWRITE, ReshapeReshape) {
  std::string_view toySource = R"(
def main(a) {
  var b<2, 3> = a;
  var c<3, 2> = b;
  return c;
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.reshape(%arg0 : tensor<*xf64>) to tensor<3x2xf64>
    toy.return %0 : tensor<3x2xf64>
  }
}
)");
}

TEST(REWRITE, ReshapeConstantFold) {
  std::string_view toySource = R"(
def main() {
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  return b;
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main() -> tensor<*xf64> {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    toy.return %0 : tensor<2x3xf64>
  }
}
)");
}

TEST(REWRITE, RedundantReshape) {
  std::string_view toySource = R"(
def main() {
  # reshape constant fold====
  var a<2, 1> = [1, 2];
  var b<2, 1> = a;
  # =========================
  # reshape reshape, redundant reshape====
  var c<2, 1> = b;
  var d<2, 1> = c;
  # ======================================
  print(d);
} 
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  // TODO: with inlining, we can compose more appropriate example
  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>} : () -> tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
)");
}