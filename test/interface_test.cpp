#include "test_util.h"
#include "gtest/gtest.h"

TEST(Interface, Inliner) {
  std::string_view toySource = R"(
	def hoge(a, b) {
		return transpose(a) * b;
	}
	def main() {
		var a = hoge([1, 2], [2, 4]);
		print(a);
	}
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main() {
    %0 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>} : () -> tensor<2xf64>
    %1 = "toy.constant"() {value = dense<[2.000000e+00, 4.000000e+00]> : tensor<2xf64>} : () -> tensor<2xf64>
    %2 = toy.transpose(%0 : tensor<2xf64>) to tensor<2xf64>
    %3 = "toy.mul"(%2, %1) : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    toy.print %3 : tensor<2xf64>
    toy.return
  }
}
)");
}

TEST(Interface, ShapeInference) {
  std::string_view toySource = R"(
	def main() {
		var a = [[1, 2, 3], [4, 5, 6]];
		var b = transpose(a);
		var c = a * b + [[100, 200], [300, 400], [500, 600]];
		print(c);
	}
)";
  auto ir = toySource2mlir(toySource, true);
  ASSERT_TRUE(ir.has_value());

  EXPECT_EQ(ir.value(), R"(module {
  toy.func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
    %2 = "toy.mul"(%0, %1) : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x3xf64>
    %3 = "toy.constant"() {value = dense<[[1.000000e+02, 2.000000e+02], [3.000000e+02, 4.000000e+02], [5.000000e+02, 6.000000e+02]]> : tensor<3x2xf64>} : () -> tensor<3x2xf64>
    %4 = "toy.add"(%2, %3) : (tensor<2x3xf64>, tensor<3x2xf64>) -> tensor<2x3xf64>
    toy.print %4 : tensor<2x3xf64>
    toy.return
  }
}
)");
}