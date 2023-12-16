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
    %2 = toy.cast %0 : tensor<2xf64> to tensor<*xf64>
    %3 = toy.cast %1 : tensor<2xf64> to tensor<*xf64>
    %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
    %5 = "toy.mul"(%4, %3) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}
)");
}