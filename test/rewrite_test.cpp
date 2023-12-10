#include "test_util.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

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