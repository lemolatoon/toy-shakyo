#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
#include "toy/mlirGen.h"
#include "toy/parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <iostream>
#include <memory>
#include <vector>

int main() {
  std::string_view toySource = R"(
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);

  print(f);
}
)";

  auto lexer =
      LexerBuffer(toySource.begin(), toySource.end() - 1, "sample.toy");
  auto parser = Parser(lexer);
  auto module = parser.parseModule();
  if (!module) {
    std::cerr << "Failed to parse sample.toy" << std::endl;
    return 1;
  }
  toy::dump(*module);

  std::string_view toySource2 = R"(
    def main() {
      print([[1, 1], [1, 2]]);
      print(1 + (2 + 3));
      print([1, 2, 3] + [4, 5, 6]);
    }
  )";

  auto lexer2 =
      LexerBuffer(toySource2.begin(), toySource2.end() - 1, "sample2.toy");
  auto parser2 = Parser(lexer2);
  auto module2 = parser2.parseModule();
  if (!module2) {
    std::cerr << "Failed to parse sample.toy" << std::endl;
    return 1;
  }
  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();

  auto moduleOp = mlirGen(context, *module2);

  moduleOp->dump();

  return 0;
}