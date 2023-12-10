#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
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

  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();
  auto builder = mlir::OpBuilder(&context);
  auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

  // Create an MLIR function for the given prototype.
  builder.setInsertionPointToEnd(theModule.getBody());
  // all arguments are unranked tensors of f64
  llvm::SmallVector<mlir::Type, 4> argTypes(
      2, mlir::UnrankedTensorType::get(builder.getF64Type()));
  auto funcType = builder.getFunctionType(argTypes, std::nullopt);
  FuncOp func =
      builder.create<FuncOp>(builder.getUnknownLoc(), "main", funcType);

  // Let's start the body of the function now!
  mlir::Block &entryBlock = func.front();

  // Set the insertion point in the builder to the beginning of the function
  // body, it will be used throughout the codegen to create operations in this
  // function.
  builder.setInsertionPointToStart(&entryBlock);

  // Body==============================================
  auto dataType = mlir::RankedTensorType::get({1, 2, 3}, builder.getF64Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(
      dataType, llvm::ArrayRef({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
  mlir::Value lhs = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                               dataType, dataAttribute);
  mlir::Value rhs = builder.create<ConstantOp>(builder.getUnknownLoc(),
                                               dataType, dataAttribute);
  mlir::Value added = builder.create<AddOp>(builder.getUnknownLoc(), lhs, rhs);

  ReturnOp ret = builder.create<ReturnOp>(builder.getUnknownLoc(), added);

  if (ret.hasOperand())
    func.setFunctionType(builder.getFunctionType(
        func.getFunctionType().getInputs(),
        mlir::UnrankedTensorType::get(builder.getF64Type())));
  // Body end==========================================
  if (mlir::failed(theModule.verify())) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }
  theModule.print(llvm::errs());
  return 0;
}