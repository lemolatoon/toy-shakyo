#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
#include "toy/mlirGen.h"
#include "toy/parser.h"

#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

std::optional<std::string> toySource2mlir(std::string_view toySource,
                                          bool enableOpt = false) {
  auto lexer = LexerBuffer(toySource.begin(), toySource.end(), "sample.toy");
  auto parser = Parser(lexer);
  auto moduleAST = parser.parseModule();
  // not null
  if (!moduleAST)
    return std::nullopt;

  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();

  auto moduleOp = mlirGen(context, *moduleAST);
  if (!moduleOp)
    return std::nullopt;
  if (enableOpt) {
    mlir::PassManager pm(&context, moduleOp.get()->getName().getStringRef());

    // Add a run of the canonicalizer to optimize the mlir module.
    pm.addNestedPass<FuncOp>(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createInlinerPass());
    if (mlir::failed(pm.run(*moduleOp))) {
      llvm::errs() << "Failed to run canonicalizer\n";
      return std::nullopt;
    }
  }
  auto buf = std::string{};
  auto stream = llvm::raw_string_ostream{buf};
  moduleOp->print(stream);

  return buf;
}