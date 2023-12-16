#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
#include "toy/mlirGen.h"
#include "toy/parser.h"
#include "toy/passes.h"

#include "llvm/Support/raw_ostream.h"

#include "test_util.h"

#include "gtest/gtest.h"

std::optional<std::string> toySource2mlir(std::string_view toySource,
                                          bool enableOpt, LowerTo lowerTo) {
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

  mlir::PassManager pm(&context, moduleOp.get()->getName().getStringRef());
  if (enableOpt || lowerTo >= LowerTo::Affine) {
    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations. Add a run of the canonicalizer to optimize the mlir
    // module.
    mlir::OpPassManager &optPM = pm.nest<toy::FuncOp>();
    optPM.addPass(toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (lowerTo >= LowerTo::Affine) {
    // Partially lower the toy dialect.
    pm.addPass(toy::createLowerToAffinePass());

    // Add a few cleanups post lowering
    auto &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
  }

  if (mlir::failed(pm.run(*moduleOp)))
    return std::nullopt;

  auto buf = std::string{};
  auto stream = llvm::raw_string_ostream{buf};
  moduleOp->print(stream);

  return buf;
}