#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/dialect.h"
#include "toy/passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <memory>

// ToyToAffineLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public mlir::PassWrapper<ToyToAffineLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print` as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
      return llvm::isa<mlir::TensorType>(type);
    });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  // TODO: Add patterns here
  // patterns.add<>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
