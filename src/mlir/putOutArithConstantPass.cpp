#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "toy/passes.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace {
struct PutOutArithConstantPass
    : public mlir::PassWrapper<PutOutArithConstantPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  }
  void runOnOperation() final;

private:
  // Put `mlir::arith::ConstantOp` in `scf.parallel`'s body to the outside of
  // `scf.parallel`. This will be able to avoid segmentation fault by
  // `--convert-parallel-loops-to-gpu` or
  // `mlir::convertParallelLoopsToGPUPass()`.
  //
  // `precondition`: `op` is in `scf.parallel`.
  // `op->getParentOfType<mlir::scf::ParallelOp>()` is not `nullptr`.
  static mlir::LogicalResult putOutArithConstant(mlir::Operation *op) {
    auto parallelOp = op->getParentOfType<mlir::scf::ParallelOp>();

    // This `mlir::arith::ConstantOp` is not in `scf.parallel`.
    if (!parallelOp)
      return mlir::success();

    parallelOp.moveOutOfLoop(op);

    return mlir::success();
  }
};
} // namespace

void PutOutArithConstantPass::runOnOperation() {
  mlir::func::FuncOp f = getOperation();

  llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
  f.walk([&](mlir::Operation *op) {
    if (auto constantOp = llvm::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (constantOp->getParentOfType<mlir::scf::ParallelOp>())
        opWorklist.insert(op);
    }
  });

  while (!opWorklist.empty()) {
    auto constantOp = *opWorklist.begin();
    if (mlir::failed(putOutArithConstant(constantOp))) {
      signalPassFailure();
      return;
    }
    if (!constantOp->getParentOfType<mlir::scf::ParallelOp>()) {
      opWorklist.erase(constantOp);
    }
  }
}

std::unique_ptr<mlir::Pass> toy::createPutOutArithConstantPass() {
  return std::make_unique<PutOutArithConstantPass>();
}