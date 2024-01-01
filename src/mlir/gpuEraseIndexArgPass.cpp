#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "toy/passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace {
struct GpuEraseIndexArgPass
    : public mlir::PassWrapper<GpuEraseIndexArgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuEraseIndexArgPass)
private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();
  }
  void runOnOperation() final;

private:
  static mlir::LogicalResult processOutOfKernel(mlir::Value arg) {
    auto prev = arg.getDefiningOp();
    if (!prev)
      return mlir::failure();
    auto unrealizedConversionCastOp =
        llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(prev);
    if (!unrealizedConversionCastOp ||
        unrealizedConversionCastOp->getOperands().size() != 1)
      return mlir::failure();
    auto prev2 = unrealizedConversionCastOp->getOperand(0).getDefiningOp();
    if (!prev2)
      return mlir::failure();
    auto constantOp = llvm::dyn_cast<mlir::LLVM::ConstantOp>(prev2);
    if (!constantOp || constantOp.getType() !=
                           mlir::IntegerType::get(constantOp.getContext(), 64))
      return mlir::failure();

    unrealizedConversionCastOp->replaceAllUsesWith(constantOp);
    unrealizedConversionCastOp->erase();

    return mlir::success();
  }
  static mlir::LogicalResult eraseArg(mlir::Operation *op) {
    auto &context = *op->getContext();
    auto i64Type = mlir::IntegerType::get(&context, 64);
    auto launchFuncOp = llvm::dyn_cast<mlir::gpu::LaunchFuncOp>(op);
    auto *block = launchFuncOp->getBlock();
    for (auto iter : llvm::enumerate(launchFuncOp.getOperands())) {
      auto arg = iter.value();
      llvm::errs() << "arg " << iter.index() << ": " << arg << "\n";
      if (mlir::failed(processOutOfKernel(iter.value())))
        continue;
    }
    return mlir::success();
  }
};
} // namespace

void GpuEraseIndexArgPass::runOnOperation() {
  mlir::ModuleOp f = getOperation();

  llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
  f.walk([&](mlir::Operation *op) {
    if (auto launchFuncOp = llvm::dyn_cast<mlir::gpu::LaunchFuncOp>(op)) {
      opWorklist.insert(op);
    }
  });

  while (!opWorklist.empty()) {
    auto constantOp = *opWorklist.begin();
    if (mlir::failed(eraseArg(constantOp))) {
      signalPassFailure();
      return;
    }

    opWorklist.erase(constantOp);
  }
}

std::unique_ptr<mlir::Pass> toy::createGpuEraseIndexArgPass() {
  return std::make_unique<GpuEraseIndexArgPass>();
}