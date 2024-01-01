#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"

#include "toy/passes.h"

namespace {
struct GpuReplaceMemoryAllocationPass
    : public mlir::PassWrapper<GpuReplaceMemoryAllocationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GpuReplaceMemoryAllocationPass)
private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::gpu::GPUDialect, mlir::memref::MemRefDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void GpuReplaceMemoryAllocationPass::runOnOperation() {
  auto f = getOperation();
  auto &context = *f.getContext();
  auto &funcRegion = f.getRegion();
  auto builder = mlir::OpBuilder(funcRegion);
  auto funcBlock = &funcRegion.front();
  f.walk([&](mlir::Operation *op) {
    if (!op) {
      return;
    }
    auto launchOp = llvm::dyn_cast<mlir::gpu::LaunchOp>(op);
    if (!launchOp) {
      return;
    }
    auto &block = launchOp.getBody();
    block.walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        auto definingOp = operand.getDefiningOp();
        if (!definingOp)
          continue;
        if (auto memrefAllocOp =
                llvm::dyn_cast<mlir::memref::AllocOp>(definingOp)) {
          std::optional<mlir::gpu::AllocOp> gpuMemory = std::nullopt;
          for (auto &use : memrefAllocOp->getUses()) {
            if (use.getOwner()->getParentRegion() != &block) {
              // not in the gpu.launch region
              continue;
            }
            if (!gpuMemory.has_value()) {
              builder.setInsertionPointToStart(funcBlock);
              // insert a memcpy to ModuleOp's start
              auto gpuAllocOp = builder.create<mlir::gpu::AllocOp>(
                  block.getLoc(), memrefAllocOp.getType(),
                  /*asyncDependencies=*/mlir::ValueRange(),
                  /*dynamicSizes=*/mlir::ValueRange(),
                  /*symbolOperands=*/mlir::ValueRange(), /*hostShared=*/false);
              // insert a memcpy to right before the launchOp
              builder.setInsertionPoint(launchOp);
              builder.create<mlir::gpu::MemcpyOp>(
                  block.getLoc(),
                  /*asyncToken=*/mlir::Type(),
                  /*asyncDependencies=*/mlir::ValueRange(),
                  /*dst=*/gpuAllocOp.getMemref(),
                  /*src=*/memrefAllocOp.getMemref());
              builder.setInsertionPointAfter(launchOp);
              builder.create<mlir::gpu::MemcpyOp>(
                  block.getLoc(),
                  /*asyncToken=*/mlir::Type(),
                  /*asyncDependencies=*/mlir::ValueRange(),
                  /*dst=*/memrefAllocOp.getMemref(),
                  /*src=*/gpuAllocOp.getMemref());
              builder.setInsertionPoint(funcBlock->getTerminator());
              builder.create<mlir::gpu::DeallocOp>(
                  block.getLoc(),
                  /*asyncToken=*/mlir::Type(),
                  /*asyncDependencies=*/mlir::ValueRange(),
                  gpuAllocOp.getMemref());
              gpuMemory = gpuAllocOp;
            }
            use.set(gpuMemory.value().getMemref());
          }
        };
      }
    });
  });
}

std::unique_ptr<mlir::Pass> toy::createGpuReplaceAllocationPass() {
  return std::make_unique<GpuReplaceMemoryAllocationPass>();
}