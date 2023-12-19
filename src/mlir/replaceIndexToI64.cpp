#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "toy/passes.h"

namespace {
struct ReplaceIndexToI64Pattern
    : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                                      mlir::PatternRewriter &rewriter);
};
} // namespace

mlir::LogicalResult
ReplaceIndexToI64Pattern::matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                                          mlir::PatternRewriter &rewriter) {
  llvm::errs() << "ReplaceIndexToI64Pattern::matchAndRewrite: " << op << "\n";
  if (op.getOperands().size() != 1)
    return mlir::failure();
  if (op.getOperand(0).getType() != rewriter.getI64Type())
    return mlir::failure();
  // op.getResult(0).setType(rewriter.getI64Type());
  rewriter.replaceOp(op, op.getOperand(0));
  return mlir::success();
}

namespace toy {
void populateReplaceIndexToI64(mlir::RewritePatternSet &patterns) {
  patterns.add<ReplaceIndexToI64Pattern>(patterns.getContext());
}

} // namespace toy
namespace {
struct ReplaceWithIndexCastsPass
    : public mlir::PassWrapper<ReplaceWithIndexCastsPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReplaceWithIndexCastsPass)
private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::index::IndexDialect>();
  }
  void runOnOperation() final;

private:
  static mlir::LogicalResult replaceWithIndexCasts(
      mlir::UnrealizedConversionCastOp unrealizedConversionCastOp) {

    if (unrealizedConversionCastOp->getOperands().size() != 1 ||
        unrealizedConversionCastOp->getResults().size() != 1)
      return mlir::failure();

    auto i64Type =
        mlir::IntegerType::get(unrealizedConversionCastOp.getContext(), 64);
    auto indexType =
        mlir::IndexType::get(unrealizedConversionCastOp.getContext());
    auto opTy = unrealizedConversionCastOp->getOperand(0).getType();
    auto resTy = unrealizedConversionCastOp->getResult(0).getType();
    auto ok1 = (opTy == indexType && resTy == i64Type);
    auto ok2 = (opTy == i64Type && resTy == indexType);
    if (!ok1 && !ok2)
      return mlir::failure();
    mlir::OpBuilder builder(unrealizedConversionCastOp);
    auto castOp = builder.create<mlir::index::CastSOp>(
        unrealizedConversionCastOp.getLoc(), i64Type,
        unrealizedConversionCastOp->getOperand(0));
    unrealizedConversionCastOp.replaceAllUsesWith(castOp);
    llvm::errs() << "castOp: " << castOp << "\n";

    return mlir::success();
  }
};
} // namespace

void ReplaceWithIndexCastsPass::runOnOperation() {
  auto f = getOperation();

  llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
  f.walk([&](mlir::Operation *op) {
    if (op)
      if (auto unrealizedConversionCastOp =
              llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
        if (mlir::failed(replaceWithIndexCasts(unrealizedConversionCastOp)))
          return;
      }
  });
}

std::unique_ptr<mlir::Pass> toy::createReplaceWithIndexCastsPass() {
  return std::make_unique<ReplaceWithIndexCastsPass>();
}