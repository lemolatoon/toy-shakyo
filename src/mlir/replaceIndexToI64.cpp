#include "mlir/IR/BuiltinOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

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