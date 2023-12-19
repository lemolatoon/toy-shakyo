namespace mlir {
class RewritePatternSet;
} // namespace mlir

namespace toy {
void populateReplaceIndexToI64(mlir::RewritePatternSet &patterns);
} // namespace toy