#ifndef TOY_PASSES_H
#define TOY_PASSES_H

#include <memory>

namespace mlir {
class Pass;
}

namespace toy {
std::unique_ptr<mlir::Pass> createShapeInferencePass();

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> createLowerToAffinePass();

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

std::unique_ptr<mlir::Pass> createLowerToLLVMWithGPUPass();

std::unique_ptr<mlir::Pass> createPutOutArithConstantPass();

std::unique_ptr<mlir::Pass> createGpuEraseIndexArgPass();

std::unique_ptr<mlir::Pass> createPrintOpLoweringPass();

std::unique_ptr<mlir::Pass> createReplaceWithIndexCastsPass();

} // namespace toy

#endif // TOY_PASSES_H