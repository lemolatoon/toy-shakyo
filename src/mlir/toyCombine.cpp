#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/dialect.h"

namespace {
#include "toyCombine.cpp.inc"
}

/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void toy::TransposeOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.add<TransposeTransposeOptPattern>(context);
}