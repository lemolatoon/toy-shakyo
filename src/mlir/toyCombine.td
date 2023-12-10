/// Defines language-specific pattern match optimization for Toy using
/// Declarative Rewrite Rules (DRR) specified using TableGen records.

#ifndef TOY_COMBINE
#define TOY_COMBINE

include "mlir/IR/PatternBase.td"
include "toy/ops.td"

/// The DRR definition used for defining pattern is shown below:
/// 
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///		 list<dag> supplementalPatterns = [],
///		 dag benefitsAdded = (addBenefit 0)
/// >;

// Basic Pattern-Match and Rewrite

// Transpose(Transpose(x)) -> x
def TransposeTransposeOptPattern : Pat<(TransposeOp(TransposeOp $arg)),
																			 (replaceWithValue $arg)>;

#endif // TOY_COMBINE