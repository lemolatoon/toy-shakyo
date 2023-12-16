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

// Reshape(Reshape(x)) -> x
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
																		(ReshapeOp $arg)>;

// Pattern Match and Rewrite with Native Code Call

def ReshapeConstant : NativeCodeCall<"$0.reshape(llvm::cast<mlir::ShapedType>($1.getType()))">;
// Reshape(Constant(x)) -> Constant(x.reshape)
def ReshapeConstantFoldPattern : Pat<(ReshapeOp:$res (ConstantOp $arg)),
																			(ConstantOp (ReshapeConstant $arg, $res))>;

// Pattern Match and Rewrite with Constraints

def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
// Reshape(x) = x, where input and output shapes are identical
def RedundantReshapeOptPattern : Pat<
	(ReshapeOp:$res $arg),
	(replaceWithValue $arg),
	[(TypesAreIdentical $arg, $res)]
>;

#endif // TOY_COMBINE