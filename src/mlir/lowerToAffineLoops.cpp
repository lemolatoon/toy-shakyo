#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "toy/dialect.h"
#include "toy/passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <algorithm>
#include <memory>

// ToyToAffine RewritePatterns

/// Convert the given RankedTensorType into the corresponding MemRefType.
static mlir::MemRefType convertTensorToMemRef(mlir::RankedTensorType type) {
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static mlir::Value insertAllocAndDealloc(mlir::MemRefType type,
                                         mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block.
  // This is fine as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for iteration. It returns a value to store at the
/// current index of iteration.
using LoopIterationFn = mlir::function_ref<mlir::Value(
    mlir::OpBuilder &rewriter, mlir::ValueRange memRefOperands,
    mlir::ValueRange loopIvs)>;
static void lowerOpToLoops(mlir::Operation *op, mlir::ValueRange operands,
                           mlir::PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType =
      llvm::cast<mlir::RankedTensorType>((*op->result_type_begin()));
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  llvm::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  // N重for文を生成する。
  // [[1, 2, 3], [4, 5, 6]]のような配列の場合。
  // ```
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     // do something
  //   }
  // }
  // ```
  // という感じになる。
  // ivsはloop induction variablesの略。(i, j)のこと。
  mlir::buildAffineLoopNest(
      rewriter, loc, lowerBounds,
      /*upperbound=*/tensorType.getShape(), steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc,
          mlir::ValueRange ivs) {
        // Call the processing function with the rewriter,
        // the memref operands, and the loop induction
        // variables. This function will return the value
        // to store at the current index.
        auto valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<mlir::AffineStoreOp>(loc, valueToStore, alloc,
                                                  ivs);
      });
  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {

// ToyToAffine RewritePatterns: Binary operations
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          // https://mlir.llvm.org/docs/DefiningDialects/Operations/#operand-adaptors
          // Generate an adaptor for the remapped operands of the
          // BinaryOp. This allows for using the nice named accessors
          // that are generated by the ODS.
          typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the
          // inner loop.
          auto loadedLhs = builder.create<mlir::AffineLoadOp>(
              loc, binaryAdaptor.getLhs(), loopIvs);

          auto loadedRhs = builder.create<mlir::AffineLoadOp>(
              loc, binaryAdaptor.getRhs(), loopIvs);

          // Create the binary operation performed on the loaded
          // values.
          return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });

    return mlir::success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, mlir::arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, mlir::arith::MulFOp>;

// ToyToAffine RewritePatterns : Constant operations
struct ConstantOpLowering : public mlir::OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(toy::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<mlir::RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    llvm::SmallVector<mlir::Value, 8> constantIndices;

    // 使う可能性のあるindexをここで生成しておく。
    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end()))) {
        constantIndices.push_back(
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
      }
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    llvm::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.value_begin<mlir::FloatAttr>();
    std::function<void(int64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++),
            alloc, llvm::ArrayRef(indices));
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i < e; i++) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // ```
    // var a = [[1, 2, 3], [4, 5, 6]]
    // ```
    // のような配列の場合、
    // [] -> [0] -> [0, 1] -> [0] -> [0, 2] -> ...
    // のようにindicesが変動して、storeElementsが呼ばれる。
    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

// ToyToAffine RewritePatterns : Func operations

struct FuncOpLowering : public mlir::OpConversionPattern<toy::FuncOp> {
  using mlir::OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main") {
      return mlir::failure();
    }

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](mlir::Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

// ToyToAffine RewritePatterns : Print operations

struct PrintOpLowering : public mlir::OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });

    return mlir::success();
  }
};

// ToyToAffine RewritePatterns : Return operations
struct ReturnOpLowering : public mlir::OpRewritePattern<toy::ReturnOp> {
  using mlir::OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(toy::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all functions have been inlined.
    if (op.hasOperand()) {
      return mlir::failure();
    }
    // We lower "toy.return" directly to "func.return"
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op);
    return mlir::success();
  }
};

// ToyToAffine RewritePatterns : Transpose operations
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
              mlir::ValueRange loopIvs) {
          // https://mlir.llvm.org/docs/DefiningDialects/Operations/#operand-adaptors
          // Generate an adaptor for the remapped operands of the
          // TransposeOp. This allows for using the nice named
          // accessors that are generated by the ODS.
          toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.getInput();

          // Transpose the elements by generating a load from the reverse
          // indices.
          llvm::SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          // returnしたmlir::Valueは、loopIvsでstoreされる。
          return builder.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });

    return mlir::success();
  }
};

} // namespace

// ToyToAffineLoweringPass

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public mlir::PassWrapper<ToyToAffineLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::AffineDialect, mlir::func::FuncDialect,
                    mlir::memref::MemRefDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect,
                         mlir::arith::ArithDialect, mlir::func::FuncDialect,
                         mlir::memref::MemRefDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print` as `legal`. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(), [](mlir::Type type) {
      return llvm::isa<mlir::TensorType>(type);
    });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  // TODO: Add patterns here
  patterns.add<AddOpLowering, MulOpLowering, FuncOpLowering, PrintOpLowering,
               ReturnOpLowering, TransposeOpLowering, ConstantOpLowering>(
      &getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<mlir::Pass> toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
