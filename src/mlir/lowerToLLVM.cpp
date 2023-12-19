#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "toy/dialect.h"
#include "toy/passes.h"
#include "toy/replaceIndexToI64.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.
class PrintOpLowering : public mlir::ConversionPattern {
public:
  explicit PrintOpLowering(mlir::MLIRContext *context)
      : mlir::ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto memRefType = mlir::cast<mlir::MemRefType>(*op->operand_type_begin());
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    mlir::ModuleOp parentModule = op->getParentOfType<mlir::ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    const char *formatSpecifierKey = "formatSpecifier";
    const char *newLineKey = "newLine";
    mlir::Value formatSpecifierCst =
        getOrCreateGlobalString(loc, rewriter, formatSpecifierKey,
                                mlir::StringRef("%f \0", 4), parentModule);
    mlir::Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, newLineKey, mlir::StringRef("\n\0", 2), parentModule);
    auto printfType = getPrintfType(context);

    // Create a loop for each of the dimensions with in the shape.
    mlir::SmallVector<mlir::Value, 4> loopIvs;
    for (uint i = 0, e = memRefShape.size(); i < e; i++) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<mlir::scf::ForOp>(loc, lowerBound, upperBound, step);
      for (mlir::Operation &nested : *loop.getBody()) {
        // erase auto-generated yield
        rewriter.eraseOp(&nested);
      }
      loopIvs.push_back(loop.getInductionVar());
      // Terminate the loop body
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimension of the shape
      if (i != e - 1) {
        rewriter.create<mlir::LLVM::CallOp>(loc, printfType.getReturnTypes(),
                                            printfRef, newLineCst);
      }

      rewriter.create<mlir::scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Here, rewriter is now set the entry of innermost loop
    auto printOp = mlir::cast<toy::PrintOp>(op);
    auto elementLoad =
        rewriter.create<mlir::memref::LoadOp>(loc, printOp.getInput(), loopIvs);
    rewriter.create<mlir::LLVM::CallOp>(
        loc, printfType.getReturnTypes(), printfRef,
        mlir::ArrayRef<mlir::Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);

    return mlir::success();
  }

private:
  static mlir::LLVM::LLVMFunctionType
  getPrintfType(mlir::MLIRContext *context) {
    auto llvmI32Type = mlir::IntegerType::get(context, 32);
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(context);
    auto llvmPrintfType = mlir::LLVM::LLVMFunctionType::get(
        llvmI32Type, llvmPtrType, /*isVarArg=*/true);
    return llvmPrintfType;
  }

  static mlir::FlatSymbolRefAttr
  getOrInsertPrintf(mlir::PatternRewriter &rewriter, mlir::ModuleOp module) {
    auto *context = module.getContext();
    const char *printfSymbol = "printf";

    // Already inserted
    if (module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(printfSymbol)) {
      return mlir::SymbolRefAttr::get(context, printfSymbol);
    }

    // Create a function declaration for printf, the signature is:
    // * `i32 (i8*, ...)`
    auto llvmFnType = getPrintfType(context);

    // Insert the printf function into the body of the parent module.
    mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<mlir::LLVM::LLVMFuncOp>(module.getLoc(), printfSymbol,
                                            llvmFnType);

    return mlir::SymbolRefAttr::get(context, printfSymbol);
  }

  static mlir::Value getOrCreateGlobalString(mlir::Location loc,
                                             mlir::OpBuilder &builder,
                                             mlir::StringRef name,
                                             mlir::StringRef value,
                                             mlir::ModuleOp module) {
    mlir::LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
      mlir::OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = mlir::LLVM::LLVMArrayType::get(
          mlir::IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<mlir::LLVM::GlobalOp>(
          loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
          builder.getStringAttr(value), /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    mlir::Value globalPtr =
        builder.create<mlir::LLVM::AddressOfOp>(loc, global);
    mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getIndexAttr(0));

    // get element pointer
    auto llvmPtrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto gep = builder.create<mlir::LLVM::GEPOp>(
        loc, /*resultType=*/llvmPtrType, global.getType(), globalPtr,
        mlir::ArrayRef<mlir::Value>({/*base addr=*/cst0, /*index=*/cst0}));
    gep.setElemTypeAttr(mlir::TypeAttr::get(builder.getI8Type()));
    return gep;
  }
};
} // namespace

namespace {
struct ToyPrintOpLoweringPass
    : public mlir::PassWrapper<ToyPrintOpLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyPrintOpLoweringPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<toy::ToyDialect>();
  }

private:
  void runOnOperation() final;
};

struct ToyGPUToLLVMPass
    : public mlir::PassWrapper<ToyGPUToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyGPUToLLVMPass)
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<toy::ToyDialect>();
  }

private:
  void runOnOperation() final;
};

struct ToyToLLVMLoweringPass
    : public mlir::PassWrapper<ToyToLLVMLoweringPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
private:
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() final;
};
} // namespace

void ToyToLLVMLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering. For this lowering, we are only targeting
  // the LLVM dialect.
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();

  // During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  mlir::LLVMTypeConverter typeConverter(&getContext());

  // Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luchily,
  // there are already exists a set of patterns to transform `affine` and `std`
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into set
  // of legal ones.
  mlir::RewritePatternSet patterns(&getContext());
  // affine -> std(scf)
  mlir::populateAffineToStdConversionPatterns(patterns);
  // scf -> cf
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  // arith -> llvm
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  // memref -> llvm
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // cf -> llvm
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  // func -> llvm
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from `toy` dialect, is the PrintOp.
  patterns.add<PrintOpLowering>(&getContext());

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (mlir::failed(
          mlir::applyFullConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void ToyGPUToLLVMPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define
  // the final target for this lowering. For this lowering, we are only
  // targeting the LLVM dialect.
  mlir::LLVMConversionTarget target(getContext());
  target.addIllegalDialect<toy::ToyDialect>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect, mlir::gpu::GPUDialect>();

  mlir::RewritePatternSet patterns(&getContext());
  mlir::LLVMTypeConverter typeConverter(&getContext());

  // affine -> std(scf)
  mlir::populateAffineToStdConversionPatterns(patterns);
  // scf -> cf
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  // arith -> llvm
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

  // memref->llvm
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // cf -> llvm
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  // func -> llvm
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // gpu -> llvm
  mlir::populateGpuToLLVMConversionPatterns(typeConverter, patterns);
  toy::populateReplaceIndexToI64(patterns);

  // The only remaining operation to lower from `toy` dialect, is the PrintOp.
  patterns.add<PrintOpLowering>(&getContext());

  auto module = getOperation();
  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void ToyPrintOpLoweringPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  mlir::LLVMConversionTarget target(getContext());

  target.addIllegalDialect<toy::ToyDialect>();
  target.markUnknownOpDynamicallyLegal([](mlir::Operation *) { return true; });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<PrintOpLowering>(&getContext());

  if (mlir::failed(
          mlir::applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}

std::unique_ptr<mlir::Pass> toy::createLowerToLLVMWithGPUPass() {
  return std::make_unique<ToyGPUToLLVMPass>();
}

std::unique_ptr<mlir::Pass> toy::createPrintOpLoweringPass() {
  return std::make_unique<ToyPrintOpLoweringPass>();
}