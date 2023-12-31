#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
#include "toy/mlirGen.h"
#include "toy/parser.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <iostream>
#include <vector>

#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "toy/passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
namespace {
enum InputType { Toy, MLIR };
} // namespace
static cl::opt<enum InputType> inputType(
    "x", cl::init(Toy), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(Toy, "toy", "load the input file as a Toy source.")),
    cl::values(clEnumValN(MLIR, "mlir",
                          "load the input file as an MLIR file")));
namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRAffine,
  DumpMLIRLLVM,
  DumpLLVMIR
};
} // namespace
static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
static cl::opt<bool> enableGpu("gpu",
                               cl::desc("Enable using gpu(use gpu dialect)"));

/// Returns a Toy AST resulting from parsing the file or a nullptr on
/// error
std::unique_ptr<toy::ModuleAST> parseInputFile(mlir::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrError =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (std::error_code ec = fileOrError.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrError.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(llvm::SourceMgr &sourceMgr, mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Handle '.toy' input to the compiler
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFilename).endswith(".mlir")) {
    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
      return 6;
    module = mlirGen(context, *moduleAST);
    if (!module)
      return 1;
    return 0;
  }

  // Otherwise, the input is '.mlir'
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  // Parse the input mlir
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return 3;
  }

  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Load our Dialect in this MLIR context
  context.getOrLoadDialect<toy::ToyDialect>();
  mlir::registerGpuSerializeToCubinPass();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerLLVMDialectTranslation(context);

  llvm::SourceMgr sourceMgr;
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  if (int error = loadMLIR(sourceMgr, context, module))
    return error;

  mlir::PassManager pm(&context, module.get()->getName().getStringRef());
  // Apply any generic pass manager command line options and run the pipeline.
  mlir::applyPassManagerCLOptions(pm);

  // Check to see waht granularity of MLIR we are compiling to.
  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToAffine) {
    // Apply any generic pass manager command line options and run the pipeline.
    mlir::applyPassManagerCLOptions(pm);

    pm.addPass(mlir::createInlinerPass());

    // Now that there is only one function, we can infer the shapes of each of
    // the operations. Add a run of the canonicalizer to optimize the mlir
    // module.
    mlir::OpPassManager &optPM = pm.nest<toy::FuncOp>();
    optPM.addPass(toy::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    // common sub expression elimination
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToAffine) {
    // Partially lower the toy dialect.
    pm.addPass(toy::createLowerToAffinePass());

    // Add a few cleanups post lowering
    auto &optPM = pm.nest<mlir::func::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
    // Add optimizations if enabled.
    if (enableOpt) {
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
  }

  if (enableGpu) {
    auto &gpuPM = pm.nest<mlir::func::FuncOp>();
    gpuPM.addPass(mlir::createAffineParallelizePass());
    // Affine to CFG to Gpu
    gpuPM.addPass(mlir::createLowerAffinePass());
    gpuPM.addPass(mlir::createGpuMapParallelLoopsPass());
    gpuPM.addPass(toy::createPutOutArithConstantPass());
    gpuPM.addPass(mlir::createParallelLoopToGpuPass());
    // Gpu Transformation
    // TODO: insert gpu.alloc and gpu.memcpy and gpu.dealloc
    gpuPM.addPass(toy::createGpuReplaceAllocationPass());
    pm.addPass(mlir::createMemRefToLLVMConversionPass());
    pm.addPass(toy::createReplaceWithIndexCastsPass());
    gpuPM.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(toy::createReplaceWithIndexCastsPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  if (isLoweringToLLVM) {
    if (enableGpu) {
      // device code lowering
      auto &gpuModulePM = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuModulePM.addPass(mlir::createLowerGpuOpsToNVVMOpsPass());

      std::string triple = "nvptx64-nvidia-cuda";
      std::string chip = "sm_75";
      std::string features = "+ptx60";
      LLVMInitializeNVPTXTarget();
      LLVMInitializeNVPTXTargetInfo();
      LLVMInitializeNVPTXTargetMC();
      LLVMInitializeNVPTXAsmPrinter();
      // Turing Architecture
      // https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
      // device code lowering
      pm.addPass(toy::createReplaceWithIndexCastsPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createConvertIndexToLLVMPass());
      auto &gpuModulePM2 = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuModulePM2.addPass(
          mlir::createGpuSerializeToCubinPass(triple, chip, features));
    }

    // Finish lowering the toy IR to the LLVM dialect.
    if (enableGpu) {
      // here we need to lower gpu_func's args to llvm
      pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuAsyncRegionPass());
      pm.addPass(mlir::createMemRefToLLVMConversionPass());
      pm.addPass(toy::createLowerToLLVMWithGPUPass());
      pm.addPass(toy::createReplaceWithIndexCastsPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createConvertIndexToLLVMPass());
      pm.addPass(mlir::createCanonicalizerPass());

      pm.addPass(mlir::createGpuToLLVMConversionPass());
      pm.addPass(toy::createReplaceWithIndexCastsPass());
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createConvertIndexToLLVMPass());
      pm.addPass(mlir::createCanonicalizerPass());
      // pm.addPass(mlir::createGpuToLLVMConversionPass());
      // pm.addPass(toy::createGpuEraseIndexArgPass());
    } else {
      pm.addPass(toy::createLowerToLLVMPass());
    }

    // This is necessary to have line tables emitted and basic
    // debugger working. In the future it will be added proper debug information
    // emission directly from their frontend.
    // NOTE: not exist on mlir 16?
    // pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
    //     mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;

  return 0;
}

int dumpAST() {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());
  mlir::registerGpuSerializeToCubinPass();
  // Translate the module, that contains the LLVM dialect, to LLVM IR.
  // Use a fresh LLVM IR context. (Note that LLVM is not thread-safe and any
  // concurrent use of a context requires external locking.)
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR" << err << "\n";
    return -1;
  }

  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int main(int argc, char **argv) {
  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");
  if (emitAction == Action::DumpAST) {
    return dumpAST();
  }

  mlir::MLIRContext context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module)) {
    return error;
  }

  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module.get()->dump();
    return 0;
  }

  if (emitAction == Action::DumpLLVMIR) {
    return dumpLLVMIR(module.get());
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}