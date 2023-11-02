#include "lib/hello.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include <iostream>
#include <memory>
#include <vector>

static std::unique_ptr<llvm::LLVMContext> the_context;
static std::unique_ptr<llvm::Module> the_module;
static std::unique_ptr<llvm::IRBuilder<>> builder;

int main() {
  // Initialize Module
  the_context = std::make_unique<llvm::LLVMContext>();
  the_module = std::make_unique<llvm::Module>("my cool jit", *the_context);

  builder = std::make_unique<llvm::IRBuilder<>>(*the_context);

  // main function
  //   auto sub_function = llvm::Function::Create(
  //       llvm::FunctionType::get(
  //           llvm::Type::getDoubleTy(*the_context),
  //           std::vector<llvm::Type *>{llvm::Type::getDoubleTy(*the_context)},
  //           false),
  //       llvm::Function::ExternalLinkage, "sub", the_module.get());

  //   sub_function->arg_begin()->setName("first_arg");

  //   auto basic_block =
  //       llvm::BasicBlock::Create(*the_context, "entry", sub_function);
  //   builder->SetInsertPoint(basic_block);

  //   auto val1 = llvm::ConstantFP::get(*the_context, llvm::APFloat(1.0));
  //   auto val3 = llvm::ConstantFP::get(*the_context, llvm::APFloat(3.0));

  //   auto lhs = builder->CreateFMul(val1, sub_function->arg_begin(),
  //   "multmp"); auto ret_val = builder->CreateFAdd(lhs, val3, "addtmp");
  //   builder->CreateRet(ret_val);

  //   llvm::verifyFunction(*sub_function);

  // ここから main 関数

  // declare printf function
  auto printf_type = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(*the_context),
      std::vector<llvm::Type *>{llvm::Type::getInt8PtrTy(*the_context)}, true);
  auto printf_func = llvm::Function::Create(
      printf_type, llvm::Function::ExternalLinkage, "printf", the_module.get());

  // main function
  auto main_function = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getInt32Ty(*the_context),
                              std::vector<llvm::Type *>{}, false),
      llvm::Function::ExternalLinkage, "main", the_module.get());

  auto basic_block2 =
      llvm::BasicBlock::Create(*the_context, "entry", main_function);
  builder->SetInsertPoint(basic_block2);

  auto hello_str = builder->CreateGlobalStringPtr("hello world!\n");
  builder->CreateCall(printf_func, hello_str);
  auto ret_val2 = llvm::ConstantInt::get(*the_context, llvm::APInt(32, 42));
  builder->CreateRet(ret_val2);

  llvm::verifyFunction(*main_function);

  the_module->print(llvm::errs(), nullptr);

  return 0;
}