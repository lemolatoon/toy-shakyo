#include "toy/dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "toy/dialect.cpp.inc"
#define GET_OP_CLASSES
#include "toy/ops.cpp.inc"

using namespace toy;

void ToyDialect::initialize() {
  this->addOperations<
#define GET_OP_LIST
#include "toy/ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

// Verifier for the constant operation. This corresponds to the
// `let hasVerifier = 1 in the op definition.
mlir::LogicalResult ConstantOp::verify() {
  // If the return type of the constant is not an unranked tensor, the shape
  // must match the shape of the attribute holding the data.
  auto resultType =
      llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return mlir::success();

  // Check that the rank of the attribute type matches the rank of the constant
  // result type.
  // まずは、attrに渡された次元の数と、ConstantOpの結果の次元の数が一致するかを確認する。
  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value "
                       "attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  // Check that each of the dimensions match between the two types.
  // 各次元が一致するかをチェックする。
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError(
                 "return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim]
             << " != " << resultType.getShape()[dim];
    }
  }

  return mlir::success();
}

// AddOp

// ops.tdのAddOpで宣言したbuild関数
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  auto dataType = mlir::UnrankedTensorType::get(builder.getF64Type());
  AddOp::build(builder, state, dataType, {lhs, rhs},
               mlir::ArrayRef<mlir::NamedAttribute>());
  // tutorial
  // は以下の記述だった。実際↑のoverloadされたAddOp::buildも同じことをしているので、おそらく同じ意味。
  // state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
  // state.addOperands({lhs, rhs});
}

// FuncOp

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::StringRef name, mlir::FunctionType type,
                   mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.

  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Dispatch to the FunctionOpInterface provided utility method that parses
  // the
  // function operation.
  auto buildFuncType =
      [](mlir::Builder &builder, mlir::ArrayRef<mlir::Type> argTypes,
         mlir::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  // Dispatch to the FunctionOpInterface provided method that prints the
  // function operation.
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

// ReturnOp

mlir::LogicalResult ReturnOp::verify() {
  // 未検証: verifyが呼ばれる時点では、traitは保証されている？
  // We know that the parent operation is a function, because of the 'HasParent'
  // trait attached to the operation definition.
  auto function = mlir::cast<FuncOp>((*this)->getParentOp());

  /// ReturnOps can only have a single optional operand.
  if (getNumOperands() > 1)
    return emitOpError() << "expects at most 1 return operand";

  // The operand number and types must match the function signature.
  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError() << "does not return the same number of values ("
                         << getNumOperands() << ") as the enclosing function ("
                         << results.size() << ")";

  // If the operation does not have an input, we are done.
  if (!hasOperand()) {
    return mlir::success();
  }

  auto inputType = *operand_type_begin();
  auto resultType = results.front();

  // Check that the result type of the function matches the operand type.
  if (inputType == resultType ||
      llvm::isa<mlir::UnrankedTensorType>(inputType) ||
      llvm::isa<mlir::UnrankedTensorType>(resultType))
    return mlir::success();

  return emitOpError() << "type of return operand (" << inputType
                       << ") doesn't match function result type (" << resultType
                       << ")";
}

// GenericCallOp

void GenericCallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::StringRef callee,
                          mlir::ArrayRef<mlir::Value> arguments) {
  // Generic call always returns an unranked Tensor initially.
  state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(arguments);
  state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

// MulOp

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  auto resultType = mlir::UnrankedTensorType::get(builder.getF64Type());
  MulOp::build(builder, state, resultType, {lhs, rhs},
               mlir::ArrayRef<mlir::NamedAttribute>());
}

// TransposeOp

void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value input) {
  state.addTypes(mlir::UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(input);
}

mlir::LogicalResult TransposeOp::verify() {
  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(getOperand().getType());
  auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(getType());

  // どちらかが、UnrankedTensorTypeならば、OK
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}