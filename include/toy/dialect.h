#ifndef TOY_DIALECT_H
#define TOY_DIALECT_H

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Region.h"

// Include the auto-generated header file containing the declarations of the
// toy dialect.
#include "toy/dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/ops.h.inc"

#endif // TOY_DIALECT_H