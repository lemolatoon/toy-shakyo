// This file implements the AST dump for the Toy Language.

#include "toy/AST.h"
#include "llvm/Support/raw_ostream.h"

namespace toy {
/// Public API for dumping AST
void dump(ModuleAST &module) { ASTDumper(llvm::errs).dump(&module); }
} // namespace toy