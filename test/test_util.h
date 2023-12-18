#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>

enum LowerTo { Toy, Affine, LLVM };
std::optional<std::string> toySource2mlir(std::string_view toySource,
                                          bool enableOpt = false,
                                          LowerTo lowerTo = LowerTo::Toy);
