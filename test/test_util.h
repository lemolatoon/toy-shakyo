#include <optional>
#include <string>

std::optional<std::string> toySource2mlir(std::string_view toySource,
                                          bool enableOpt = false);