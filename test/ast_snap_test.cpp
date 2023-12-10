#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "toy/AST.h"
#include "toy/dialect.h"
#include "toy/lexer.h"
#include "toy/mlirGen.h"
#include "toy/parser.h"
#include "gtest/gtest.h"
#include <llvm-16/llvm/Support/raw_ostream.h>

TEST(AST, Snap) {
  std::string_view toySource = R"(
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(c, d);

  # Finally, calling into `multiply_transpose` with incompatible shapes
  # (<2, 3> and <3, 2>) will trigger a shape inference error.
  var f = multiply_transpose(a, c);

  print(f);
}
)";

  auto lexer =
      LexerBuffer(toySource.begin(), toySource.end() - 1, "sample.toy");
  auto parser = Parser(lexer);
  auto module = parser.parseModule();
  // not null
  ASSERT_TRUE(module);

  auto buf = std::string{};
  auto stream = llvm::raw_string_ostream{buf};
  auto streamer = [&stream]() -> llvm::raw_string_ostream & { return stream; };
  toy::dump_s(streamer, *module);
  // snapshot
  EXPECT_EQ(buf, R"( Module:
  Function 
   Proto 'multiply_transpose' @:3:1
   Params: [a, b]
   Block {
    Return @:4:3
     BinOp '*' @:4:25
      Call 'transpose' [@:4:10
       var: a @:4:20
      ]
      Call 'transpose' [@:4:25
       var: b @:4:35
      ]
   } // Block
  Function 
   Proto 'main' @:7:1
   Params: []
   Block {
    VarDecl a<> @:9:3
     Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00] , <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00] ]  @:9:11
    VarDecl b<2, 3> @:10:3
     Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]  @:10:17
    VarDecl c<> @:14:3
     Call 'multiply_transpose' [@:14:11
      var: a @:14:30
      var: b @:14:33
     ]
    VarDecl d<> @:18:3
     Call 'multiply_transpose' [@:18:11
      var: b @:18:30
      var: a @:18:33
     ]
    VarDecl e<> @:22:3
     Call 'multiply_transpose' [@:22:11
      var: c @:22:30
      var: d @:22:33
     ]
    VarDecl f<> @:26:3
     Call 'multiply_transpose' [@:26:11
      var: a @:26:30
      var: c @:26:33
     ]
    Print [@:28:3
     var: f @:28:9
    ]
   } // Block
)");
}

TEST(MLIR, Snap) {
  std::string_view toySource = R"(
def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b = just_add(a, a);
  print([[1, 1], [1, 2]]);
  print(1 + (2 + 3));
  print([1, 2, 3] + [4, 5, 6]);
}
  )";
  auto lexer =
      LexerBuffer(toySource.begin(), toySource.end() - 1, "sample.toy");
  auto parser = Parser(lexer);
  auto module = parser.parseModule();
  // not null
  ASSERT_TRUE(module);

  mlir::MLIRContext context;
  context.getOrLoadDialect<toy::ToyDialect>();

  auto moduleOp = mlirGen(context, *module);
  auto buf = std::string{};
  auto stream = llvm::raw_string_ostream{buf};
  moduleOp->print(stream);
  EXPECT_EQ(buf, R"(module {
  "toy.func"() ({
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = toy.generic_call @just_add(%0, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %2 = "toy.constant"() {value = dense<[[1.000000e+00, 1.000000e+00], [1.000000e+00, 2.000000e+00]]> : tensor<2x2xf64>} : () -> tensor<2x2xf64>
    toy.print %2 : tensor<2x2xf64>
    %3 = "toy.constant"() {value = dense<1.000000e+00> : tensor<f64>} : () -> tensor<f64>
    %4 = "toy.constant"() {value = dense<2.000000e+00> : tensor<f64>} : () -> tensor<f64>
    %5 = "toy.constant"() {value = dense<3.000000e+00> : tensor<f64>} : () -> tensor<f64>
    %6 = "toy.add"(%4, %5) : (tensor<f64>, tensor<f64>) -> tensor<*xf64>
    %7 = "toy.add"(%3, %6) : (tensor<f64>, tensor<*xf64>) -> tensor<*xf64>
    toy.print %7 : tensor<*xf64>
    %8 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>} : () -> tensor<3xf64>
    %9 = "toy.constant"() {value = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<3xf64>} : () -> tensor<3xf64>
    %10 = "toy.add"(%8, %9) : (tensor<3xf64>, tensor<3xf64>) -> tensor<*xf64>
    toy.print %10 : tensor<*xf64>
    toy.return
  }) {function_type = () -> (), sym_name = "main"} : () -> ()
}
)");
}