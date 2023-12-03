#ifndef TOY_AST_H
#define TOY_AST_H

#include "toy/lexer.h"
#include "llvm/ADT/ArrayRef.h"

namespace toy {

/// shape of variable
struct VarType {
  std::vector<int64_t> shape;
};

/// Base class for all expression nodes
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }
  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(Location loc, double val) : ExprAST(Expr_Num, loc), val(val) {}

  double getValue() { return val; }

  /// 実行時型識別（英: Run-Time Type Identification, RTTI）
  ///
  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

class LiteralExprAST : public ExprAST {
  std::vector<std::unique_ptr<ExprAST>> values;
  std::vector<int64_t> dims;

public:
  LiteralExprAST(Location loc, std::vector<std::unique_ptr<ExprAST>> values,
                 std::vector<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(std::move(values)),
        dims(std::move(dims)) {}

  llvm::ArrayRef<std::unique_ptr<ExprAST>> getValues() { return values; }
  llvm::ArrayRef<int64_t> getDims() { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a"
class VariableExprAST : public ExprAST {
  std::string name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, std::move(loc)), name(name) {}

  llvm::StringRef getName() { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

class VarDeclExprAST : public ExprAST {
  std::string name;
  VarType type;
  std::unique_ptr<ExprAST> initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, VarType type,
                 std::unique_ptr<ExprAST> initVal)
      : ExprAST(Expr_VarDecl, std::move(loc)), name(name),
        type(std::move(type)), initVal(std::move(initVal)) {}

  llvm::StringRef getName() { return name; }
  ExprAST *getInitVal() { return initVal.get(); }
  const VarType &getType() { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

class ReturnExprAST : public ExprAST {
  std::optional<std::unique_ptr<ExprAST>> expr;

public:
  ReturnExprAST(Location loc, std::optional<std::unique_ptr<ExprAST>> expr)
      : ExprAST(Expr_Return, std::move(loc)), expr(std::move(expr)) {}

  std::optional<ExprAST *> getExpr() {
    if (expr.has_value())
      return expr->get();
    return std::nullopt;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

class BinaryExprAST : public ExprAST {
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;

public:
  char getOp() { return op; }
  ExprAST *getLHS() { return lhs.get(); }
  ExprAST *getRHS() { return rhs.get(); }

  BinaryExprAST(Location loc, char op, std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, std::move(loc)), op(op), lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

class CallExprAST : public ExprAST {
  std::string callee;
  ExprASTList args;

public:
  CallExprAST(Location loc, const std::string &callee, ExprASTList args)
      : ExprAST(Expr_Call, std::move(loc)), callee(callee),
        args(std::move(args)) {}

  llvm::StringRef getCallee() { return callee; }
  llvm::ArrayRef<std::unique_ptr<ExprAST>> getArgs() { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// for builtin print calls
class PrintExprAST : public ExprAST {
  std::unique_ptr<ExprAST> arg;

public:
  PrintExprAST(Location loc, std::unique_ptr<ExprAST> arg)
      : ExprAST(Expr_Print, std::move(loc)), arg(std::move(arg)) {}

  ExprAST *getArg() { return arg.get(); }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

/// 関数名、引数たちの名前、(暗黙的に引数の数)を持つ
class PrototypeAST {
  Location location;
  std::string name;
  std::vector<std::unique_ptr<VariableExprAST>> args;

public:
  PrototypeAST(Location location, const std::string &name,
               std::vector<std::unique_ptr<VariableExprAST>> args)
      : location(std::move(location)), name(name), args(std::move(args)) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<std::unique_ptr<VariableExprAST>> getArgs() { return args; }
};

class FunctionAST {
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<ExprASTList> body;

public:
  FunctionAST(std::unique_ptr<PrototypeAST> proto,
              std::unique_ptr<ExprASTList> body)
      : proto(std::move(proto)), body(std::move(body)) {}

  PrototypeAST *getProto() { return proto.get(); }
  ExprASTList *getBody() { return body.get(); }
};

/// 同時に処理される関数定義の列を表す
class ModuleAST {
  std::vector<FunctionAST> functions;

public:
  ModuleAST(std::vector<FunctionAST> functions)
      : functions(std::move(functions)) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }

  void dump(ModuleAST &);
  template <typename F> void dump(F streamer, ModuleAST &module) {
    ASTDumper(streamer).dump(module);
  };
};

} // namespace toy

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace toy;

namespace {
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// F: []() -> stream_like&
template <typename F> class ASTDumper {
public:
  ASTDumper(F &streamer) : stream(streamer){};
  void dump(ModuleAST *node);

private:
  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *type);
  void dump(FunctionAST *type);

  void indent() {
    for (int i = 0; i < curIndent; i++) {
      stream() << " ";
    }
  }
  int curIndent = 0;
  F &stream;
};
}; // namespace

template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();

  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

// Helper macro for indentation
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// LLVM style RTTIを使って、ASTの種類を判定する
template <typename F> void ASTDumper<F>::dump(ExprAST *expr) {
  llvm::TypeSwitch<ExprAST *>(expr)
      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
          [&](auto *node) { this->dump(node); })
      .Default([&](ExprAST *) {
        INDENT();

        stream() << "<unknown Expr, kind " << expr->getKind() << ">\n";
      });
}

/// Variable Declaration dumps
/// - variable name
/// - variable type
/// - variable initializer
template <typename F> void ASTDumper<F>::dump(VarDeclExprAST *varDecl) {
  INDENT();
  stream() << "VarDecl " << varDecl->getName();
  dump(varDecl->getType());
  stream() << " " << loc(varDecl) << "\n";
  dump(varDecl->getInitVal());
}

/// A "block", or a list of expressions
template <typename F> void ASTDumper<F>::dump(ExprASTList *exprList) {
  INDENT();
  stream() << "Block {\n";
  for (auto &expr : *exprList)
    dump(expr.get());
  indent(); // indent for '}'
  stream() << "} // Block\n";
}

/// A number literal
template <typename F> void ASTDumper<F>::dump(NumberExprAST *num) {
  INDENT();
  stream() << num->getValue() << " " << loc(num) << "\n";
}

/// 再帰的なリテラルを出力する
/// [ [ 1, 2 ], [ 3, 4 ] ]
/// の場合、
/// <2, 2>[<2>[ 1, 2 ], <2>[ 3, 4 ] ]
/// のように、次元とともに出力する
template <typename F> void printLitHelper(F &stream, ExprAST *litOrNum) {
  if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
    stream() << num->getValue();
    return;
  }

  auto *literal = llvm::cast<LiteralExprAST>(litOrNum);

  // 次元を出力
  stream() << "<";
  llvm::interleaveComma(literal->getDims(), stream());
  stream() << ">";

  // 再帰的に要素を出力
  stream() << "[ ";
  llvm::interleaveComma(literal->getValues(), stream(),
                        [&](auto &val) { printLitHelper(stream, val.get()); });
  stream() << "] ";
}

/// Print a literal
template <typename F> void ASTDumper<F>::dump(LiteralExprAST *literal) {
  INDENT();
  stream() << "Literal: ";
  printLitHelper(stream, literal);
  stream() << " " << loc(literal) << "\n";
}

template <typename F> void ASTDumper<F>::dump(VariableExprAST *node) {
  INDENT();
  stream() << "var: " << node->getName() << " " << loc(node) << "\n";
}

template <typename F> void ASTDumper<F>::dump(ReturnExprAST *node) {
  INDENT();
  stream() << "Return " << loc(node) << "\n";
  if (node->getExpr().has_value())
    return dump(*node->getExpr());
  {
    // return with no value
    // return ;
    INDENT();
    stream() << "void\n";
  }
}

template <typename F> void ASTDumper<F>::dump(BinaryExprAST *node) {
  INDENT();
  stream() << "BinOp '" << node->getOp() << "' " << loc(node) << "\n";
  dump(node->getLHS());
  dump(node->getRHS());
}

template <typename F> void ASTDumper<F>::dump(CallExprAST *node) {

  INDENT();
  stream() << "Call '" << node->getCallee() << "' [" << loc(node) << "\n";
  for (auto &arg : node->getArgs()) {
    dump(arg.get());
  }
  indent();
  stream() << "]\n";
}

template <typename F> void ASTDumper<F>::dump(PrintExprAST *node) {
  INDENT();
  stream() << "Print [" << loc(node) << "\n";
  dump(node->getArg());
  indent();
  stream() << "]\n";
}

template <typename F> void ASTDumper<F>::dump(const VarType &type) {
  stream() << "<";
  llvm::interleaveComma(type.shape, stream());
  stream() << ">";
}

template <typename F> void ASTDumper<F>::dump(PrototypeAST *node) {
  INDENT();
  stream() << "Proto '" << node->getName() << "' " << loc(node) << "\n";
  indent();
  stream() << "Params: [";
  llvm::interleaveComma(node->getArgs(), stream(),
                        [&](auto &arg) { stream() << arg->getName(); });
  stream() << "]\n";
}

template <typename F> void ASTDumper<F>::dump(FunctionAST *node) {
  INDENT();
  stream() << "Function \n";
  dump(node->getProto());
  dump(node->getBody());
}

template <typename F> void ASTDumper<F>::dump(ModuleAST *node) {
  INDENT();
  stream() << "Module:\n";
  for (auto &func : *node) {
    dump(&func);
  }
}

namespace toy {
/// Public API for dumping AST
void dump(ModuleAST &module);
template <typename F> void dump_s(F streamer, ModuleAST &module) {
  ASTDumper(streamer).dump(&module);
}
} // namespace toy

#endif // TOY_AST_H