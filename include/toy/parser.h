#ifndef TOY_PARSER_H
#define TOY_PARSER_H

#include "toy/AST.h"
#include "toy/lexer.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace toy {

/// Parser of Toy Language.
/// BNF:
///======= module ========
/// module ::= definition*
///========================
///====== functions =======
/// definition ::= prototype block
/// prototype ::= 'def' identifier '(' declaration_list ')'
/// declaration_list ::= identifier | identifier ',' declaration_list
///
/// block ::= '{' expression_list '}'
/// expression_list ::= block_expr ';' expression_list
/// block_expr ::= declaration | return | expression
///========================
///===== declaration ======
/// declaration ::= 'var' identifier '[' type ']' '=' expression
/// type ::= < shape_list >
/// shape_list ::= num | num ',' shape_list
///========================
///====== expression ======
/// expression ::= add
///
/// add ::= mul ('+' mul | '-' mul)*
///
/// mul ::= primary ('*' primary)*
///
/// primary
///	 ::= identifierexpr
///  ::= numberexpr
///  ::= parenexpr
///  ::= parenexpr
///  ::= tensorLiteral
///
/// identifierexpr
///  ::= identifier
///  ::= identifier '(' argument_list ')'
/// argument_list ::= expression ',' argument_list
///
/// numberexpr ::= number
///
/// parenexpr ::= '(' expression ')'
///
/// tensorLiteral ::= '[' literalList ']' | number
/// literalList ::= tensorLiteral | tensorLiteral ',' literalList
///
/// return :== 'return' ';' | 'return' expression ';'
///========================
class Parser {
public:
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// module ::= definition*
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // get first token to lexer's buffer

    std::vector<FunctionAST> functions;
    while (auto f = parseDefinition()) {
      functions.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof) {
        break;
      }
    }

    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;

  /// Parse a return statement.
  /// return ::= 'return' ';' | 'return' expression ';'
  std::unique_ptr<ReturnExprAST> parseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    std::optional<std::unique_ptr<ExprAST>> expr;
    if (lexer.getCurToken() != ';') {
      expr = parseExpression();
      if (!expr)
        return nullptr;
    }

    return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result =
        std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(result);
  }

  /// Parse a literal array expression.
  /// tensorListは、','で区切られた多次元配列
  /// tensorLiteral ::= '[' literalList ']' | number
  /// literalList ::= tensorLiteral | tensorLiteral ',' literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    /// 現在のネストのレベルでの値
    std::vector<std::unique_ptr<ExprAST>> values;
    // 現在のネストのレベルにおける次元
    std::vector<int64_t> dims;

    do {
      if (lexer.getCurToken() == '[') {
        // さらにネストする
        values.push_back(parseTensorLiteralExpr());

        if (!values.back())
          return nullptr; // parseした結果がnullptrなら、エラー
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or '['", "in literal expression");

        values.push_back(parseNumberExpr());
      }

      // End of the list
      if (lexer.getCurToken() == ']')
        break;

      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("']' or ','", "in literal expression");

      lexer.getNextToken(); // eat ','
    } while (true);

    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ']'

    // まずはじめに、現在のネストレベルの次元数をpushする。
    dims.push_back(values.size());

    // 内側にネストしたarrayがあるなら、それらの次元数が同じかどうかをチェックする。
    if (llvm::any_of(values, [](auto &expr) {
          return llvm::isa<LiteralExprAST>(expr.get());
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // ひとつ下のarrayの次元を、現在の次元の末尾に追加する。
      auto firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // すべてのネストされたarrayのすべての次元が同じかどうかチェックする。
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");

        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }

    return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                            std::move(dims));
  }

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr() {
    lexer.getNextToken(); // eat '('
    auto v = parseExpression();

    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");

    lexer.consume(Token(')'));

    return v;
  }

  /// identifierexpr
  ///  ::= identifier
  ///  ::= identifier '(' argument_list ')'
  /// argument_list ::= expression ',' argument_list
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    std::string name(lexer.getIdent());

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat Identifier

    if (lexer.getCurToken() != '(') // Simple variable ref.
      return std::make_unique<VariableExprAST>(std::move(loc), name);

    // This is a function call.
    lexer.consume(Token('('));
    std::vector<std::unique_ptr<ExprAST>> args;

    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto arg = parseExpression()) {
          args.push_back(std::move(arg));
        } else {
          return nullptr;
        }

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>("',' or ')'", "in argument list");
        lexer.getNextToken(); // eat ','
      }
    }

    lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (name == "print") {
      if (args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
    }

    // Call to a user-defined function
    return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
  }

  /// primary
  ///	 ::= identifierexpr
  ///  ::= numberexpr
  ///  ::= parenexpr
  ///  ::= tensorLiteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return parseError<ExprAST>("an expression");
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case '(':
      return parseParenExpr();
    case '[':
      return parseTensorLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// `exprPrec`は現在処理している二項演算子の優先度を表す。
  ///
  /// binoprhs ::= (binop primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    while (true) {
      int tokPrec = getTokPrecedence();

      // 二項演算子の優先度が、現在処理している二項演算子の優先度よりも低い場合、
      // それ以上は処理しない。
      if (tokPrec < exprPrec)
        return lhs;

      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      // 二項演算子の右辺をパースする。
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // 次の二項演算子の優先度(二項演算子でないときは-1)が、
      // 現在処理している二項演算子の優先度よりも高い場合、
      // 二項演算子の右辺を、次の二項演算子の左辺として処理する。
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // 二項演算子の左辺と右辺を結合する。
      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                            std::move(lhs), std::move(rhs));
    }
  }

  /// expression ::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;

    return parseBinOpRHS(0, std::move(lhs));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num ',' shape_list
  std::unique_ptr<VarType> parseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");

    lexer.getNextToken(); // eat '<'

    auto type = std::make_unique<VarType>();

    while (lexer.getCurToken() == tok_number) {
      type->shape.push_back(lexer.getValue());
      lexer.getNextToken();
      if (lexer.getCurToken() == ',')
        lexer.getNextToken();
    }

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");

    lexer.getNextToken(); // eat '>'
    return type;
  }

  /// declaration ::= 'var' identifier '[' type ']' '=' expression
  std::unique_ptr<VarDeclExprAST> parseDeclaration() {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");

    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat 'var'

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identifier",
                                        "after 'var' declaration");

    std::string ident(lexer.getIdent());
    lexer.getNextToken(); // eat identifier

    std::unique_ptr<VarType> type; // Type is optional, it can be inferred
    if (lexer.getCurToken() == '<') {
      type = parseType();
      if (!type)
        return nullptr;
    }

    if (!type)
      type = std::make_unique<VarType>(); // initialize with empty type

    lexer.consume(Token('='));
    auto expr = parseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(ident),
                                            std::move(*type), std::move(expr));
  }

  /// block ::= '{' expression_list '}'
  /// expression_list ::= block_expr ';' expression_list
  /// block_expr ::= declaration | 'return' | expression
  std::unique_ptr<ExprASTList> parseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = std::make_unique<ExprASTList>();

    // empty expressionを無視する。
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        // variable declaration
        auto var_decl = parseDeclaration();
        if (!var_decl)
          return nullptr;
        exprList->push_back(std::move(var_decl));
      } else if (lexer.getCurToken() == tok_return) {
        // return statement
        auto ret = parseReturn();
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else {
        // General expression
        auto expr = parseExpression();
        if (!expr)
          return nullptr;

        exprList->push_back(std::move(expr));
      }

      // block内ではそれぞれが、`;`で区切られている。
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // empty expressionを無視する。
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));

    return exprList;
  }

  /// prototype ::= 'def' identifier '(' declaration_list ')'
  /// declaration_list ::= identifier | identifier ',' declaration_list
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_def)
      return parseError<PrototypeAST>("def", "in prototype");
    lexer.consume(tok_def);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string fnName(lexer.getIdent());
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<VariableExprAST>> args;
    if (lexer.getCurToken() != ')') {
      // 引数がある
      do {
        std::string name(lexer.getIdent());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
        args.push_back(std::move(decl));

        if (lexer.getCurToken() != ',')
          break;

        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }

    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    lexer.consume(Token(')'));
    return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                          std::move(args));
  }

  /// `def` から始まる prototype から始まる関数定義をパースする。
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition() {
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    if (auto block = parseBlock())
      return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
    return nullptr;
  }

  /// 現在のトークンの優先度を返す。
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// parse中にエラーを知らせるためのHelper Function.
  /// expectedなトークンと、より多くの情報を知らせるためのcontextを受け取る。
  /// Location情報は、Lexerから取得する。
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();

    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has token " << curToken;

    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace toy

#endif // TOY_PARSER_H