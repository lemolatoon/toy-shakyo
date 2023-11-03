#ifndef TOY_PARSER_H
#define TOY_PARSER_H

#include "toy/AST.h"
#include "toy/lexer.h"

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
/// block_expr ::= declaration | 'return' | expression
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

  std::unique_ptr<ModuleAST> parserModule() {
    lexer.getNextToken(); // get first token to lexer's buffer

    std::vector<FunctionAST> functions;
    // TODO
  }

private:
  Lexer &lexer;

  /// Parse a return statement.
  /// return ::= 'return' ';' | 'return' expression ';'
  std::unique_ptr<ReturnExprAST> parseReturn();

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr();

  /// Parse a literal array expression.
  /// tensorListは、','で区切られた多次元配列
  /// tensorLiteral ::= '[' literalList ']' | number
  /// literalList ::= tensorLiteral | tensorLiteral ',' literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr();

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr();

  /// identifierexpr
  ///  ::= identifier
  ///  ::= identifier '(' argument_list ')'
  /// argument_list ::= expression ',' argument_list
  std::unique_ptr<ExprAST> parseIdentifierExpr();

  /// primary
  ///	 ::= identifierexpr
  ///  ::= numberexpr
  ///  ::= parenexpr
  ///  ::= parenexpr
  ///  ::= tensorLiteral
  std::unique_ptr<ExprAST> parsePrimary();

  /// binoprhs ::= (binop primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs);

  /// expression ::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression();

  /// type ::= < shape_list >
  /// shape_list ::= num | num ',' shape_list
  std::unique_ptr<VarType> parseType();

  /// declaration ::= 'var' identifier '[' type ']' '=' expression
  std::unique_ptr<VarDeclExprAST> parseDeclaration();

  /// block ::= '{' expression_list '}'
  /// expression_list ::= block_expr ';' expression_list
  /// block_expr ::= declaration | 'return' | expression
  std::unique_ptr<ExprASTList> parseBlock();

  /// prototype ::= 'def' identifier '(' declaration_list ')'
  /// declaration_list ::= identifier | identifier ',' declaration_list
  std::unique_ptr<PrototypeAST> parsePrototype();

  /// `def` から始まる prototype から始まる関数定義をパースする。
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition();
};

} // namespace toy

#endif // TOY_PARSER_H