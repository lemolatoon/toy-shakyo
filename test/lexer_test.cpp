#include "toy/lexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

TEST(LexerTest, LexerTest) {
  llvm::StringRef buffer = R"(
def add(a, b) {
[] return a + b + 33.3;
}
)";

  auto lexer = toy::LexerBuffer(buffer.data(),
                                buffer.data() + buffer.size() - 1, "test.toy");
  EXPECT_EQ(toy::tok_def, lexer.getNextToken());
  EXPECT_EQ(toy::tok_identifier, lexer.getNextToken());
  EXPECT_EQ("add", lexer.getIdent());
  EXPECT_EQ(toy::tok_parenthese_open, lexer.getNextToken());
  EXPECT_EQ(toy::tok_identifier, lexer.getNextToken());
  EXPECT_EQ(toy::Token(','), lexer.getNextToken());
  EXPECT_EQ(toy::tok_identifier, lexer.getNextToken());
  EXPECT_EQ(toy::tok_parenthese_close, lexer.getNextToken());
  EXPECT_EQ(2, lexer.getLastLocation().line);
  EXPECT_EQ(13, lexer.getLastLocation().col);
  EXPECT_EQ(toy::tok_brace_open, lexer.getNextToken());
  EXPECT_EQ(toy::tok_sbracket_open, lexer.getNextToken());
  EXPECT_EQ(toy::tok_sbracket_close, lexer.getNextToken());
  EXPECT_EQ(toy::tok_return, lexer.getNextToken());
  EXPECT_EQ(toy::tok_identifier, lexer.getNextToken());
  EXPECT_EQ(toy::Token('+'), lexer.getNextToken());
  EXPECT_EQ(toy::tok_identifier, lexer.getNextToken());
  EXPECT_EQ(toy::Token('+'), lexer.getNextToken());
  EXPECT_EQ(toy::tok_number, lexer.getNextToken());
  EXPECT_EQ(33.3, lexer.getValue());
  EXPECT_EQ(toy::tok_semicolon, lexer.getNextToken());
  EXPECT_EQ(toy::tok_brace_close, lexer.getNextToken());
  EXPECT_EQ(toy::tok_eof, lexer.getNextToken());
}