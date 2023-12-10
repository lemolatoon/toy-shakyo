#ifndef TOY_LEXER_H
#define TOY_LEXER_H

#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace toy {

/// ファイル内での位置を表す構造体
struct Location {
  std::shared_ptr<std::string> file; ///< filename
  int line;                          ///< line number
  int col;                           ///< column number
};

/// 下に該当しないトークンは、そのASCIIコードの文字。
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_brace_open = '{',
  tok_brace_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // 予約語
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};

/// Parserが想定するLexerのインターフェースを定義する抽象クラス
class Lexer {
public:
  Lexer(std::string filename)
      : lastLocation(
            {std::make_shared<std::string>(std::move(filename), 0, 0)}) {}
  virtual ~Lexer() = default;

  /// 現在のトークンを取得する。
  Token getCurToken() { return curTok; }

  /// 次のトークンを取得し、それを返す。
  Token getNextToken() { return curTok = getTok(); }

  /// 現在のトークンの種類を確認(assert)してから、次のトークンを取得する。
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// 現在のidentifierを返す。（前提: getCurToken() == tok_identifier）
  llvm::StringRef getIdent() {
    assert(curTok == tok_identifier && "expecting an identifier failed");
    return identifierStr;
  }

  /// 現在のnumberを返す。（前提: getCurToken() == tok_number）
  double getValue() {
    assert(curTok == tok_number && "expecting a number failed");
    return numVal;
  }

  Location getLastLocation() { return lastLocation; }

  /// 現在の行番号を返す。
  int getLine() { return curLineNum; }

  /// 現在の列番号を返す。
  int getCol() { return curCol; }

private:
  /// 次の行を取得するメソッド。この実装は、このクラスを継承したクラスで実装する。
  /// end of line(EOF)を知らせるときは、空文字列を返す。
  /// 通常は、"\n"で終わる文字列を返す。
  virtual llvm::StringRef readNextLine() = 0;

  /// 次の文字を取得するメソッド。このメソッドでは、実装先の`readNextLine`を必要に応じて呼んだりする。
  int getNextChar() {
    /// `readNextLine`の返り値が空文字列のときは、EOFを表す。
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;

    auto nextChar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();

    // 一行全部読み切った
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();

    if (nextChar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextChar;
  }

  /// 次のトークンを取得するメソッド。`getNextChar`を呼ぶ。
  Token getTok() {
    /// 空白を呼んでいるうちは読み飛ばす。
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // これから読むトークンの開始位置
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
      identifierStr = (char)lastChar;

      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char)lastChar;

      if (identifierStr == "return")
        return tok_return;
      if (identifierStr == "def")
        return tok_def;
      if (identifierStr == "var")
        return tok_var;
      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;

      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      // コメントを読み飛ばして、まだその先に文字があるなら、次のトークンを返す。
      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    int thisChar = lastChar;
    lastChar = Token(getNextChar());
    return Token(thisChar);
  }

  /// 入力から最後に読んだトークン
  Token curTok = tok_eof;

  /// Location for the current token `curTok`
  Location lastLocation;

  /// curTok == tok_identifierのときのみ有効
  std::string identifierStr;

  /// curTok == tok_numberのときのみ有効
  double numVal = 0;

  /// `getNextChar`によって一文字先読みした文字を格納する
  Token lastChar = Token(' ');

  /// input streamにおける現在の行番号
  int curLineNum = 0;
  /// input streamにおける現在の列番号
  int curCol = 0;

  /// input streamから`readNextLine()`で読み込んだ一行分の文字列を格納する
  llvm::StringRef curLineBuffer = "\n";
};

/// Lexer classの実装。読み出す文字列はメモリ上に保持する。
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    // 改行文字か、バッファの終端まで読み進める
    while (current <= end && *current && *current != '\n')
      ++current;

    // '\n'を読む
    if (current <= end && *current)
      ++current;

    llvm::StringRef line(begin, static_cast<size_t>(current - begin));
    return line;
  }

  const char *current, *end;
};

} // namespace toy

#endif // TOY_LEXER_H