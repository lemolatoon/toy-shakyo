#include "lib/hello.h"
#include "gtest/gtest.h"

TEST(LIB, HELLO) {
  auto result = hello();
  EXPECT_EQ(result, "Hello, world!");
}

TEST(LIB, ADD) {
  auto result = add(1, 2);
  EXPECT_EQ(result, 3);
}