#include "desul/utility.hpp"

#include "gtest/gtest.h"

TEST(utility, swap)
{
   int a = 1;
   int b = 2;

   desul::swap(a, b);

   EXPECT_EQ(a, 2);
   EXPECT_EQ(b, 1);
}

TEST(utility, swap_c_array)
{
   int a[2] = {1, 2};
   int b[2] = {3, 4};

   desul::swap(a, b);

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
}

