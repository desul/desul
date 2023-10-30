#include "desul/array.hpp"
#include "Test.hpp"

#include "gtest/gtest.h"

TEST(array, initialization)
{
   desul::array<int, 3> a{1, 2, 10};

   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 10);
}

DESUL_TEST_BEGIN(array, initialization) {
   desul::array<int, 3> a{1, 2, 10};

   return a[0] == 1 &&
          a[1] == 2 &&
          a[2] == 10;
} DESUL_TEST_END(array, initialization)

TEST(array, copy_initialization)
{
   desul::array<int, 3> a = {10, 2, 1};

   EXPECT_EQ(a[0], 10);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 1);
}

TEST(array, copy_construct)
{
   desul::array<int, 3> a{1, 2, 10};
   desul::array<int, 3> b{a};

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
   EXPECT_EQ(b[2], 10);
}

TEST(array, copy_assignment)
{
   desul::array<int, 3> a{1, 2, 10};
   a = desul::array<int, 3>{3, 4, 6};

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);
   EXPECT_EQ(a[2], 6);
}

TEST(array, at)
{
   desul::array<int, 1> a = {-4};

   int resultAt0 = -1;
   int resultAt1 = -1;
   bool exception = false;

   try {
      resultAt0 = a.at(0);
      resultAt1 = a.at(1);
   }
   catch (std::out_of_range e) {
      exception = true;
   }

   EXPECT_EQ(resultAt0, -4);
   EXPECT_EQ(resultAt1, -1);
   EXPECT_TRUE(exception);
}

TEST(array, access)
{
   desul::array<int, 2> a = {1, 12};
   a[0] = 3;
   EXPECT_EQ(a[0], 3);

   const desul::array<int, 2>& b = a;
   EXPECT_EQ(b[0], 3);
}

TEST(array, front)
{
   desul::array<int, 2> a = {1, 12};
   a.front() = 3;
   EXPECT_EQ(a[0], 3);

   const desul::array<int, 2>& b = a;
   EXPECT_EQ(b.front(), 3);
   EXPECT_EQ(b[0], 3);
}

TEST(array, back)
{
   desul::array<int, 2> a = {1, 12};
   a.back() = 5;
   EXPECT_EQ(a[1], 5);

   const desul::array<int, 2>& b = a;
   EXPECT_EQ(b.back(), 5);
   EXPECT_EQ(b[1], 5);
}

TEST(array, data)
{
   desul::array<int, 2> a = {1, 12};
   int* a_data = a.data();
   EXPECT_EQ(a_data[0], a[0]);
   EXPECT_EQ(a_data[1], a[1]);

   const desul::array<int, 2>& b = a;
   const int* b_data = b.data();
   EXPECT_EQ(b_data[0], b[0]);
   EXPECT_EQ(b_data[1], b[1]);
}

TEST(array, begin)
{
   desul::array<int, 2> a = {1, 12};
   auto a_it = a.begin();
   *a_it = 4;
   EXPECT_EQ(a[0], 4);
   *(++a_it) = 6;
   EXPECT_EQ(a[1], 6);

   const desul::array<int, 2>& b = a;
   auto b_it = b.begin();
   EXPECT_EQ(*b_it, b[0]);
   EXPECT_EQ(*(++b_it), b[1]);
}

TEST(array, cbegin)
{
   desul::array<int, 2> a = {1, 12};
   auto a_it = a.cbegin();
   EXPECT_EQ(*a_it, a[0]);
   EXPECT_EQ(*(++a_it), a[1]);

   const desul::array<int, 2>& b = a;
   auto b_it = b.begin();
   EXPECT_EQ(*b_it, b[0]);
   EXPECT_EQ(*(++b_it), b[1]);
}

TEST(array, end)
{
   desul::array<int, 2> a = {1, 12};
   auto a_it = a.end();
   *(--a_it) = 4;
   EXPECT_EQ(a[1], 4);
   *(--a_it) = 6;
   EXPECT_EQ(a[0], 6);

   const desul::array<int, 2>& b = a;
   auto b_it = b.end();
   EXPECT_EQ(*(--b_it), b[1]);
   EXPECT_EQ(*(--b_it), b[0]);
}

TEST(array, cend)
{
   desul::array<int, 2> a = {1, 12};
   auto a_it = a.cend();
   EXPECT_EQ(*(--a_it), a[1]);
   EXPECT_EQ(*(--a_it), a[0]);

   const desul::array<int, 2>& b = a;
   auto b_it = b.cend();
   EXPECT_EQ(*(--b_it), b[1]);
   EXPECT_EQ(*(--b_it), b[0]);
}

TEST(array, empty)
{
   desul::array<double, 0> a{};
   EXPECT_TRUE(a.empty());

   desul::array<double, 1> b{1.0};
   EXPECT_FALSE(b.empty());
}

TEST(array, size)
{
   desul::array<double, 0> a{};
   EXPECT_EQ(a.size(), 0);

   desul::array<double, 2> b{1.0, 3.0};
   EXPECT_EQ(b.size(), 2);
}

TEST(array, max_size)
{
   desul::array<double, 0> a{};
   EXPECT_EQ(a.max_size(), 0);

   desul::array<double, 2> b{1.0, 3.0};
   EXPECT_EQ(b.max_size(), 2);
}

TEST(array, fill)
{
   desul::array<int, 3> a{1, 2, 3};
   a.fill(0);

   for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(a[i], 0);
   }
}

TEST(array, swap)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{3, 4};

   a.swap(b);

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
}

TEST(array, equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(a == a);
   EXPECT_FALSE(a == b);
}

TEST(array, not_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(a != b);
   EXPECT_FALSE(a != a);
}

TEST(array, less_than)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(a < b);
   EXPECT_FALSE(b < a);
}

TEST(array, less_than_or_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(a <= a);
   EXPECT_TRUE(a <= b);
   EXPECT_FALSE(b <= a);
}

TEST(array, greater_than)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(b > a);
   EXPECT_FALSE(a > b);
}

TEST(array, greater_than_or_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   EXPECT_TRUE(a >= a);
   EXPECT_TRUE(b >= a);
   EXPECT_FALSE(a >= b);
}

TEST(array, get_lvalue_reference)
{
   desul::array<int, 2> a = {1, 12};
   desul::get<0>(a) = 3;
   EXPECT_EQ(a[0], 3);

   const desul::array<int, 2>& b = a;
   EXPECT_EQ(desul::get<0>(b), 3);
}

TEST(array, get_rvalue_reference)
{
   desul::array<int, 2> a = {1, 12};
   int&& a0 = desul::get<0>(desul::move(a));
   EXPECT_EQ(a0, 1);
   EXPECT_EQ(a[0], 1);

   const desul::array<int, 2> b{6, 8};
   const int&& b1 = desul::get<1>(desul::move(b));
   EXPECT_EQ(b1, 8);
   EXPECT_EQ(b[1], 8);
}

TEST(array, generic_swap)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{3, 4};

   desul::swap(a, b);

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
}

TEST(array, to_array)
{
   int temp[3] = {1, 2, 10};

   desul::array<int, 3> a = desul::to_array(temp);
   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 10);

   desul::array<int, 3> b = desul::to_array(desul::move(temp));
   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
   EXPECT_EQ(b[2], 10);
}

TEST(array, tuple_size)
{
   constexpr std::size_t size = std::tuple_size<desul::array<double, 7>>::value;
   constexpr std::size_t size_v = std::tuple_size_v<desul::array<double, 11>>;

   EXPECT_EQ(size, 7);
   EXPECT_EQ(size_v, 11);
}

TEST(array, tuple_element)
{
   constexpr bool element0 = std::is_same_v<double, std::tuple_element_t<0, desul::array<double, 5>>>;
   constexpr bool element4 = std::is_same_v<double, std::tuple_element_t<4, desul::array<double, 5>>>;

   EXPECT_TRUE(element0);
   EXPECT_TRUE(element4);
}

TEST(array, structured_binding)
{
   desul::array<int, 2> a{-1, 1};
   auto& [a0, a1] = a;
   EXPECT_EQ(a0, -1);
   EXPECT_EQ(a1, 1);

   a1 = 3;
   EXPECT_EQ(a[1], 3);
}

TEST(array, deduction_guide)
{
   desul::array a{-1, 1};
   EXPECT_EQ(a[0], -1);
   EXPECT_EQ(a[1], 1);
}

