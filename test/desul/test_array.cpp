#include "desul/array.hpp"
#include "Test.hpp"

#include "gtest/gtest.h"

DESUL_TEST_BEGIN(array, initialize) {
   desul::array<int, 3> a{1, 2, 10};

   return a[0] == 1 &&
          a[1] == 2 &&
          a[2] == 10;
} DESUL_TEST_END(array, initialize)

DESUL_TEST_BEGIN(array, copy_initialize)
{
   desul::array<int, 3> a = {10, 2, 1};

   return a[0] == 10 &&
          a[1] == 2 &&
          a[2] == 1;
} DESUL_TEST_END(array, copy_initialize)

DESUL_TEST_BEGIN(array, copy_construct)
{
   desul::array<int, 3> a{1, 2, 10};
   desul::array<int, 3> b{a};

   return b[0] == 1 &&
          b[1] == 2 &&
          b[2] == 10;
} DESUL_TEST_END(array, copy_construct) 

DESUL_TEST_BEGIN(array, copy_assignment)
{
   desul::array<int, 3> a{1, 2, 10};
   a = desul::array<int, 3>{3, 4, 6};

   return a[0] == 3 &&
          a[1] == 4 &&
          a[2] == 6;
} DESUL_TEST_END(array, copy_assignment)

// As currently implemented, at is not portable
TEST(host_array, at)
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

DESUL_TEST_BEGIN(array, subscript)
{
   desul::array<int, 2> a = {1, 8};
   a[0] = 3;

   const desul::array<int, 2> b = {8, 1};

   return a[0] == 3 &&
          b[0] == 8;
} DESUL_TEST_END(array, subscript);

DESUL_TEST_BEGIN(array, front)
{
   desul::array<int, 2> a = {1, 8};
   a.front() = 3;

   const desul::array<int, 2> b = {8, 1};

   return a.front() == 3 &&
          b.front() == 8;
} DESUL_TEST_END(array, front)

DESUL_TEST_BEGIN(array, back)
{
   desul::array<int, 2> a = {1, 8};
   a.back() = 3;

   const desul::array<int, 2> b = {8, 1};

   return a.back() == 3 &&
          b.back() == 1;
} DESUL_TEST_END(array, back)

DESUL_TEST_BEGIN(array, data)
{
   desul::array<int, 2> a = {1, 8};
   int* a_data = a.data();
   a_data[0] = 3;

   const desul::array<int, 2> b = {8, 1};
   const int* b_data = b.data();

   return a_data[0] == 3 &&
          a_data[1] == 8 &&
          b_data[0] == 8 &&
          b_data[1] == 1;
} DESUL_TEST_END(array, data)

DESUL_TEST_BEGIN(array, begin)
{
   desul::array<int, 2> a = {1, 8};
   auto a_it = a.begin();
   *a_it = 4;

   const desul::array<int, 2> b = {8, 1};
   auto b_it = b.begin();

   return *a_it++ == a[0] &&
          *a_it++ == a[1] &&
          *b_it++ == b[0] &&
          *b_it++ == b[1];
} DESUL_TEST_END(array, begin)

DESUL_TEST_BEGIN(array, cbegin)
{
   desul::array<int, 2> a = {1, 8};
   auto a_it = a.cbegin();

   const desul::array<int, 2>& b = a;
   auto b_it = b.begin();

   return *(a_it++) == a[0] &&
          *(a_it++) == a[1] &&
          *(b_it++) == b[0] &&
          *(b_it++) == b[1];
} DESUL_TEST_END(array, cbegin)

DESUL_TEST_BEGIN(array, end)
{
   desul::array<int, 2> a = {1, 8};
   auto a_it = a.end();
   *(--a_it) = 4;

   const desul::array<int, 2>& b = a;
   auto b_it = b.end();

   return *(a_it--) == a[1] &&
          *(a_it--) == a[0] &&
          *(--b_it) == b[1] &&
          *(--b_it) == b[0];
} DESUL_TEST_END(array, end)

DESUL_TEST_BEGIN(array, cend)
{
   desul::array<int, 2> a = {1, 8};
   auto a_it = a.cend();
   --a_it;

   const desul::array<int, 2>& b = a;
   auto b_it = b.cend();

   return *(a_it--) == a[1] &&
          *(a_it--) == a[0] &&
          *(--b_it) == b[1] &&
          *(--b_it) == b[0];
} DESUL_TEST_END(array, cend)

DESUL_TEST_BEGIN(array, empty)
{
   // Zero sized arrays are technically not allowed,
   // and are explicitly disallowed in device code.
   desul::array<double, 1> a{1.0};

   return !a.empty();
} DESUL_TEST_END(array, empty)

DESUL_TEST_BEGIN(array, size)
{
   // Zero sized arrays are technically not allowed,
   // and are explicitly disallowed in device code.
   desul::array<double, 2> a{1.0, 3.0};

   return a.size() == 2;
} DESUL_TEST_END(array, size)

DESUL_TEST_BEGIN(array, max_size)
{
   // Zero sized arrays are technically not allowed,
   // and are explicitly disallowed in device code.
   desul::array<double, 2> a{1.0, 3.0};

   return a.size() == 2;
} DESUL_TEST_END(array, max_size)

DESUL_TEST_BEGIN(array, fill)
{
   desul::array<int, 3> a{1, 2, 3};
   a.fill(0);

   return a[0] == 0 &&
          a[1] == 0 &&
          a[2] == 0;
} DESUL_TEST_END(array, fill)

DESUL_TEST_BEGIN(array, swap)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{3, 4};

   a.swap(b);

   return a[0] == 3 &&
          a[1] == 4 &&
          b[0] == 1 &&
          b[1] == 2;
} DESUL_TEST_END(array, swap)

DESUL_TEST_BEGIN(array, equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return a == a &&
          !(a == b);
} DESUL_TEST_END(array, equal)

DESUL_TEST_BEGIN(array, not_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return a != b &&
          !(a != a);
} DESUL_TEST_END(array, not_equal)

DESUL_TEST_BEGIN(array, less_than)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return a < b &&
          !(b < a);
} DESUL_TEST_END(array, less_than)

DESUL_TEST_BEGIN(array, less_than_or_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return a <= a &&
          a <= b &&
          !(b <= a);
} DESUL_TEST_END(array, less_than_or_equal)

DESUL_TEST_BEGIN(array, greater_than)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return b > a &&
          !(a > b);
} DESUL_TEST_END(array, greater_than)

DESUL_TEST_BEGIN(array, greater_than_or_equal)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{1, 3};

   return a >= a &&
          b >= a &&
          !(a >= b);
} DESUL_TEST_END(array, greater_than_or_equal)

DESUL_TEST_BEGIN(array, get_lvalue_reference)
{
   desul::array<int, 2> a = {1, 8};
   desul::get<0>(a) = 3;

   const desul::array<int, 2> b = {8, 1};

   return desul::get<0>(a) == 3 &&
          desul::get<0>(b) == 8;
} DESUL_TEST_END(array, get_lvalue_reference)

DESUL_TEST_BEGIN(array, get_rvalue_reference)
{
   desul::array<int, 2> a = {1, 8};
   int&& a0 = desul::get<0>(desul::move(a));

   const desul::array<int, 2> b{6, 8};
   const int&& b1 = desul::get<1>(desul::move(b));

   return a0 == 1 &&
          b1 == 8;
} DESUL_TEST_END(array, get_rvalue_reference)

DESUL_TEST_BEGIN(array, generic_swap)
{
   desul::array<int, 2> a{1, 2};
   desul::array<int, 2> b{3, 4};

   desul::swap(a, b);

   return a[0] == 3 &&
          a[1] == 4 &&
          b[0] == 1 &&
          b[1] == 2;
} DESUL_TEST_END(array, generic_swap)

DESUL_TEST_BEGIN(array, to_array)
{
   int temp[3] = {1, 2, 10};
   desul::array<int, 3> a = desul::to_array(temp);
   desul::array<int, 3> b = desul::to_array(desul::move(temp));

   return a[0] == 1 &&
          a[1] == 2 &&
          a[2] == 10 &&
          b[0] == 1 &&
          b[1] == 2 &&
          b[2] == 10;
} DESUL_TEST_END(array, to_array)

DESUL_TEST_BEGIN(array, tuple_size)
{
   constexpr std::size_t size = std::tuple_size<desul::array<double, 7>>::value;
   constexpr std::size_t size_v = std::tuple_size_v<desul::array<double, 11>>;

   return size == 7 &&
          size_v == 11;
} DESUL_TEST_END(array, tuple_size)

DESUL_TEST_BEGIN(array, tuple_element)
{
   constexpr bool element0 = std::is_same_v<double, std::tuple_element_t<0, desul::array<double, 5>>>;
   constexpr bool element4 = std::is_same_v<double, std::tuple_element_t<4, desul::array<double, 5>>>;

   return element0 &&
          element4;
} DESUL_TEST_END(array, tuple_element)

DESUL_TEST_BEGIN(array, structured_binding)
{
   desul::array<int, 2> a{-1, 1};
   auto& [a0, a1] = a;
   a1 = 3;

   return a0 == -1 &&
          a1 == 3;
} DESUL_TEST_END(array, structured_binding)

DESUL_TEST_BEGIN(array, deduction_guide)
{
   desul::array a{-1, 1};

   return a[0] == -1 &&
          a[1] == 1;
} DESUL_TEST_END(array, deduction_guide)

