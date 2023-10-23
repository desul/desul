#if !defined(DESUL_UTILITY_HPP)
#define DESUL_UTILITY_HPP

#include "desul/Macros.h"

#include <cstddef>
#include <type_traits>

namespace desul {
   template <class T>
   DESUL_HOST_DEVICE constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
      return static_cast<typename std::remove_reference<T>::type&&>(t);
   }

   template <class T>
   DESUL_HOST_DEVICE constexpr void swap(T& a, T& b) noexcept(
                                                        std::is_nothrow_move_constructible<T>::value &&
                                                        std::is_nothrow_move_assignable<T>::value
                                                     )
   {
      T tmp = move(a);
      a = move(b);
      b = move(tmp);
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr void swap(T (&a)[N], T (&b)[N]) noexcept(std::is_nothrow_swappable_v<T>)
   {
      for (std::size_t i = 0; i < N; ++i) {
         desul::swap(a[i], b[i]);
      }
   }
} // namespace desul

#endif // !defined(DESUL_UTILITY_HPP)

