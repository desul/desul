#if !defined(DESUL_UTILITY_HPP)
#define DESUL_UTILITY_HPP

#include <cstddef>
#include <type_traits>

namespace desul {
   template <class T, T... Ints>
   struct integer_sequence {
      using value_type = T;

      DESUL_HOST_DEVICE static constexpr std::size_t size() noexcept {
         return sizeof...(Ints);
      }
   };

   template <std::size_t... Ints>
   using index_sequence = integer_sequence<std::size_t, Ints...>;

#if 0
   // TODO: Implement make_integer_sequence (may be able to pull from camp)
   template <class T, T N>
   using make_integer_sequence = integer_sequence<T, /* a sequence 0, 1, 2, ..., N-1 */>;

   template <std::size_t N>
   using make_index_sequence = make_integer_sequence<std::size_t, N>;

   template <class... T>
   using index_sequence_for = make_index_sequence<sizeof...(T)>;
#endif

   template <class T>
   DESUL_HOST_DEVICE constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
      return static_cast<typename std::remove_reference<T>::type&&>(t);
   }

   template <class T>
   DESUL_HOST_DEVICE void swap(T& lhs, T& rhs) {
      T tmp = move(lhs);
      lhs = move(rhs);
      rhs = move(tmp);
   }
} // namespace desul

#endif // !defined(DESUL_UTILITY_HPP)

