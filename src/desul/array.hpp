//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#if !defined(DESUL_ARRAY_HPP)
#define DESUL_ARRAY_HPP

#include "desul/Macros.h"

#include <cstddef>

#if !defined(DESUL_GPUCC)
#include <stdexcept>
#endif

#include <type_traits>
#include <utility>

namespace desul {
   ///
   /// Provides a portable std::array-like class.
   /// 
   /// Differences from std::array are listed below:
   /// - No __device__ "at" method (exceptions are not GPU friendly)
   /// - No reverse iterators have been implemented yet
   ///
   template <class T, std::size_t N>
   struct array {
      using value_type = T;
      using size_type = std::size_t;
      using difference_type = std::ptrdiff_t;
      using reference = value_type&;
      using const_reference = const value_type&;
      using pointer = value_type*;
      using const_pointer = const value_type*;
      using iterator = pointer;
      using const_iterator = const_pointer;
      using reverse_iterator = pointer;
      using const_reverse_iterator = const_pointer;

      constexpr reference at(size_type pos) {
         if (pos >= size()) {
            throw std::out_of_range{"desul::array::at detected out of range access"};
         }

         return elements[pos];
      }

      constexpr const_reference at(size_type pos) const {
         if (pos >= size()) {
            throw std::out_of_range{"desul::array::at detected out of range access"};
         }

         return elements[pos];
      }

      DESUL_HOST_DEVICE constexpr reference operator[](size_type pos) {
         return elements[pos];
      }

      DESUL_HOST_DEVICE constexpr const_reference operator[](size_type pos) const {
         return elements[pos];
      }

      DESUL_HOST_DEVICE constexpr reference front() {
         return elements[0];
      }

      DESUL_HOST_DEVICE constexpr const_reference front() const {
         return elements[0];
      }

      DESUL_HOST_DEVICE constexpr reference back() {
         return elements[size() - 1];
      }

      DESUL_HOST_DEVICE constexpr const_reference back() const {
         return elements[size() - 1];
      }

      DESUL_HOST_DEVICE constexpr pointer data() noexcept {
         return elements;
      }

      DESUL_HOST_DEVICE constexpr const_pointer data() const noexcept {
         return elements;
      }

      DESUL_HOST_DEVICE constexpr iterator begin() noexcept {
         return elements;
      }

      DESUL_HOST_DEVICE constexpr const_iterator begin() const noexcept {
         return elements;
      }

      DESUL_HOST_DEVICE constexpr const_iterator cbegin() const noexcept {
         return elements;
      }

      DESUL_HOST_DEVICE constexpr iterator end() noexcept {
         return &elements[size()];
      }

      DESUL_HOST_DEVICE constexpr const_iterator end() const noexcept {
         return &elements[size()];
      }

      DESUL_HOST_DEVICE constexpr const_iterator cend() const noexcept {
         return &elements[size()];
      }

      DESUL_HOST_DEVICE constexpr bool empty() const noexcept {
         return size() == 0;
      }

      DESUL_HOST_DEVICE constexpr size_type size() const noexcept {
         return size();
      }

      DESUL_HOST_DEVICE constexpr size_type max_size() const noexcept {
         return size();
      }

      DESUL_HOST_DEVICE constexpr void fill(const T& value) {
         for (std::size_t i = 0; i < size(); ++i) {
            elements[i] = value;
         }
      }

      DESUL_HOST_DEVICE constexpr void swap(array& other) noexcept(std::is_nothrow_swappable_v<T>) {
         for (std::size_t i = 0; i < size(); ++i) {
            swap(elements[i], other[i]);
         }
      }

      value_type elements[N];
   };

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator==(const array<T, N>& lhs,
                                               const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] != rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator!=(const array<T, N>& lhs,
                                               const array<T, N>& rhs) {
      return !(lhs == rhs);
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator<(const array<T, N>& lhs,
                                              const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] >= rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator<=(const array<T, N>& lhs,
                                               const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] > rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator>(const array<T, N>& lhs,
                                              const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] <= rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr bool operator>=(const array<T, N>& lhs,
                                               const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] < rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <std::size_t I, class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr T& get(array<T, N>& a) noexcept {
      return a[I];
   }

   template <std::size_t I, class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr T&& get(array<T, N>&& a) noexcept {
      return move(a[I]);
   }

   template <std::size_t I, class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr const T& get(const array<T, N>& a) noexcept {
      return a[I];
   }

   template <std::size_t I, class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr const T&& get(const array<T, N>&& a) noexcept {
      return move(a[I]);
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr void swap(array<T, N>& lhs,
                                         array<T, N>& rhs)
                                            noexcept(noexcept(lhs.swap(rhs))) {
      lhs.swap(rhs);
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&a)[N]) {
      array<std::remove_cv_t<T>, N> result;

      for (std::size_t i = 0; i < N; ++i) {
         result[i] = a[i];
      }

      return result;
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&&a)[N]) {
      array<std::remove_cv_t<T>, N> result;

      for (std::size_t i = 0; i < N; ++i) {
         result[i] = move(a[i]);
      }

      return result;
   }

#if 0
   // TODO: Test if this implementation is faster when make_integer_sequence
   //       is implemented.
   namespace detail
   {
      template <class T, std::size_t N, std::size_t... I>
      DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
         to_array_impl(T (&a)[N], index_sequence<I...>)
      {
         return {{a[I]...}};
      }

      template <class T, std::size_t N, std::size_t... I>
      DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N>
         to_array_impl(T (&&a)[N], index_sequence<I...>)
      {
         return {{move(a[I])...}};
      }
   }
    
   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&a)[N]) {
      return detail::to_array_impl(a, make_index_sequence<N>{});
   }

   template <class T, std::size_t N>
   DESUL_HOST_DEVICE constexpr array<std::remove_cv_t<T>, N> to_array(T (&&a)[N]) {
      return detail::to_array_impl(std::move(a), make_index_sequence<N>{});
   }
#endif

   template <class T, std::size_t N>
   struct tuple_size<array<T, N>> :
      std::integral_constant<std::size_t, N>
   { };

   template <class T>
   inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;

   template <std::size_t I, class T>
   struct tuple_element;
    
   template <std::size_t I, class T, std::size_t N>
   struct tuple_element<I, array<T,N>> {
      using type = T;
   };
} // namespace desul

#endif // !defined(DESUL_ARRAY_H)

