/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMIC_REF_IMPL_HPP_
#define DESUL_ATOMIC_REF_IMPL_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>

#include "desul/atomics/Common.hpp"
#include "desul/atomics/Generic.hpp"
#include "desul/atomics/Macros.hpp"

namespace desul {
namespace Impl {

// TODO current implementation is missing the following:
// * member functions
//   * is_lock_free
//   * compare_exchange_weak
//   * compare_exchange_strong
//   * wait
//   * notify_one
//   * notify_all
// * constants
//   * is_always_lock_free
//   * required_alignment

template <typename T,
          typename MemoryOrder,
          typename MemoryScope,
          bool = std::is_integral<T>{},
          bool = std::is_floating_point<T>{}>
struct _atomic_ref;

// base class for non-integral, non-floating-point, non-pointer types
template <typename T, typename MemoryOrder, typename MemoryScope>
struct _atomic_ref<T, MemoryOrder, MemoryScope, false, false> {
  static_assert(std::is_trivially_copyable<T>{}, "");

 private:
  T* _ptr;

 public:
  using value_type = T;

  _atomic_ref() = delete;
  _atomic_ref& operator=(_atomic_ref const&) = delete;

  _atomic_ref(_atomic_ref const&) = default;

  explicit _atomic_ref(T& obj) : _ptr(std::addressof(obj)) {}

  T operator=(T desired) const noexcept {
    this->store(desired);
    return desired;
  }

  operator T() const noexcept { return this->load(); }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION void store(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    atomic_store(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T load(_MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T exchange(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, desired, order, MemoryScope());
  }
};

// base class for atomic_ref<integral-type>
template <typename T, typename MemoryOrder, typename MemoryScope>
struct _atomic_ref<T, MemoryOrder, MemoryScope, true, false> {
  static_assert(std::is_integral<T>{}, "");

 private:
  T* _ptr;

 public:
  using value_type = T;
  using difference_type = value_type;

  _atomic_ref() = delete;
  _atomic_ref& operator=(_atomic_ref const&) = delete;

  explicit _atomic_ref(T& obj) : _ptr(&obj) {}

  _atomic_ref(_atomic_ref const&) = default;

  T operator=(T desired) const noexcept {
    this->store(desired);
    return desired;
  }

  operator T() const noexcept { return this->load(); }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION void store(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    atomic_store(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T load(_MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T exchange(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_add(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_add(_ptr, arg, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_sub(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_sub(_ptr, arg, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_and(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_and(_ptr, arg, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_or(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_or(_ptr, arg, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_xor(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_xor(_ptr, arg, order, MemoryScope());
  }

  DESUL_FUNCTION value_type operator++() const noexcept {
    return atomic_add_fetch(_ptr, value_type(1), MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator++(int) const noexcept { return fetch_add(1); }

  DESUL_FUNCTION value_type operator--() const noexcept {
    return atomic_sub_fetch(_ptr, value_type(1), MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator--(int) const noexcept { return fetch_sub(1); }

  DESUL_FUNCTION value_type operator+=(value_type arg) const noexcept {
    atomic_add_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator-=(value_type arg) const noexcept {
    atomic_sub_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator&=(value_type arg) const noexcept {
    atomic_and_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator|=(value_type arg) const noexcept {
    atomic_or_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator^=(value_type arg) const noexcept {
    atomic_xor_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }
};

// base class for atomic_ref<floating-point-type>
template <typename T, typename MemoryOrder, typename MemoryScope>
struct _atomic_ref<T, MemoryOrder, MemoryScope, false, true> {
  static_assert(std::is_floating_point<T>{}, "");

 private:
  T* _ptr;

 public:
  using value_type = T;
  using difference_type = value_type;

  _atomic_ref() = delete;
  _atomic_ref& operator=(_atomic_ref const&) = delete;

  explicit _atomic_ref(T& obj) : _ptr(&obj) {}

  _atomic_ref(_atomic_ref const&) = default;

  T operator=(T desired) const noexcept {
    this->store(desired);
    return desired;
  }

  operator T() const noexcept { return this->load(); }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION void store(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    atomic_store(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T load(_MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T exchange(T desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_add(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_add(_ptr, arg, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_sub(value_type arg, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_sub(_ptr, arg, order, MemoryScope());
  }

  DESUL_FUNCTION value_type operator+=(value_type arg) const noexcept {
    atomic_add_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator-=(value_type arg) const noexcept {
    atomic_sub_fetch(_ptr, arg, MemoryOrder(), MemoryScope());
  }
};

// base class for atomic_ref<pointer-type>
template <typename T, typename MemoryOrder, typename MemoryScope>
struct _atomic_ref<T*, MemoryOrder, MemoryScope, false, false> {
 private:
  T** _ptr;

 public:
  using value_type = T*;
  using difference_type = std::ptrdiff_t;

  _atomic_ref() = delete;
  _atomic_ref& operator=(_atomic_ref const&) = delete;

  explicit _atomic_ref(T*& arg) : _ptr(std::addressof(arg)) {}

  _atomic_ref(_atomic_ref const&) = default;

  T* operator=(T* desired) const noexcept {
    this->store(desired);
    return desired;
  }

  operator T*() const noexcept { return this->load(); }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION void store(T* desired,
                            _MemoryOrder order = _MemoryOrder()) const noexcept {
    atomic_store(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T* load(_MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION T* exchange(T* desired,
                             _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_load(_ptr, desired, order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_add(difference_type d, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_add(_ptr, _type_size(d), order, MemoryScope());
  }

  template <typename _MemoryOrder = MemoryOrder>
  DESUL_FUNCTION value_type
  fetch_sub(difference_type d, _MemoryOrder order = _MemoryOrder()) const noexcept {
    return atomic_fetch_sub(_ptr, _type_size(d), order, MemoryScope());
  }

  DESUL_FUNCTION value_type operator++() const noexcept {
    return atomic_add_fetch(_ptr, _type_size(1), MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator++(int) const noexcept { return fetch_add(1); }

  DESUL_FUNCTION value_type operator--() const noexcept {
    return atomic_sub_fetch(_ptr, _type_size(1), MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator--(int) const noexcept { return fetch_sub(1); }

  DESUL_FUNCTION value_type operator+=(difference_type d) const noexcept {
    atomic_add_fetch(_ptr, _type_size(d), MemoryOrder(), MemoryScope());
  }

  DESUL_FUNCTION value_type operator-=(difference_type d) const noexcept {
    atomic_sub_fetch(_ptr, _type_size(d), MemoryOrder(), MemoryScope());
  }

 private:
  static constexpr std::ptrdiff_t _type_size(std::ptrdiff_t d) noexcept {
    static_assert(std::is_object<T>{}, "");
    return d * sizeof(T);
  }
};

}  // namespace Impl

template <typename T, typename MemoryOrder, typename MemoryScope>
struct atomic_ref : Impl::_atomic_ref<T, MemoryOrder, MemoryScope> {
  explicit atomic_ref(T& obj) noexcept
      : Impl::_atomic_ref<T, MemoryOrder, MemoryScope>(obj) {}

  atomic_ref& operator=(atomic_ref const&) = delete;

  atomic_ref(atomic_ref const&) = default;

  using Impl::_atomic_ref<T, MemoryOrder, MemoryScope>::operator=;
};

}  // namespace desul

#endif
