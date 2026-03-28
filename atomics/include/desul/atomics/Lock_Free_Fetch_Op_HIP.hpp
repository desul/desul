/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_LOCK_FREE_FETCH_OP_HIP_HPP_
#define DESUL_ATOMICS_LOCK_FREE_FETCH_OP_HIP_HPP_

#include <desul/atomics/Adapt_HIP.hpp>
#include <desul/atomics/Compare_Exchange.hpp>
#include <type_traits>
#include "Common.hpp"

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

namespace desul {
namespace Impl {

template <class Oper,
            class T,
            class MemoryOrder,
            class MemoryScope,
            std::enable_if_t<atomic_always_lock_free(sizeof(T)) && atomic_has_builtin_load(), int> = 0>
   __device__ T device_atomic_fetch_oper(
      const Oper& op,
      T* const dest,
      dont_deduce_this_parameter_t<const T> val,
      MemoryOrder order,
      MemoryScope scope) {
    using cas_t = atomic_compare_exchange_t<T>;
    static constexpr auto hip_mem_order = HIPMemoryOrder<MemoryOrder>::value;
    static constexpr auto hip_mem_scope = HIPMemoryScope<MemoryScope>::value;
    cas_t oldval = __hip_atomic_load(reinterpret_cast<cas_t*>(dest), hip_mem_order, hip_mem_scope);
    cas_t assume;
    do {
      assume = oldval;
      if (check_early_exit(op, reinterpret_cast<T&>(oldval), val))
        return reinterpret_cast<T&>(oldval);
      T newval = op.apply(reinterpret_cast<T&>(assume), val);
      oldval =
          device_atomic_compare_exchange(reinterpret_cast<cas_t*>(dest),
                                                   assume,
                                                   reinterpret_cast<cas_t&>(newval),
                                                   order,
                                                   scope);
    } while (assume != oldval);

    return reinterpret_cast<T&>(oldval);
  }

}  // namespace Impl
}  // namespace desul

#if defined(__GNUC__) && (!defined(__clang__))
#pragma GCC diagnostic pop
#endif

#endif
