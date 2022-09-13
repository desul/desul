/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#ifndef DESUL_ATOMICS_THREAD_FENCE_SCOPECALLER_HPP_
#define DESUL_ATOMICS_THREAD_FENCE_SCOPECALLER_HPP_

#include <desul/atomics/Common.hpp>

namespace desul {
namespace Impl {

template <class MemoryOrder>
DESUL_INLINE_FUNCTION void host_atomic_thread_fence(MemoryOrder, MemoryScopeCaller) {}

template <class MemoryOrder>
DESUL_INLINE_FUNCTION void device_atomic_thread_fence(MemoryOrder, MemoryScopeCaller) {}

}  // namespace Impl
}  // namespace desul

#endif
