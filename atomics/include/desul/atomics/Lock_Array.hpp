/* 
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/
#include "desul/atomics/Compare_Exchange.hpp"
#include "desul/atomics/Macros.hpp"
namespace desul {
namespace Impl {
void init_lock_arrays();

bool lock_address(void* ptr, MemoryScopeNode);
void unlock_address(void* ptr, MemoryScopeNode);
bool lock_address(void* ptr, MemoryScopeDevice);
void unlock_address(void* ptr, MemoryScopeDevice);
bool lock_address(void* ptr, MemoryScopeCore);
void unlock_address(void* ptr, MemoryScopeCore);
}  // namespace Impl
}  // namespace desul
#include "desul/atomics/Lock_Array_Cuda.hpp"
