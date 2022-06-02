/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "PerfTestAtomics.hpp"
#define MEMORY_SPACE DefaultExecutionSpace::memory_space
#define EXECUTION_SPACE DefaultExecutionSpace
#define SCALAR int32_t
#define SCALAR_NAME int32_t
#define MEMORY_OP atomic_add_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_add_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_sub_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_sub_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_inc_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_inc_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_dec_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_dec_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_min_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_min_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_max_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP
#define MEMORY_OP atomic_fetch_max_op
#include "PerfTestAtomicsNeigh_Scope.inc"
#undef MEMORY_OP

/*
#define MEMORY_ORDER desul::MemoryOrderRelaxed
#define MEMORY_SCOPE desul::MemoryScopeDevice
namespace Test {
TEST(atomic, kokkos_perf_random_loc_add_int32_t) {
  desul::ensure_cuda_lock_arrays_on_device();
  test_atomic_perf_random_loc<int32_t,atomic_add_opp,Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space>(10000000);
}
TEST(atomic, kokkos_perf_random_neigh_add_int32_t) {
  desul::ensure_cuda_lock_arrays_on_device();
  test_atomic_perf_random_neighs<int32_t,atomic_add_opp,Kokkos::DefaultExecutionSpace,Kokkos::DefaultExecutionSpace::memory_space>(1000000);
}

}  // namespace Test*/
