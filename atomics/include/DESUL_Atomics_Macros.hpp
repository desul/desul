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

#ifndef DESUL_ATOMICS_MACROS_HPP_
#define DESUL_ATOMICS_MACROS_HPP_

// Macros

#if defined(__GNUC__) && (!defined(__CUDA_ARCH__) || !defined(__NVCC__))
#define DESUL_HAVE_GCC_ATOMICS
#endif

#ifdef _MSC_VER
#define DESUL_HAVE_MSVC_ATOMICS
#endif

#ifdef __CUDACC__
#define DESUL_HAVE_CUDA_ATOMICS
#endif

#ifdef __CUDA_ARCH__
#define DESUL_HAVE_GPU_LIKE_PROGRESS
#endif

#ifdef DESUL_HAVE_CUDA_ATOMICS
#define DESUL_FORCEINLINE_FUNCTION inline __host__ __device__
#define DESUL_INLINE_FUNCTION inline __host__ __device__
#define DESUL_FUNCTION __host__ __device__
#else
#define DESUL_FORCEINLINE_FUNCTION inline
#define DESUL_INLINE_FUNCTION inline
#define DESUL_FUNCTION
#endif

#if !defined(__CUDA_ARCH__)
#define DESUL_HAVE_FORWARD_PROGRESS
#endif

#endif  // DESUL_ATOMICS_MACROS_HPP_
