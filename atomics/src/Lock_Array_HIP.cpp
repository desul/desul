/*
Copyright (c) 2019, Lawrence Livermore National Security, LLC
and DESUL project contributors. See the COPYRIGHT file for details.
Source: https://github.com/desul/desul

SPDX-License-Identifier: (BSD-3-Clause)
*/

#include <cinttypes>
#include <desul/atomics/Lock_Array.hpp>
#include <sstream>
#include <string>

#ifdef DESUL_ATOMICS_ENABLE_HIP_SEPARABLE_COMPILATION
namespace desul {
namespace Impl {
__device__ __constant__ int32_t* HIP_SPACE_ATOMIC_LOCKS_DEVICE = nullptr;
__device__ __constant__ int32_t* HIP_SPACE_ATOMIC_LOCKS_NODE   = nullptr;
}  // namespace Impl
}  // namespace desul
#endif

namespace desul {

namespace {

__global__ void init_lock_arrays_hip_kernel() {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HIP_SPACE_ATOMIC_MASK + 1) {
    Impl::HIP_SPACE_ATOMIC_LOCKS_DEVICE[i] = 0;
    Impl::HIP_SPACE_ATOMIC_LOCKS_NODE[i]   = 0;
  }
}

}  // namespace

namespace Impl {

std::unordered_map<int, int32_t*> HIP_SPACE_ATOMIC_LOCKS_DEVICE_h = {};
std::unordered_map<int, int32_t*> HIP_SPACE_ATOMIC_LOCKS_NODE_h   = {};

// Putting this into anonymous namespace so we don't have multiple defined
// symbols When linking in more than one copy of the object file
namespace {

void check_error_and_throw_hip(hipError_t e, const std::string msg) {
  if (e != hipSuccess) {
    std::ostringstream out;
    out << "Desul::Error: " << msg << " error(" << hipGetErrorName(e)
        << "): " << hipGetErrorString(e);
    throw std::runtime_error(out.str());
  }
}

}  // namespace

template <typename T>
void init_lock_arrays_hip(int device_id) {
  if (HIP_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id] != nullptr) return;
  auto error_set_device = hipSetDevice(device_id);
  check_error_and_throw_hip(error_set_device,
                            "init_lock_arrays_hip: hipSetDevice");
  auto error_malloc1 = hipMalloc(&HIP_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id],
                                 sizeof(int32_t) * (HIP_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_hip(error_malloc1,
                            "init_lock_arrays_hip: hipMalloc device locks");

  auto error_malloc2 =
      hipHostMalloc(&HIP_SPACE_ATOMIC_LOCKS_NODE_h[device_id],
                    sizeof(int32_t) * (HIP_SPACE_ATOMIC_MASK + 1));
  check_error_and_throw_hip(error_malloc2,
                            "init_lock_arrays_hip: hipMallocHost host locks");

  auto error_sync1 = hipDeviceSynchronize();
  copy_hip_lock_arrays_to_device(device_id);
  check_error_and_throw_hip(error_sync1, "init_lock_arrays_hip: post malloc");

  init_lock_arrays_hip_kernel<<<(HIP_SPACE_ATOMIC_MASK + 1 + 255) / 256,
                                256>>>();

  auto error_sync2 = hipDeviceSynchronize();
  check_error_and_throw_hip(error_sync2, "init_lock_arrays_hip: post init");
}

template <typename T>
void finalize_lock_arrays_hip() {
  for (auto& host_device_lock_arrays : CUDA_SPACE_ATOMIC_LOCKS_DEVICE_h) {
    if (HIP_SPACE_ATOMIC_LOCKS_DEVICE_h == nullptr) continue;
    auto error_free1 = hipFree(HIP_SPACE_ATOMIC_LOCKS_DEVICE_h);
    check_error_and_throw_hip(error_free1,
                              "finalize_lock_arrays_hip: free device locks");
    auto error_free2 = hipHostFree(HIP_SPACE_ATOMIC_LOCKS_NODE_h);
    check_error_and_throw_hip(error_free2,
                              "finalize_lock_arrays_hip: free host locks");
    HIP_SPACE_ATOMIC_LOCKS_DEVICE_h[device_id] = nullptr;
    HIP_SPACE_ATOMIC_LOCKS_NODE_h[device_id]   = nullptr;
#ifdef DESUL_ATOMICS_ENABLE_HIP_SEPARABLE_COMPILATION
    copy_hip_lock_arrays_to_device(device_id);
#endif
  }
}

template void init_lock_arrays_hip<int>(int device_id);
template void finalize_lock_arrays_hip<int>();

}  // namespace Impl

}  // namespace desul
