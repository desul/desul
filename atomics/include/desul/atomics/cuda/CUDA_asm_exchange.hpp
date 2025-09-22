#include <limits>

#include <cuda.h>
#if (CUDA_VERSION >= 12080) && not defined(DESUL_CUDA_ARCH_IS_PRE_HOPPER)
#define DESUL_HAVE_CUDA_128BIT_CAS
#endif

namespace desul {
namespace Impl {

#include <desul/atomics/cuda/cuda_cc7_asm_exchange.inc>

#ifdef DESUL_HAVE_CUDA_128BIT_CAS
// Hopper (CC90) and above has some 128 bit atomic support
#include <desul/atomics/cuda/cuda_cc9_asm_exchange.inc>
#endif
}
}  // namespace desul
