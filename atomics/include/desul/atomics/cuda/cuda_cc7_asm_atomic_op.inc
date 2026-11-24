#ifdef DESUL_IMPL_ATOMIC_CUDA_PTX_ISGLOBAL
#include "cuda_cc7_asm_atomic_op.inc_isglobal"
#endif

#ifdef DESUL_IMPL_ATOMIC_CUDA_PTX_PREDICATE
#include "cuda_cc7_asm_atomic_op.inc_predicate"
#endif

// Integer types that are not one of the fixed-width types handled above
#define __DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD(op) \
template<class type> \
inline __device__ typename std::enable_if<cuda_asm_is_forwarded_integer_v<type>>::type device_atomic_##op(type* dest, type value, __DESUL_IMPL_CUDA_ASM_MEMORY_ORDER order, __DESUL_IMPL_CUDA_ASM_MEMORY_SCOPE scope) { \
  device_atomic_##op(reinterpret_cast<cuda_asm_fixed_width_integer_t<type>*>(dest), value, order, scope); \
}

__DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD(add)
__DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD(sub)
__DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD(min)
__DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD(max)

#undef __DESUL_IMPL_CUDA_ASM_ATOMIC_OP_FORWARD
