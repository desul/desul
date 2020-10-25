#define MEMORY_SCOPE MemoryScopeDevice
#include "PerfTestAtomicsNeigh_Order.inc"
#undef MEMORY_SCOPE 
#ifdef DESUL_TEST_MEMORY_SCOPE_NODE
#define MEMORY_SCOPE MemoryScopeNode
#include "PerfTestAtomicsNeigh_Order.inc"
#undef MEMORY_SCOPE
#endif
