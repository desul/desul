#define MEMORY_SCOPE MemoryScopeDevice
#include "PerfTestAtomicsLoc_Order.inc"
#undef MEMORY_SCOPE 
#ifdef DESUL_TEST_SCOPE_NODE
#define MEMORY_SCOPE MemoryScopeNode
#include "PerfTestAtomicsLoc_Order.inc"
#undef MEMORY_SCOPE
#endif
