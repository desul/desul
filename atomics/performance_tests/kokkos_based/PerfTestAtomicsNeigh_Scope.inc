#define MEMORY_SCOPE MemoryScopeDevice
#include "PerfTestAtomicsNeigh_Order.inc"
#undef MEMORY_SCOPE 
#define MEMORY_SCOPE MemoryScopeNode
#include "PerfTestAtomicsNeigh_Order.inc"
#undef MEMORY_SCOPE
