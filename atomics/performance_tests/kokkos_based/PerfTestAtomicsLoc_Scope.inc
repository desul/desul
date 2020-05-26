#define MEMORY_SCOPE MemoryScopeDevice
#include "PerfTestAtomicsLoc_Order.inc"
#undef MEMORY_SCOPE 
#define MEMORY_SCOPE MemoryScopeNode
#include "PerfTestAtomicsLoc_Order.inc"
#undef MEMORY_SCOPE
