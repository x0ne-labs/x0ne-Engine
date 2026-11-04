// Position advance: the last node of the decode graph. pos[0] += 1, one thread.
#include "x0ne_kernels.h"

// grid (1), block 32. The position never leaves the device between two decode steps.
extern "C" __global__ void x0ne_advance_u32(uint32_t* __restrict__ pos) {
    if (threadIdx.x == 0 && blockIdx.x == 0) pos[0] += 1u;
}
