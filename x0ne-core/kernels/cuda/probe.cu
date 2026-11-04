// Smallest kernel that proves the build path end to end: nvcc -> cubin -> load -> launch.
#include "x0ne_kernels.h"

extern "C" __global__ void x0ne_probe_iota_u32(uint32_t* out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}
