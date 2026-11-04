// Embedding lookup: out[r] = table[tokens[r]]. Token ids are read from device memory.
#include "x0ne_kernels.h"

// grid (rows), block 128. hidden % 8 == 0 (16-byte copies). An id >= vocab writes a zero row.
extern "C" __global__ void x0ne_embedding_bf16(const bf16* __restrict__ table, const uint32_t* __restrict__ tokens,
                                               bf16* __restrict__ out, uint32_t hidden, uint32_t vocab) {
    const uint32_t row = blockIdx.x;
    const uint32_t t = tokens[row];
    uint4* dst = reinterpret_cast<uint4*>(out + (size_t)row * hidden);
    const uint32_t n = hidden / 8;
    if (t >= vocab) {
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = make_uint4(0, 0, 0, 0);
        return;
    }
    const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)t * hidden);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}
