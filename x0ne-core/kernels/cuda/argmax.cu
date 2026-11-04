// Greedy pick: index of the largest logit of each row, lowest index on ties (torch.argmax).
#include "x0ne_kernels.h"

__device__ __forceinline__ bool better(float v, uint32_t i, float bv, uint32_t bi) {
    return v > bv || (v == bv && i < bi);
}

// grid (rows), block 1024. A row holding any NaN yields 0xFFFFFFFF (2026-10-21 review, C3:
// torch.argmax returns the NaN's index; here the host refuses the token). out[r] is written
// to device memory: the next decode step can read it as its token.
extern "C" __global__ void x0ne_argmax_f32(const float* __restrict__ logits, uint32_t n, uint32_t* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ uint32_t si[32];
    const float* row = logits + (size_t)blockIdx.x * n;
    float bv = -INFINITY;
    uint32_t bi = 0xFFFFFFFFu;
    bool nan = false;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = row[i];
        nan |= v != v;
        if (better(v, i, bv, bi)) {
            bv = v;
            bi = i;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    if (__syncthreads_or(nan)) {
        if (threadIdx.x == 0) out[blockIdx.x] = 0xFFFFFFFFu;
        return;
    }
    if (warp == 0) {
        bv = lane < nw ? sv[lane] : -INFINITY;
        bi = lane < nw ? si[lane] : 0xFFFFFFFFu;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) out[blockIdx.x] = bi;
    }
}
