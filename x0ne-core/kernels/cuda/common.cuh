// Shared device helpers. Reductions use a fixed order, so a launch is bit-reproducible.
#pragma once
#include <math.h>
#include <stdint.h>
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

// Every attention and RoPE kernel is written for this head size (all Qwen3 dense models).
#define X0NE_HEAD_DIM 128

// Split-K decode attention (E3): keys per tile, and floats per partial state (m, l, 2 unused,
// then o[128], so o starts 16-byte aligned).
#define X0NE_SPLIT_TILE 128
#define X0NE_SPLIT_STATE (X0NE_HEAD_DIM + 4)

// Sparse MoE (GM1): largest expert count of the routing kernel, and the index of no expert.
#define X0NE_MOE_MAX_EXPERTS 1024
#define X0NE_MOE_NO_EXPERT 0xFFFFFFFFu

__device__ __forceinline__ float bf2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2bf(float x) { return __float2bfloat16_rn(x); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the block (blockDim.x a multiple of 32, at most 1024); every thread gets it.
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp_sum(lane < nw ? sh[lane] : 0.f);
}
