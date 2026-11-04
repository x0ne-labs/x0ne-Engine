// Sparse MoE block (GM1): routing, expert GEMVs through device-held expert indices, combination.
// A row whose routing is undefined (a non-finite router logit) gets the expert index
// X0NE_MOE_NO_EXPERT; the expert kernels then write NaN, so the logits are NaN and argmax refuses.
#include "x0ne_kernels.h"

__device__ __forceinline__ bool route_better(float v, uint32_t i, float bv, uint32_t bi) {
    return v > bv || (v == bv && i < bi);
}

// grid (rows), block 128 (a multiple of 32). experts <= X0NE_MOE_MAX_EXPERTS, top_k <= 32 and
// <= experts. Chosen experts: larger logit first, then the smaller index (-0.0 equals 0.0).
// Weights: exp(l_e - l_max) / sum over the chosen, in f64, rounded once to f32; the softmax
// denominator over every expert cancels in the renormalisation.
extern "C" __global__ void x0ne_moe_route_f32(const float* __restrict__ logits, uint32_t experts, uint32_t top_k,
                                              uint32_t* __restrict__ ids, float* __restrict__ weights) {
    __shared__ uint8_t taken[X0NE_MOE_MAX_EXPERTS];
    __shared__ float sv[32];
    __shared__ uint32_t si[32];
    __shared__ uint32_t chosen[32];
    const float* row = logits + (size_t)blockIdx.x * experts;
    uint32_t* row_ids = ids + (size_t)blockIdx.x * top_k;
    float* row_w = weights + (size_t)blockIdx.x * top_k;
    bool bad = false;
    for (uint32_t i = threadIdx.x; i < experts; i += blockDim.x) {
        taken[i] = 0;
        bad |= !isfinite(row[i]);
    }
    if (__syncthreads_or(bad)) {
        if (threadIdx.x < top_k) {
            row_ids[threadIdx.x] = X0NE_MOE_NO_EXPERT;
            row_w[threadIdx.x] = __int_as_float(0x7fc00000);
        }
        return;
    }
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (uint32_t s = 0; s < top_k; ++s) {
        float bv = -INFINITY;
        uint32_t bi = 0xFFFFFFFFu;
        for (uint32_t i = threadIdx.x; i < experts; i += blockDim.x) {
            if (!taken[i] && route_better(row[i], i, bv, bi)) {
                bv = row[i];
                bi = i;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (route_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            sv[warp] = bv;
            si[warp] = bi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = sv[0];
            uint32_t b = si[0];
            for (uint32_t w = 1; w < nw; ++w) {
                if (route_better(sv[w], si[w], v, b)) {
                    v = sv[w];
                    b = si[w];
                }
            }
            chosen[s] = b;
            taken[b] = 1;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double m = (double)row[chosen[0]];
        double e[32], sum = 0.0;
        for (uint32_t s = 0; s < top_k; ++s) {
            e[s] = exp((double)row[chosen[s]] - m);
            sum += e[s];
        }
        for (uint32_t s = 0; s < top_k; ++s) {
            row_ids[s] = chosen[s];
            row_w[s] = (float)(e[s] / sum);
        }
    }
}

__device__ __forceinline__ void put(bf16* p, float v) { *p = f2bf(v); }
__device__ __forceinline__ void put(float* p, float v) { *p = v; }

// One output row per warp: dot of row `r` of expert ids[slot] ([experts][rows][cols] bf16) with
// the slot's input x + (slot / x_group) * x_stride, f32 lanes then a fixed-order warp sum.
// cols % 8 == 0 and 16-byte aligned w and x (8 bf16 per load).
template <typename Out>
__device__ __forceinline__ void moe_gemv_row(const bf16* __restrict__ w, const uint32_t* __restrict__ ids,
                                             const bf16* __restrict__ x, Out* __restrict__ y, uint32_t rows,
                                             uint32_t cols, uint32_t experts, uint32_t x_group,
                                             uint32_t x_stride) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint32_t slot = blockIdx.y;
    if (r >= rows) return;
    const uint32_t e = ids[slot];
    Out* out = y + (size_t)slot * rows + r;
    if (e >= experts) {
        if (lane == 0) put(out, __int_as_float(0x7fc00000));
        return;
    }
    const uint4* wr = reinterpret_cast<const uint4*>(w + ((size_t)e * rows + r) * cols);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(slot / x_group) * x_stride);
    float acc = 0.f;
    for (uint32_t c = lane; c < cols / 8; c += 32) {
        const uint4 a = wr[c], b = xr[c];
        const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(&a);
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 fa = __bfloat1622float2(a2[j]), fb = __bfloat1622float2(b2[j]);
            acc = fmaf(fa.x, fb.x, acc);
            acc = fmaf(fa.y, fb.y, acc);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) put(out, acc);
}

// grid (ceil(rows / 8), slots), block 256. y [slots][rows] bf16 (the [gate; up] rows).
extern "C" __global__ void x0ne_moe_gemv_bf16(const bf16* __restrict__ w, const uint32_t* __restrict__ ids,
                                              const bf16* __restrict__ x, bf16* __restrict__ y, uint32_t rows,
                                              uint32_t cols, uint32_t experts, uint32_t x_group,
                                              uint32_t x_stride) {
    moe_gemv_row<bf16>(w, ids, x, y, rows, cols, experts, x_group, x_stride);
}

// grid (ceil(rows / 8), slots), block 256. y [slots][rows] f32 (the down rows, kept in f32 for
// the combination).
extern "C" __global__ void x0ne_moe_gemv_f32(const bf16* __restrict__ w, const uint32_t* __restrict__ ids,
                                             const bf16* __restrict__ x, float* __restrict__ y, uint32_t rows,
                                             uint32_t cols, uint32_t experts, uint32_t x_group,
                                             uint32_t x_stride) {
    moe_gemv_row<float>(w, ids, x, y, rows, cols, experts, x_group, x_stride);
}

// grid (ceil(hidden / 256), rows), block 256. out[r][i] = bf16(sum_s weights[r][s] y[r][s][i]),
// f32 fused multiply-adds in slot order, one rounding.
extern "C" __global__ void x0ne_moe_combine_bf16(const float* __restrict__ y, const float* __restrict__ weights,
                                                 bf16* __restrict__ out, uint32_t hidden, uint32_t top_k) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= hidden) return;
    const float* wr = weights + (size_t)r * top_k;
    const float* yr = y + (size_t)r * top_k * hidden + i;
    float acc = 0.f;
    for (uint32_t s = 0; s < top_k; ++s) acc = fmaf(wr[s], yr[(size_t)s * hidden], acc);
    out[(size_t)r * hidden + i] = f2bf(acc);
}
