// SwiGLU on the packed [gate; up] row: out[i] = silu(gate[i]) * up[i], silu(g) = g / (1 + e^-g).
#include "x0ne_kernels.h"

// grid (ceil(inter / 256), rows), block 256. For g << 0, e^-g overflows to inf and silu is -0.
extern "C" __global__ void x0ne_swiglu_bf16(const bf16* __restrict__ gate_up, bf16* __restrict__ out, uint32_t inter) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (i >= inter) return;
    const bf16* row = gate_up + (size_t)r * 2 * inter;
    const float g = bf2f(row[i]), u = bf2f(row[inter + i]);
    out[(size_t)r * inter + i] = f2bf(g / (1.0f + expf(-g)) * u);
}
