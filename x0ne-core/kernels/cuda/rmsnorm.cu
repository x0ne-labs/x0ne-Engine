// Residual add then RMSNorm, one row per block. The residual stream stays in f32.
#include "x0ne_kernels.h"

// grid (rows), block 256. residual[r] += x[r]; out[r] = bf16(residual[r] / rms * w).
extern "C" __global__ void x0ne_add_rmsnorm_bf16(float* __restrict__ residual, const bf16* __restrict__ x,
                                                 const bf16* __restrict__ w, bf16* __restrict__ out, uint32_t hidden,
                                                 float eps) {
    __shared__ float sh[32];
    const size_t base = (size_t)blockIdx.x * hidden;
    float ss = 0.f;
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float r = residual[base + i] + bf2f(x[base + i]);
        residual[base + i] = r;
        ss += r * r;
    }
    const float total = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(total / (float)hidden + eps);
    for (uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        out[base + i] = f2bf(residual[base + i] * scale * bf2f(w[i]));
    }
}
