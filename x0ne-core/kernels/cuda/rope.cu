// Per-head RMSNorm of q and k, rotate-half RoPE, and the K/V write into the cache.
#include "x0ne_kernels.h"

// grid (q_heads + kv_heads, rows), block 64: thread i owns the pair (i, i + 64) of one head.
// Row r sits at position pos0[0] + r. Blocks of k heads also copy v. A position >= max_ctx
// writes nothing into the cache. inv_freq[i] = theta^(-2i/128), computed on the host in f32.
extern "C" __global__ void x0ne_qk_norm_rope_kv_write_bf16(const bf16* __restrict__ qkv, const bf16* __restrict__ q_norm,
                                                           const bf16* __restrict__ k_norm,
                                                           const float* __restrict__ inv_freq, bf16* __restrict__ q_out,
                                                           bf16* __restrict__ k_cache, bf16* __restrict__ v_cache,
                                                           const uint32_t* __restrict__ pos0, uint32_t q_heads,
                                                           uint32_t kv_heads, uint32_t max_ctx, float eps) {
    constexpr uint32_t HD = X0NE_HEAD_DIM, HALF = HD / 2;
    __shared__ float sh[32];
    const uint32_t head = blockIdx.x, row = blockIdx.y, i = threadIdx.x;
    const uint32_t q_dim = q_heads * HD, kv_dim = kv_heads * HD, qkv_dim = q_dim + 2 * kv_dim;
    const uint32_t pos = pos0[0] + row;
    const bool is_q = head < q_heads;
    const uint32_t kh = is_q ? 0 : head - q_heads;
    const bf16* src = qkv + (size_t)row * qkv_dim + (is_q ? head * HD : q_dim + kh * HD);
    const bf16* nw = is_q ? q_norm : k_norm;

    const float a = bf2f(src[i]), b = bf2f(src[i + HALF]);
    const float total = block_sum(a * a + b * b, sh);
    const float scale = 1.0f / sqrtf(total / (float)HD + eps);
    const float na = a * scale * bf2f(nw[i]);
    const float nb = b * scale * bf2f(nw[i + HALF]);
    float s, c;
    sincosf((float)pos * inv_freq[i], &s, &c);
    const float ra = na * c - nb * s;
    const float rb = nb * c + na * s;

    if (is_q) {
        bf16* d = q_out + (size_t)row * q_dim + head * HD;
        d[i] = f2bf(ra);
        d[i + HALF] = f2bf(rb);
        return;
    }
    if (pos >= max_ctx) return;
    bf16* kd = k_cache + (size_t)pos * kv_dim + kh * HD;
    kd[i] = f2bf(ra);
    kd[i + HALF] = f2bf(rb);
    const bf16* vs = qkv + (size_t)row * qkv_dim + q_dim + kv_dim + kh * HD;
    bf16* vd = v_cache + (size_t)pos * kv_dim + kh * HD;
    vd[i] = vs[i];
    vd[i + HALF] = vs[i + HALF];
}
