// GQA attention over the KV cache, softmax computed online in f32 (running max and sum).
// Cache layout per layer: K and V each [max_ctx][kv_heads][128] bf16. q head h reads kv head
// h / (q_heads / kv_heads). Scores are q.k * scale; the max is subtracted before every exp.
#include "x0ne_kernels.h"

#define WARPS 8
#define TILE 32

// Decode: query at pos[0] over keys 0..pos[0]. grid (q_heads), block 256: warp w scores keys
// w*32 + lane (+256 per step), then accumulates p_j * v_j with 4 of the 128 dims per lane.
// The 8 partial softmaxes are merged at the end in shared memory.
extern "C" __global__ void x0ne_attention_decode_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                      const bf16* __restrict__ v_cache, bf16* __restrict__ out,
                                                      const uint32_t* __restrict__ pos, uint32_t q_heads,
                                                      uint32_t kv_heads, uint32_t max_ctx, float scale) {
    constexpr uint32_t HD = X0NE_HEAD_DIM;
    __shared__ float qs[HD];
    __shared__ float wm[WARPS], wl[WARPS];
    __shared__ float wacc[WARPS][HD];
    const uint32_t h = blockIdx.x, kh = h / (q_heads / kv_heads), kv_dim = kv_heads * HD;
    const uint32_t ctx = min(pos[0] + 1u, max_ctx);
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < HD) qs[threadIdx.x] = bf2f(q[(size_t)h * HD + threadIdx.x]) * scale;
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t base = warp * 32; base < ctx; base += WARPS * 32) {
        const uint32_t p = base + lane;
        float s = -INFINITY;
        if (p < ctx) {
            const uint4* kr = reinterpret_cast<const uint4*>(k_cache + (size_t)p * kv_dim + kh * HD);
            float d = 0.f;
#pragma unroll
            for (uint32_t c = 0; c < HD / 8; ++c) {
                const uint4 u = kr[c];
                const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d += qs[c * 8 + j] * bf2f(e[j]);
            }
            s = d;
        }
        // Lane 0 always holds a valid key (base < ctx), so mn is finite.
        const float mn = fmaxf(m, warp_max(s));
        const float pr = p < ctx ? expf(s - mn) : 0.f;
        const float alpha = expf(m - mn);
        l = l * alpha + warp_sum(pr);
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d] *= alpha;
        const uint32_t n = min(32u, ctx - base);
#pragma unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            const float pj = __shfl_sync(0xffffffffu, pr, j);
            const uint2 vv = *reinterpret_cast<const uint2*>(v_cache + (size_t)(base + j) * kv_dim + kh * HD + lane * 4);
            const bf16* e = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[d] += pj * bf2f(e[d]);
        }
        m = mn;
    }

    if (lane == 0) {
        wm[warp] = m;
        wl[warp] = l;
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) wacc[warp][lane * 4 + d] = acc[d];
    __syncthreads();
    if (threadIdx.x < HD) {
        float mx = -INFINITY;
        for (int w = 0; w < WARPS; ++w) mx = fmaxf(mx, wm[w]);
        float sum = 0.f, a = 0.f;
        for (int w = 0; w < WARPS; ++w) {
            if (wm[w] == -INFINITY) continue;  // warp saw no key
            const float f = expf(wm[w] - mx);
            sum += wl[w] * f;
            a += wacc[w][threadIdx.x] * f;
        }
        out[(size_t)h * HD + threadIdx.x] = f2bf(a / sum);
    }
}

// Prefill: rows at positions pos0[0] + r, causal over the cache (already holding their keys).
// grid (q_heads, ceil(rows / 8)), block 256, warp w owns row blockIdx.y * 8 + w. Keys come in
// tiles of 32 through shared memory; lane j scores key j, then the warp accumulates p_j * v_j.
extern "C" __global__ void x0ne_attention_prefill_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                       const bf16* __restrict__ v_cache, bf16* __restrict__ out,
                                                       const uint32_t* __restrict__ pos0, uint32_t rows,
                                                       uint32_t q_heads, uint32_t kv_heads, uint32_t max_ctx,
                                                       float scale) {
    constexpr uint32_t HD = X0NE_HEAD_DIM, KPAD = HD + 2;  // +2: row stride of 65 words, no bank conflict
    __shared__ float qs[WARPS][HD];
    __shared__ __align__(16) bf16 ks[TILE][KPAD];
    __shared__ __align__(16) bf16 vs[TILE][HD];
    const uint32_t h = blockIdx.x, kh = h / (q_heads / kv_heads);
    const uint32_t q_dim = q_heads * HD, kv_dim = kv_heads * HD;
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const uint32_t p0 = pos0[0];
    const uint32_t r = blockIdx.y * WARPS + warp;
    const bool valid = r < rows;
    const uint32_t rlast = min(blockIdx.y * WARPS + WARPS - 1, rows - 1);
    const uint32_t nk = min(p0 + rlast + 1, max_ctx);
    const uint32_t qlim = valid ? min(p0 + r, max_ctx - 1) : 0;
    if (valid) {
        for (uint32_t d = lane; d < HD; d += 32) qs[warp][d] = bf2f(q[(size_t)r * q_dim + h * HD + d]) * scale;
    }

    float m = -INFINITY, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const __nv_bfloat162 zero2 = __floats2bfloat162_rn(0.f, 0.f);
    for (uint32_t t0 = 0; t0 < nk; t0 += TILE) {
        __syncthreads();
        for (uint32_t u = threadIdx.x; u < TILE * HD / 2; u += blockDim.x) {
            const uint32_t kr = u / (HD / 2), c2 = u % (HD / 2), key = t0 + kr;
            __nv_bfloat162 kv = zero2, vv = zero2;
            if (key < nk) {
                kv = reinterpret_cast<const __nv_bfloat162*>(k_cache + (size_t)key * kv_dim + kh * HD)[c2];
                vv = reinterpret_cast<const __nv_bfloat162*>(v_cache + (size_t)key * kv_dim + kh * HD)[c2];
            }
            reinterpret_cast<__nv_bfloat162*>(&ks[kr][0])[c2] = kv;
            reinterpret_cast<__nv_bfloat162*>(&vs[kr][0])[c2] = vv;
        }
        __syncthreads();
        if (!valid || t0 > qlim) continue;  // warp-uniform
        const uint32_t key = t0 + lane;
        float s = -INFINITY;
        if (key <= qlim) {
            float d = 0.f;
            const __nv_bfloat162* kr2 = reinterpret_cast<const __nv_bfloat162*>(&ks[lane][0]);
#pragma unroll 16
            for (uint32_t c = 0; c < HD / 2; ++c) {
                const float2 kf = __bfloat1622float2(kr2[c]);
                d += qs[warp][2 * c] * kf.x + qs[warp][2 * c + 1] * kf.y;
            }
            s = d;
        }
        // Lane 0 holds key t0 <= qlim, so mn is finite.
        const float mn = fmaxf(m, warp_max(s));
        const float pr = key <= qlim ? expf(s - mn) : 0.f;
        const float alpha = expf(m - mn);
        l = l * alpha + warp_sum(pr);
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d] *= alpha;
#pragma unroll 8
        for (uint32_t j = 0; j < TILE; ++j) {
            const float pj = __shfl_sync(0xffffffffu, pr, j);
            const uint2 vv = *reinterpret_cast<const uint2*>(&vs[j][lane * 4]);
            const bf16* e = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[d] += pj * bf2f(e[d]);
        }
        m = mn;
    }
    if (valid) {
        bf16* o = out + (size_t)r * q_dim + h * HD + lane * 4;
#pragma unroll
        for (int d = 0; d < 4; ++d) o[d] = f2bf(acc[d] / l);
    }
}

// ---- E3: split-K decode (flash-decoding), iteration 1: one block per (q head, slice) ----

#define SPLIT_WARPS (X0NE_SPLIT_TILE / 32)  // one 32-key chunk of the tile per warp

// Decode, split-K: query at pos[0] over keys 0..pos[0]. grid (q_heads, P), block 128. The keys
// come in tiles of 128 at absolute positions; block (h, y) takes tiles y, y + P, y + 2P, ...
// below ctx, and blocks without a tile exit at once. In a tile, warp w scores keys
// tile * 128 + w * 32 + lane, then accumulates p_j * v_j with 4 of the 128 dims per lane, as
// the E2 kernel does. The 4 warps merge in shared memory and the block writes its state in f32
// at state[(h * P + y) * X0NE_SPLIT_STATE]: [0] the max score m, [1] l = sum exp(s - m),
// [4..131] o = sum exp(s - m) v, not normalised. The result depends on the inputs, ctx and P only.
extern "C" __global__ void x0ne_attention_decode_split_bf16(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                                            const bf16* __restrict__ v_cache, float* __restrict__ state,
                                                            const uint32_t* __restrict__ pos, uint32_t q_heads,
                                                            uint32_t kv_heads, uint32_t max_ctx, uint32_t splits,
                                                            float scale) {
    constexpr uint32_t HD = X0NE_HEAD_DIM, TILE_K = X0NE_SPLIT_TILE;
    __shared__ float qs[HD];
    __shared__ float wm[SPLIT_WARPS], wl[SPLIT_WARPS];
    __shared__ float wacc[SPLIT_WARPS][HD];
    const uint32_t h = blockIdx.x, y = blockIdx.y, kh = h / (q_heads / kv_heads), kv_dim = kv_heads * HD;
    const uint32_t ctx = min(pos[0] + 1u, max_ctx);
    const uint32_t ntiles = (ctx + TILE_K - 1) / TILE_K;
    if (y >= ntiles) return;  // block-uniform
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < HD) qs[threadIdx.x] = bf2f(q[(size_t)h * HD + threadIdx.x]) * scale;
    __syncthreads();

    float m = -INFINITY, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint32_t t = y; t < ntiles; t += splits) {
        const uint32_t base = t * TILE_K + warp * 32;
        if (base >= ctx) continue;  // warp-uniform: part of the last tile
        const uint32_t p = base + lane;
        float s = -INFINITY;
        if (p < ctx) {
            const uint4* kr = reinterpret_cast<const uint4*>(k_cache + (size_t)p * kv_dim + kh * HD);
            float d = 0.f;
#pragma unroll
            for (uint32_t c = 0; c < HD / 8; ++c) {
                const uint4 u = kr[c];
                const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) d += qs[c * 8 + j] * bf2f(e[j]);
            }
            s = d;
        }
        // Lane 0 holds key base < ctx, so mn is finite.
        const float mn = fmaxf(m, warp_max(s));
        const float pr = p < ctx ? expf(s - mn) : 0.f;
        const float alpha = expf(m - mn);
        l = l * alpha + warp_sum(pr);
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[d] *= alpha;
        const uint32_t n = min(32u, ctx - base);
#pragma unroll 4
        for (uint32_t j = 0; j < n; ++j) {
            const float pj = __shfl_sync(0xffffffffu, pr, j);
            const uint2 vv = *reinterpret_cast<const uint2*>(v_cache + (size_t)(base + j) * kv_dim + kh * HD + lane * 4);
            const bf16* e = reinterpret_cast<const bf16*>(&vv);
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[d] += pj * bf2f(e[d]);
        }
        m = mn;
    }

    if (lane == 0) {
        wm[warp] = m;
        wl[warp] = l;
    }
#pragma unroll
    for (int d = 0; d < 4; ++d) wacc[warp][lane * 4 + d] = acc[d];
    __syncthreads();
    if (threadIdx.x < HD) {
        float mx = -INFINITY;
        for (int w = 0; w < SPLIT_WARPS; ++w) mx = fmaxf(mx, wm[w]);
        float sum = 0.f, a = 0.f;
        for (int w = 0; w < SPLIT_WARPS; ++w) {
            if (wm[w] == -INFINITY) continue;  // warp saw no key
            const float f = expf(wm[w] - mx);
            sum += wl[w] * f;
            a += wacc[w][threadIdx.x] * f;
        }
        float* st = state + ((size_t)h * splits + y) * X0NE_SPLIT_STATE;
        st[4 + threadIdx.x] = a;
        if (threadIdx.x == 0) {
            st[0] = mx;
            st[1] = sum;
        }
    }
}

// Merge of the split-K states of one q head, thread d owning dim d: reads the states of slices
// 0..active-1 in that fixed order (through L2: other blocks wrote them); every active slice saw
// at least one key, so each m is finite. FlashInfer's merge: M = max m_y, f_y = exp(m_y - M),
// l = sum l_y f_y, o = sum o_y f_y; one rounding to bf16, of o / l.
__device__ __forceinline__ void merge_head(const float* __restrict__ s, bf16* __restrict__ out_h, uint32_t active,
                                           uint32_t d) {
    float mx = -INFINITY;
    for (uint32_t y = 0; y < active; ++y) mx = fmaxf(mx, __ldcg(s + (size_t)y * X0NE_SPLIT_STATE));
    float l = 0.f, o = 0.f;
    for (uint32_t y = 0; y < active; ++y) {
        const float* st = s + (size_t)y * X0NE_SPLIT_STATE;
        const float f = expf(__ldcg(st) - mx);
        l += __ldcg(st + 1) * f;
        o += __ldcg(st + 4 + d) * f;
    }
    out_h[d] = f2bf(o / l);
}

// Merge of the split-K states: grid (q_heads), block 128, active = min(P, ceil(ctx / 128)).
extern "C" __global__ void x0ne_attention_merge_bf16(const float* __restrict__ state, bf16* __restrict__ out,
                                                     const uint32_t* __restrict__ pos, uint32_t max_ctx,
                                                     uint32_t splits) {
    constexpr uint32_t HD = X0NE_HEAD_DIM, TILE_K = X0NE_SPLIT_TILE;
    const uint32_t h = blockIdx.x, d = threadIdx.x;
    const uint32_t ctx = min(pos[0] + 1u, max_ctx);
    const uint32_t active = min(splits, (ctx + TILE_K - 1) / TILE_K);
    if (d < HD) merge_head(state + (size_t)h * splits * X0NE_SPLIT_STATE, out + (size_t)h * HD, active, d);
}

// ---- E3, iteration 2: one block per (kv head, slice), the G q heads of the GQA group together ----

#define SPLIT_KROW (X0NE_HEAD_DIM * 2 + 16)  // bytes per K row in shared memory: 256 + 16 of padding

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, uint32_t src_bytes) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
}

// Same tiles, slices, state layout and arithmetic as x0ne_attention_decode_split_bf16 (each warp's
// scores, online softmax and V sums are those of the E2 kernel, in the same order), but a block
// serves the G q heads that share kv head blockIdx.x, so each K and V row is read once per kv
// head instead of once per q head. Per 32-key chunk, a warp copies its 32 K rows into shared
// memory with 16-byte cp.async (two whole rows per instruction), issues its 32 V row loads (a
// whole row per instruction, 8 bytes per lane), then each lane scores its key from shared memory.
// grid (kv_heads, P), block 128; writes the states of q heads kh * G .. kh * G + G - 1.
// With counter != 0 (option 7), the last block of a kv head to finish merges the states of its
// G q heads into out, as x0ne_attention_merge_bf16 would (same code, same order, same bits), and
// sets counter[kh] back to 0; with counter == 0, out is unused and the merge kernel follows.
template <int G>
__device__ __forceinline__ void split_group(const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
                                            const bf16* __restrict__ v_cache, float* __restrict__ state,
                                            uint32_t* __restrict__ counter, bf16* __restrict__ out,
                                            const uint32_t* __restrict__ pos, uint32_t kv_heads, uint32_t max_ctx,
                                            uint32_t splits, float scale) {
    constexpr uint32_t HD = X0NE_HEAD_DIM, TILE_K = X0NE_SPLIT_TILE;
    // The K rows of each warp during the tile loop, then the warps' V sums (wacc) for the merge.
    constexpr uint32_t KBYTES = SPLIT_WARPS * 32 * SPLIT_KROW, ABYTES = SPLIT_WARPS * G * HD * 4;
    __shared__ __align__(16) unsigned char smem[KBYTES > ABYTES ? KBYTES : ABYTES];
    __shared__ __align__(16) float qs[G][HD];
    __shared__ float wm[SPLIT_WARPS][G], wl[SPLIT_WARPS][G];
    float(*wacc)[G][HD] = reinterpret_cast<float(*)[G][HD]>(smem);
    const uint32_t kh = blockIdx.x, y = blockIdx.y, kv_dim = kv_heads * HD;
    const uint32_t ctx = min(pos[0] + 1u, max_ctx);
    const uint32_t ntiles = (ctx + TILE_K - 1) / TILE_K;
    if (y >= ntiles) return;  // block-uniform
    const uint32_t lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (uint32_t i = threadIdx.x; i < G * HD; i += blockDim.x)
        qs[i / HD][i % HD] = bf2f(q[(size_t)kh * G * HD + i]) * scale;
    __syncthreads();

    unsigned char* kw = smem + warp * 32 * SPLIT_KROW;
    float m[G], l[G], acc[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int d = 0; d < 4; ++d) acc[g][d] = 0.f;
    }
    for (uint32_t t = y; t < ntiles; t += splits) {
        const uint32_t base = t * TILE_K + warp * 32;
        if (base >= ctx) continue;  // warp-uniform: part of the last tile
        const uint32_t n = min(32u, ctx - base);
        // K rows base + r, r = 2 i + lane / 16, 16 bytes at (lane % 16) * 16; rows past ctx are
        // zero-filled (no global read).
#pragma unroll
        for (uint32_t i = 0; i < 16; ++i) {
            const uint32_t r = 2 * i + (lane >> 4), col = (lane & 15) * 16;
            const bool in = r < n;
            const bf16* src = k_cache + (size_t)(base + (in ? r : 0)) * kv_dim + kh * HD;
            cp_async16(kw + r * SPLIT_KROW + col, reinterpret_cast<const unsigned char*>(src) + col, in ? 16u : 0u);
        }
        // V rows base + j: a whole row per load, dims lane * 4 .. lane * 4 + 3.
        uint2 vv[32];
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j)
            vv[j] = j < n ? *reinterpret_cast<const uint2*>(v_cache + (size_t)(base + j) * kv_dim + kh * HD + lane * 4)
                          : make_uint2(0u, 0u);
        cp_async_wait_all();
        __syncwarp();

        const uint32_t p = base + lane;
        float s[G];
#pragma unroll
        for (int g = 0; g < G; ++g) s[g] = -INFINITY;
        if (p < ctx) {
            const uint4* kr = reinterpret_cast<const uint4*>(kw + lane * SPLIT_KROW);
            float d[G];
#pragma unroll
            for (int g = 0; g < G; ++g) d[g] = 0.f;
#pragma unroll
            for (uint32_t c = 0; c < HD / 8; ++c) {
                const uint4 u = kr[c];
                const bf16* e = reinterpret_cast<const bf16*>(&u);
#pragma unroll
                for (uint32_t j = 0; j < 8; ++j) {
                    const float kf = bf2f(e[j]);
#pragma unroll
                    for (int g = 0; g < G; ++g) d[g] += qs[g][c * 8 + j] * kf;
                }
            }
#pragma unroll
            for (int g = 0; g < G; ++g) s[g] = d[g];
        }
        __syncwarp();  // the next chunk overwrites kw
        float pr[G];
#pragma unroll
        for (int g = 0; g < G; ++g) {
            // Lane 0 holds key base < ctx, so mn is finite.
            const float mn = fmaxf(m[g], warp_max(s[g]));
            pr[g] = p < ctx ? expf(s[g] - mn) : 0.f;
            const float alpha = expf(m[g] - mn);
            l[g] = l[g] * alpha + warp_sum(pr[g]);
#pragma unroll
            for (int d = 0; d < 4; ++d) acc[g][d] *= alpha;
            m[g] = mn;
        }
#pragma unroll
        for (uint32_t j = 0; j < 32; ++j) {
            const bf16* e = reinterpret_cast<const bf16*>(&vv[j]);
#pragma unroll
            for (int g = 0; g < G; ++g) {
                const float pj = __shfl_sync(0xffffffffu, pr[g], j);
#pragma unroll
                for (int d = 0; d < 4; ++d) acc[g][d] += pj * bf2f(e[d]);
            }
        }
    }

    __syncthreads();  // every warp is done with its K rows: smem becomes wacc
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            wm[warp][g] = m[g];
            wl[warp][g] = l[g];
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int d = 0; d < 4; ++d) wacc[warp][g][lane * 4 + d] = acc[g][d];
    }
    __syncthreads();
    if (threadIdx.x < HD) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float mx = -INFINITY;
            for (int w = 0; w < SPLIT_WARPS; ++w) mx = fmaxf(mx, wm[w][g]);
            float sum = 0.f, a = 0.f;
            for (int w = 0; w < SPLIT_WARPS; ++w) {
                if (wm[w][g] == -INFINITY) continue;  // warp saw no key
                const float f = expf(wm[w][g] - mx);
                sum += wl[w][g] * f;
                a += wacc[w][g][threadIdx.x] * f;
            }
            float* st = state + ((size_t)(kh * G + g) * splits + y) * X0NE_SPLIT_STATE;
            st[4 + threadIdx.x] = a;
            if (threadIdx.x == 0) {
                st[0] = mx;
                st[1] = sum;
            }
        }
    }
    if (counter == nullptr) return;  // grid-uniform
    __shared__ uint32_t last;
    const uint32_t active = min(splits, ntiles);
    __threadfence();  // this block's states, visible to the block that merges
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(counter + kh, 1u) == active - 1;
    __syncthreads();
    if (!last) return;  // block-uniform
    __threadfence();
    if (threadIdx.x < HD) {
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const uint32_t h = kh * G + g;
            merge_head(state + (size_t)h * splits * X0NE_SPLIT_STATE, out + (size_t)h * HD, active, threadIdx.x);
        }
    }
    if (threadIdx.x == 0) counter[kh] = 0;  // ready for the next step
}

#define X0NE_SPLIT_GROUP(G)                                                                                         \
    extern "C" __global__ void x0ne_attention_decode_split_g##G##_bf16(                                             \
        const bf16* __restrict__ q, const bf16* __restrict__ k_cache, const bf16* __restrict__ v_cache,             \
        float* __restrict__ state, uint32_t* __restrict__ counter, bf16* __restrict__ out,                          \
        const uint32_t* __restrict__ pos, uint32_t q_heads, uint32_t kv_heads, uint32_t max_ctx, uint32_t splits,   \
        float scale) {                                                                                              \
        (void)q_heads;                                                                                              \
        split_group<G>(q, k_cache, v_cache, state, counter, out, pos, kv_heads, max_ctx, splits, scale);            \
    }
X0NE_SPLIT_GROUP(1)
X0NE_SPLIT_GROUP(2)
X0NE_SPLIT_GROUP(4)
X0NE_SPLIT_GROUP(8)
