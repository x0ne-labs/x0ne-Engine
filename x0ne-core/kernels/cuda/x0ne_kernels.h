// x0ne Core kernel ABI: C linkage, so the cubin symbol is the kernel name.
// Conventions: row-major bf16, f32 accumulation, positions and token ids read from device
// memory, bounds checked without trap. Launch shapes are documented at each definition.
#pragma once
#include "common.cuh"

extern "C" __global__ void x0ne_probe_iota_u32(uint32_t* out, uint32_t n);

extern "C" __global__ void x0ne_embedding_bf16(const bf16* table, const uint32_t* tokens, bf16* out, uint32_t hidden,
                                               uint32_t vocab);

extern "C" __global__ void x0ne_add_rmsnorm_bf16(float* residual, const bf16* x, const bf16* w, bf16* out,
                                                 uint32_t hidden, float eps);

extern "C" __global__ void x0ne_qk_norm_rope_kv_write_bf16(const bf16* qkv, const bf16* q_norm, const bf16* k_norm,
                                                           const float* inv_freq, bf16* q_out, bf16* k_cache,
                                                           bf16* v_cache, const uint32_t* pos0, uint32_t q_heads,
                                                           uint32_t kv_heads, uint32_t max_ctx, float eps);

extern "C" __global__ void x0ne_attention_decode_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache, bf16* out,
                                                      const uint32_t* pos, uint32_t q_heads, uint32_t kv_heads,
                                                      uint32_t max_ctx, float scale);

extern "C" __global__ void x0ne_attention_decode_split_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                            float* state, const uint32_t* pos, uint32_t q_heads,
                                                            uint32_t kv_heads, uint32_t max_ctx, uint32_t splits,
                                                            float scale);

extern "C" __global__ void x0ne_attention_decode_split_g1_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                               float* state, uint32_t* counter, bf16* out,
                                                               const uint32_t* pos, uint32_t q_heads, uint32_t kv_heads,
                                                               uint32_t max_ctx, uint32_t splits, float scale);

extern "C" __global__ void x0ne_attention_decode_split_g2_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                               float* state, uint32_t* counter, bf16* out,
                                                               const uint32_t* pos, uint32_t q_heads, uint32_t kv_heads,
                                                               uint32_t max_ctx, uint32_t splits, float scale);

extern "C" __global__ void x0ne_attention_decode_split_g4_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                               float* state, uint32_t* counter, bf16* out,
                                                               const uint32_t* pos, uint32_t q_heads, uint32_t kv_heads,
                                                               uint32_t max_ctx, uint32_t splits, float scale);

extern "C" __global__ void x0ne_attention_decode_split_g8_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                               float* state, uint32_t* counter, bf16* out,
                                                               const uint32_t* pos, uint32_t q_heads, uint32_t kv_heads,
                                                               uint32_t max_ctx, uint32_t splits, float scale);

extern "C" __global__ void x0ne_attention_merge_bf16(const float* state, bf16* out, const uint32_t* pos,
                                                     uint32_t max_ctx, uint32_t splits);

extern "C" __global__ void x0ne_attention_prefill_bf16(const bf16* q, const bf16* k_cache, const bf16* v_cache,
                                                       bf16* out, const uint32_t* pos0, uint32_t rows, uint32_t q_heads,
                                                       uint32_t kv_heads, uint32_t max_ctx, float scale);

extern "C" __global__ void x0ne_swiglu_bf16(const bf16* gate_up, bf16* out, uint32_t inter);

extern "C" __global__ void x0ne_argmax_f32(const float* logits, uint32_t n, uint32_t* out);

extern "C" __global__ void x0ne_advance_u32(uint32_t* pos);

extern "C" __global__ void x0ne_moe_route_f32(const float* logits, uint32_t experts, uint32_t top_k, uint32_t* ids,
                                              float* weights);

extern "C" __global__ void x0ne_moe_gemv_bf16(const bf16* w, const uint32_t* ids, const bf16* x, bf16* y, uint32_t rows,
                                              uint32_t cols, uint32_t experts, uint32_t x_group, uint32_t x_stride);

extern "C" __global__ void x0ne_moe_gemv_f32(const bf16* w, const uint32_t* ids, const bf16* x, float* y, uint32_t rows,
                                             uint32_t cols, uint32_t experts, uint32_t x_group, uint32_t x_stride);

extern "C" __global__ void x0ne_moe_combine_bf16(const float* y, const float* weights, bf16* out, uint32_t hidden,
                                                 uint32_t top_k);
