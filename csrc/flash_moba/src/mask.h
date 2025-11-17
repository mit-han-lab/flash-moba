/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/mask.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_MOBA_NAMESPACE {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor, const int max_seqlen_k,
                                  const int col_idx_offset_ = 0) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k) {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi) {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}


template <typename Engine, typename Layout, typename IdxTensor>
__forceinline__ __device__ void apply_moba_mask_causal(Tensor<Engine, Layout> &tensor, const int col_idx_offset_,
                                        const int max_seqlen_k, const int relative_row_idx_offset,
                                        const int max_seqlen_q, const int warp_row_stride, 
                                        const IdxTensor& sMobaIdx,
                                        const int idx_count) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = relative_row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            // const int row_idx = row_idx_base + i * 8;
            const int logical_row_idx = row_idx_base + i * 8;
            const int row_idx = logical_row_idx < idx_count ? sMobaIdx(logical_row_idx) : 0;
            const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q + 1);
            const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    if (logical_row_idx >= idx_count) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    } else {
                        if (col_idx >= col_idx_limit_right) {
                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
}

template <bool Is_causal, bool Is_local, bool Has_alibi>
struct Mask {

    const int max_seqlen_k, max_seqlen_q;
    const int window_size_left, window_size_right;
    const float alibi_slope;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q,
                                    const int window_size_left, const int window_size_right,
                                    const float alibi_slope=0.f)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , alibi_slope(!Has_alibi ? 0.0 : alibi_slope) {
    };

    // Apply mask for MOBA sparse pattern
    // Causal_mask: whether this particular iteration needs causal masking
    // sMobaIdx: shared memory tensor containing the actual row indices for this block
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout, typename IdxTensor>
    __forceinline__ __device__ void apply_moba_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int relative_row_idx_offset,
                                               const int warp_row_stride,
                                               const IdxTensor& sMobaIdx,
                                               const int idx_count,
                                               const int idx_offset) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Has_alibi || Causal_mask || !Is_even_MN;
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), FLASH_MOBA_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !(Has_alibi && !Is_causal) && !Causal_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            // No causal, no local
                            if constexpr (Has_alibi) {
                                tensor(mi, make_coord(j, nj)) += alibi_slope * col_idx;
                            }
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = relative_row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        // Get the actual row index from sMobaIdx instead of computing it
                        // const int ref = relative_row_idx_offset + mi * warp_row_stride + i * 8;
                        const int logical_row_idx = row_idx_base + i * 8;
                        const int row_idx = logical_row_idx < idx_count ? sMobaIdx(logical_row_idx + idx_offset) : 0;
                        const int col_idx_limit_left = std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q + window_size_right);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                const int col_idx = col_idx_base + j;
                                if (logical_row_idx >= idx_count) {
                                    tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                } else {
                                    if constexpr (Has_alibi) {
                                        if constexpr (Is_causal) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) += alibi_slope * col_idx;
                                        } else {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) -= alibi_slope * abs(row_idx + max_seqlen_k - max_seqlen_q - col_idx);
                                        }
                                    }
                                    if constexpr (Causal_mask) {
                                        if (col_idx >= col_idx_limit_right) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                    if constexpr (!Causal_mask && !Is_even_MN) {
                                        // Causal and Local already handles MN masking
                                        if (col_idx >= max_seqlen_k) {
                                            tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };
};

} // namespace FLASH_MOBA_NAMESPACE
