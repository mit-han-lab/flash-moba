/******************************************************************************
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once
#include <torch/extension.h>
#include "namespace_config.h"

namespace FLASH_TOPK_NAMESPACE {

struct Flash_topk_params {
    using index_t = int64_t;

    int d, d_rounded;

    // The number of heads.
    int h, h_k;
    int h_h_k_ratio; // precompute h / h_k,
    int moba_topk, moba_topk_rounded, moba_chunk_size;
};

struct Fused_topk_params : public Flash_topk_params {
    void *__restrict__ q_ptr;
    void * __restrict__ k_ptr;

    index_t q_batch_stride;
    index_t q_row_stride;
    index_t q_head_stride;

    index_t k_batch_stride;
    index_t k_row_stride;
    index_t k_head_stride;

    void * __restrict__ topk_idx_ptr; // indices pointer to the topk indices
    void * __restrict__ topk_val_ptr; // values pointer to the topk best values
    
    void * __restrict__ km_ptr; // pointer to the k-mean values for moba 

    // The stride between rows of indices and values matrices.
    index_t idx_batch_stride;
    index_t idx_row_stride;
    index_t idx_head_stride;
    index_t val_batch_stride;
    index_t val_row_stride;
    index_t val_head_stride;
    index_t km_batch_stride;
    index_t km_row_stride;
    index_t km_head_stride;

    // Sparse pattern data structures
    void * __restrict__ col_offsets_ptr; // [batch_size, num_heads, max_lg_col_num]
    void * __restrict__ col_nnz_ptr; // [batch_size, num_heads, 
    void * __restrict__ indices_ptr; // [total_selected_rows]
    
    // Context information
    int max_lg_col_num;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_km, seqlen_q_rounded, seqlen_k_rounded, seqlen_km_rounded, total_q;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ cu_seqlens_km;

    bool is_bf16;
    bool is_causal;
};


torch::Tensor varlen_sort(torch::Tensor cu_seq_starts, torch::Tensor cu_seq_ends, torch::Tensor src);

template<typename T, int Headdim, int Topk, bool Is_causal> void run_fused_topk_(Fused_topk_params &params, cudaStream_t stream);

} // namespace FLASH_TOPK_NAMESPACE
