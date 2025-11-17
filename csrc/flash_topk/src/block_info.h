/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/block_info.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
namespace FLASH_TOPK_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

struct TopKBlockInfo {

    template<typename Params>
    __device__ TopKBlockInfo(const Params &params, const int bidb)
        : sum_s_q(params.cu_seqlens_q[bidb])
        , sum_s_k(params.cu_seqlens_k[bidb])
        , sum_s_km(params.cu_seqlens_km[bidb])
        , actual_seqlen_q(params.cu_seqlens_q[bidb + 1] - sum_s_q)
        , actual_seqlen_k(params.cu_seqlens_k[bidb + 1] - sum_s_k)
        , actual_seqlen_km(params.cu_seqlens_km[bidb + 1] - sum_s_km)
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return uint32_t(sum_s_k) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t km_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return uint32_t(sum_s_km) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int sum_s_km;
    const int actual_seqlen_q;
    const int actual_seqlen_k;
    const int actual_seqlen_km;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_TOPK_NAMESPACE
