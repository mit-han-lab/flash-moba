/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo and Kasra Mazaheri from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_topk_kernel.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
// #include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "mask.h"

namespace FLASH_TOPK_NAMESPACE {

using namespace cute;


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_fused_topk_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kTopk = Kernel_traits::kTopk;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int topk = params.moba_topk;
    const int kChunkSize = params.moba_chunk_size;
    
    const TopKBlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = 0;
    int chunk_block_max = binfo.actual_seqlen_km;
    int n_block_max = cute::ceil_div(chunk_block_max, kBlockN);
    if (Is_causal) {
        chunk_block_max = cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kChunkSize);
        n_block_max = std::min(n_block_max, cute::ceil_div(chunk_block_max, kBlockN));
    }
    int n_block = n_block_max - 1;

    // const index_t row_offset_q = binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
    //     + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;

    const index_t row_offset_km = binfo.km_offset(params.km_batch_stride, params.km_row_stride, bidb)
        + n_block * kBlockN * params.km_row_stride + bidh * params.km_head_stride;
    

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                            + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));

    Tensor gQ = local_tile(mQ(_, bidh, _), 
                           Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)


    Tensor mKM = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.km_ptr) 
                             + binfo.km_offset(params.km_batch_stride, params.km_row_stride, bidb)),
                             make_shape(binfo.actual_seqlen_km, params.h_k, params.d),
                             make_stride(params.km_row_stride, params.km_head_stride, _1{}));

    Tensor gKM = local_tile(mKM(_, bidh / params.h_h_k_ratio, _), 
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    // Shared memory tensor for storing one block at a time
    Tensor sIdx = make_tensor(make_smem_ptr(reinterpret_cast<int *>(smem_)),
                              typename Kernel_traits::SmemLayoutTopK{});
    char* smem_topk_val = reinterpret_cast<char *>(smem_) + sizeof(int) * size(sIdx);
    Tensor sVal = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(smem_topk_val)),
                              typename Kernel_traits::SmemLayoutTopK{});
    char* smem_km = smem_topk_val + sizeof(ElementAccum) * size(sVal);
    Tensor sKM = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_km)),
                             typename Kernel_traits::SmemLayoutKV{});
    char* smem_q = smem_km + sizeof(Element) * size(sKM);
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_q)),
                            typename Kernel_traits::SmemLayoutQ{});
    char* smem_reduce = smem_q + (!Kernel_traits::Is_Q_in_regs ? sizeof(Element) * size(sQ) : 0); // be careful here, we're using the same smem for sK and sReduce
    Tensor sReduce = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(smem_reduce)),
                                 typename Kernel_traits::SmemLayoutTopKReduce{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKMgKM = gmem_thr_copy_QKV.partition_S(gKM);
    Tensor tKMsKM = gmem_thr_copy_QKV.partition_D(sKM);

    typename Kernel_traits::TiledMma tiled_mma; // ?
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sKM);                           // (MMA,MMA_N,MMA_K)

    // Tensor tSsR = thr_mma.partition_C(sReduce);

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sKM);

    auto smem_tiled_copy_R = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomR{}, tiled_mma);
    auto smem_thr_copy_R = smem_tiled_copy_R.get_thread_slice(tidx);
    Tensor tSsR = smem_thr_copy_R.partition_S(sReduce);
    // Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    // Tensor taccOrR = smem_thr_copy_R.retile_S(acc_s);   // dst: reg
    Tensor taccOsR = smem_thr_copy_R.partition_D(sReduce);// src: smem

    //
    // PREDICATES
    //
    
    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKM = make_identity_tensor(make_shape(size<0>(sKM), size<1>(sKM)));
    
    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKMcKM = gmem_thr_copy_QKV.partition_S(cKM);
    
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKMpKM = make_tensor<bool>(make_shape(size<2>(tKMsKM)));
    
    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKMpKM); ++k) { tKMpKM(k) = get<1>(tKMcKM(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // __syncthreads();

    FLASH_TOPK_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                        binfo.actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    FLASH_TOPK_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKMgKM(_, _, _, n_block), tKMsKM, tKMcKM, tKMpKM,
                                       binfo.actual_seqlen_km - n_block * kBlockN);
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_TOPK_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    #pragma unroll
    for (int i = tidx; i < kBlockM; i += blockDim.x) {
        #pragma unroll
        for (int j = 0; j < kTopk; ++j) {
            sIdx(i, j) = int(-1);
            sVal(i, j) = -std::numeric_limits<float>::infinity();
        }
    }

    __syncthreads();

    const int num_q_rows = Is_even_MN ? kBlockM : std::min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
    if (Is_causal) {
        for (int q_row = tidx; q_row < num_q_rows; q_row += blockDim.x) {
            const int row_idx = m_block * kBlockM + q_row;
            const int self_idx = row_idx + binfo.actual_seqlen_k - binfo.actual_seqlen_q;
            if (self_idx >= 0) {
                const int pblock = self_idx / kChunkSize;
                sIdx(q_row, 0) = pblock;
                sVal(q_row, 0) = std::numeric_limits<float>::infinity();
            }
        }
        __syncthreads();
    }


    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        FLASH_TOPK_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if (n_block > n_block_min) {
            FLASH_TOPK_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKMgKM(_, _, _, n_block - 1), tKMsKM, tKMcKM, tKMpKM);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // CUTE_STATIC_ASSERT_V(size(acc_s) == size(tSsR));
        // #pragma unroll
        // for (int i = 0; i < size(acc_s); ++i) {
        //     tSsR(i) = acc_s(i);
        // }

        Tensor taccOrR = smem_thr_copy_R.retile_S(acc_s);
        
        cute::copy(smem_tiled_copy_R, taccOrR, taccOsR);
        __syncthreads();

        for (int q_row = tidx; q_row < num_q_rows; q_row += blockDim.x) {
            const int row_idx = m_block * kBlockM + q_row;
            // Load the current topk values for this row
            // We use a simple insertion sort since topk is small
            int row_topk_indices[kTopk];
            ElementAccum row_topk_values[kTopk];
            // Replace cute::copy with explicit loop to avoid template mismatch errors
            #pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                row_topk_indices[k] = sIdx(q_row, k);
                row_topk_values[k]  = sVal(q_row, k);
            }

            // Insert the new value in the correct position if it is in the topk
            const int actual_k_elements = std::min(kBlockN, binfo.actual_seqlen_km - n_block * kBlockN);
            for (int j = actual_k_elements - 1; j >= 0; --j) {
                ElementAccum val = sReduce(q_row, j);
                int km_idx = n_block * kBlockN + j;
                const int max_k_idx = (km_idx + 1) * kChunkSize - 1;
                // const bool seqlen_km_limit = km_idx <= binfo.actual_seqlen_km - 1;
                const bool causal_limit = max_k_idx < row_idx + binfo.actual_seqlen_k - binfo.actual_seqlen_q; // < instead of <= since we're not including the current row
                const bool mask_ok = !Is_causal || causal_limit;

                if (mask_ok && val > row_topk_values[topk - 1]) {
                    row_topk_values[topk - 1] = val;
                    row_topk_indices[topk - 1] = km_idx;
                    // Bubble up the new value to its correct position
                    // #pragma unroll
                    for (int k = topk - 1; k > 0; --k) {
                        if (row_topk_values[k] > row_topk_values[k - 1]) {
                            ElementAccum temp_val = row_topk_values[k];
                            row_topk_values[k] = row_topk_values[k - 1];
                            row_topk_values[k - 1] = temp_val;
                            // Swap indices
                            int temp_idx = row_topk_indices[k];
                            row_topk_indices[k] = row_topk_indices[k - 1];
                            row_topk_indices[k - 1] = temp_idx;
                        }
                    }
                }
            }

            // Write back the updated topk values for this Q row
            // Replace cute::copy with explicit loop
            #pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                sIdx(q_row, k) = row_topk_indices[k];
                sVal(q_row, k) = row_topk_values[k];
            }
        }
        __syncthreads();
    }
    // Epilogue

    const index_t col_offset = (bidb * params.h + bidh) * params.max_lg_col_num; 
    int *col_nnz_ptr = reinterpret_cast<int *>(params.col_nnz_ptr) + col_offset;

    #pragma unroll
    for (int i = tidx; i < num_q_rows; i += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < topk; ++k) {
            int lg_col_idx = sIdx(i, k);
            if (lg_col_idx >= 0) { // If it's -1, it means no valid index
                atomicAdd(&col_nnz_ptr[lg_col_idx], 1);
            }
        }
    }

    Tensor mIdx = make_tensor(make_gmem_ptr(reinterpret_cast<int *>(params.topk_idx_ptr)
        + binfo.q_offset(params.idx_batch_stride, params.idx_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.moba_topk_rounded),
        make_stride(params.idx_row_stride, params.idx_head_stride, _1{}));

    Tensor gIdx = local_tile(mIdx(_, bidh, _), 
       Shape<Int<kBlockM>, Int<kTopk>>{},
       make_coord(m_block, 0));  // (kBlockM, kTopk)

    Tensor mVal = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.topk_val_ptr)
        + binfo.q_offset(params.val_batch_stride, params.val_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.moba_topk_rounded),
        make_stride(params.val_row_stride, params.val_head_stride, _1{}));

    Tensor gVal = local_tile(mVal(_, bidh, _), 
       Shape<Int<kBlockM>, Int<kTopk>>{},
       make_coord(m_block, 0));  // (kBlockM, kTopk)

    typename Kernel_traits::GmemTiledCopyTopKIdx gmem_tiled_copy_idx;
    auto gmem_thr_copy_idx = gmem_tiled_copy_idx.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyTopKVal gmem_tiled_copy_val;
    auto gmem_thr_copy_val = gmem_tiled_copy_val.get_thread_slice(tidx);

    Tensor tQgIdx = gmem_thr_copy_idx.partition_D(gIdx);
    Tensor tQsIdx = gmem_thr_copy_idx.partition_S(sIdx);
    Tensor tQgVal = gmem_thr_copy_val.partition_D(gVal);
    Tensor tQsVal = gmem_thr_copy_val.partition_S(sVal);

    Tensor cIdx = make_identity_tensor(make_shape(size<0>(sIdx), size<1>(sIdx)));
    Tensor cVal = make_identity_tensor(make_shape(size<0>(sVal), size<1>(sVal)));
    Tensor tQcIdx = gmem_thr_copy_idx.partition_D(cIdx);
    Tensor tQcVal = gmem_thr_copy_val.partition_D(cVal);
    Tensor tQpIdx = make_tensor<bool>(make_shape(size<2>(tQgIdx)));
    Tensor tQpVal = make_tensor<bool>(make_shape(size<2>(tQgVal)));

    #pragma unroll
    for (int k = 0; k < size(tQpIdx); ++k) { tQpIdx(k) = get<1>(tQcIdx(0, 0, k)) < params.moba_topk_rounded; }
    #pragma unroll
    for (int k = 0; k < size(tQpVal); ++k) { tQpVal(k) = get<1>(tQcVal(0, 0, k)) < params.moba_topk_rounded; }

    FLASH_TOPK_NAMESPACE::copy<Is_even_MN, false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(gmem_tiled_copy_idx, tQsIdx, tQgIdx, tQcIdx, tQpIdx, binfo.actual_seqlen_q - m_block * kBlockM);
    FLASH_TOPK_NAMESPACE::copy<Is_even_MN, false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(gmem_tiled_copy_val, tQsVal, tQgVal, tQcVal, tQpVal, binfo.actual_seqlen_q - m_block * kBlockM);
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_fused_topk(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    FLASH_TOPK_NAMESPACE::compute_fused_topk_1rowblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K>(params, bidb, bidh, m_block);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_topk_offset(const Params &params) {
    // The block index for the batch.
    const int bidb = blockIdx.x;
    // The block index for the head.
    const int bidh = blockIdx.y;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int topk = params.moba_topk;
    const int tidx = threadIdx.x;
    if (tidx != 0) return;

    const TopKBlockInfo binfo(params, bidb);

    const index_t col_offset = (bidb * params.h + bidh) * params.max_lg_col_num; 
    int *col_nnz_ptr = reinterpret_cast<int *>(params.col_nnz_ptr) + col_offset;
    index_t *col_offsets_ptr = reinterpret_cast<index_t *>(params.col_offsets_ptr) + col_offset;

    index_t cumsum = (binfo.sum_s_q * params.h + binfo.actual_seqlen_q * bidh) * topk;
    for (int i = 0; i < params.max_lg_col_num; ++i) {
        col_offsets_ptr[i] = cumsum;
        cumsum += col_nnz_ptr[i];
        col_nnz_ptr[i] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_M, bool Is_even_TopK, typename Params>
inline __device__ void compute_topk_epilogue_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kTopk = Kernel_traits::kTopk;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int topk = params.moba_topk;
    
    const TopKBlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    Tensor mIdx = make_tensor(make_gmem_ptr(reinterpret_cast<int *>(params.topk_idx_ptr)
        + binfo.q_offset(params.idx_batch_stride, params.idx_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.moba_topk_rounded),
        make_stride(params.idx_row_stride, params.idx_head_stride, _1{}));

    Tensor gIdx = local_tile(mIdx(_, bidh, _), 
       Shape<Int<kBlockM>, Int<kTopk>>{},
       make_coord(m_block, 0));  // (kBlockM, kTopk)
    
    // Shared memory tensor for storing the topk indices
    Tensor sIdx = make_tensor(make_smem_ptr(reinterpret_cast<int *>(smem_)),
                              typename Kernel_traits::SmemLayoutTopK{});

    // Gmem tiled copy for the topk indices
    typename Kernel_traits::GmemTiledCopyTopKIdx gmem_tiled_copy_idx;
    auto gmem_thr_copy_idx = gmem_tiled_copy_idx.get_thread_slice(tidx);

    Tensor tQgIdx = gmem_thr_copy_idx.partition_S(gIdx);
    Tensor tQsIdx = gmem_thr_copy_idx.partition_D(sIdx);

    Tensor cIdx = make_identity_tensor(make_shape(size<0>(sIdx), size<1>(sIdx)));
    Tensor tQcIdx = gmem_thr_copy_idx.partition_D(cIdx);
    Tensor tQpIdx = make_tensor<bool>(make_shape(size<2>(tQgIdx)));

    #pragma unroll
    for (int k = 0; k < size(tQpIdx); ++k) { tQpIdx(k) = get<1>(tQcIdx(0, 0, k)) < params.moba_topk_rounded; }

    FLASH_TOPK_NAMESPACE::copy<Is_even_M, Is_even_TopK>(gmem_tiled_copy_idx, tQgIdx, tQsIdx, tQcIdx, tQpIdx, binfo.actual_seqlen_q - m_block * kBlockM);

    cute::cp_async_fence();

    const index_t col_offset = (bidb * params.h + bidh) * params.max_lg_col_num; 
    int *col_nnz_ptr = reinterpret_cast<int *>(params.col_nnz_ptr) + col_offset;
    index_t *col_offsets_ptr = reinterpret_cast<index_t *>(params.col_offsets_ptr) + col_offset;
    int *indices_ptr = reinterpret_cast<int *>(params.indices_ptr);
    const int num_q_rows = Is_even_M ? kBlockM : std::min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);

    FLASH_TOPK_NAMESPACE::cp_async_wait<0>();
    __syncthreads();

    for (int i = tidx; i < num_q_rows; i += blockDim.x) {
        #pragma unroll
        for (int k = 0; k < topk; ++k) {
            int lg_col_idx = sIdx(i, k);
            if (lg_col_idx >= 0) { // If it's -1, it means no valid index
                int offset = atomicAdd(&col_nnz_ptr[lg_col_idx], 1);
                indices_ptr[col_offsets_ptr[lg_col_idx] + offset] = m_block * kBlockM + i;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_M, bool Is_even_TopK, typename Params>
inline __device__ void compute_topk_epilogue(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    FLASH_TOPK_NAMESPACE::compute_topk_epilogue_1rowblock<Kernel_traits, Is_even_M, Is_even_TopK>(params, bidb, bidh, m_block);
}


} // namespace FLASH_TOPK_NAMESPACE
