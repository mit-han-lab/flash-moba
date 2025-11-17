/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Junxian Guo from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h
 * Copyright (c) 2025, FlashMoBA Team.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "philox_unpack.cuh" // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"
#include "index_gather_scatter.h"

namespace FLASH_MOBA_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}

// Enum to specify which row statistic to access
enum class RowStatType {
    SUM,
    MAX
};

// Generic function to get row statistic tiles (sum or max)
template<RowStatType StatType, typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_row_stat_tile(const Params &params, const int bidb, const int bidh, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // Select the appropriate pointer based on StatType
        void* base_ptr = (StatType == RowStatType::SUM) ? params.row_sum_ptr : params.row_max_ptr;
        auto gmem_ptr_row_stat = make_gmem_ptr(reinterpret_cast<ElementAccum*>(base_ptr));

        Tensor mRowStat = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(base_ptr)),
                                      make_shape(params.b, params.h, _4{}, params.seqlen_q_rounded),
                                      make_stride(params.row_stat_batch_stride, params.row_stat_head_stride, params.row_stat_row_stride, _1{}));
        // auto mRowStat_slice = mRowStat(bidb, bidh, _);

        // Tensor t0 = local_tile(mRowStat(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(0));
        // Tensor t1 = local_tile(mRowStat(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(_));
        // Tensor t2 = t1(_, _);
        // Tensor t3 = t1(_, 0);

        return local_tile(mRowStat(bidb, bidh, threadIdx.x % 4, _), Shape<Int<kBlockM>>{}, make_coord(_));
}


// Helper function to load row indices from global memory to shared memory
template<int kMxLgBlockM, typename IdxTensor>
__forceinline__ __device__ void load_row_indices_to_smem(const int* row_indices, 
                                                          const int count, 
                                                          IdxTensor& sMobaIdx) {
    // Copy valid indices and pad with 0 for out-of-bounds accesses
    for (int idx = threadIdx.x; idx < count && idx < kMxLgBlockM; idx += blockDim.x) {
        sMobaIdx(idx) = row_indices[idx];
    }
    __syncthreads();  // Ensure the cache is fully populated before use
}

// Helper function to create gather tensor for Q with cached indices
template<typename Element, int kBlockM, int kHeadDim, typename Params>
__forceinline__ __device__ auto make_gather_tensor_Q(const Params& params, 
                                                      Element* q_head_ptr,
                                                      const int count,
                                                      const int* cached_row_indices) {
    // Create 1D gather tensor for rows
    auto gQ_rows = make_gather_tensor(
        make_gmem_ptr(q_head_ptr),
        make_shape(count, params.d),
        make_stride(params.q_row_stride, _1{}),
        IndexedGather(cached_row_indices)
    );
    
    // Create 2D view with proper shape
    return local_tile(gQ_rows, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(0, 0));
}


template<typename ElementAccum, int kBlockM, int kHeadDim, typename Params>
__forceinline__ __device__ auto make_gather_tensor_Otmp(const Params& params, 
                                                      ElementAccum* o_tmp_head_ptr,
                                                      const int count,
                                                      const int* cached_row_indices) {
    // Create 1D gather tensor for rows
    auto gOtmp_rows = make_gather_tensor(
        make_gmem_ptr(o_tmp_head_ptr),
        make_shape(count, params.d),
        // Shape<Int<kBlockM>, Int<kHeadDim>>{}, 
        make_stride(params.o_tmp_row_stride, _1{}),
        IndexedGather(cached_row_indices)
    );
    
    // Create 2D view with proper shape
    return local_tile(gOtmp_rows, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(0, 0));
}

template<int kBlockM, int kHeadDim, class ThrMMA>
CUTE_HOST_DEVICE auto make_acc_row(const ThrMMA &thr_mma) {
    auto caccO = make_identity_tensor(
        Shape<Int<kBlockM>, Int<kHeadDim>>{});
    auto taccOcO = thr_mma.partition_C(caccO);
    return logical_divide(taccOcO, Shape<_2>{})
           (make_coord(0, _), _, 0);
}

template<bool Is_even_MN, bool Is_even_K,
         typename SmemTiledCopyOtmp,
         typename SmemThrCopyOtmp,
         typename GmemTiledCopyOtmp,
         typename GmemThrCopyOtmp,
         typename TensorAccO,
         typename TensorSmemOtmp,
         typename TensorGmemOtmp,
         typename TOcOtmp,
         typename TOpOtmp>
__forceinline__ __device__ void
o_tmp_store(
    const SmemTiledCopyOtmp& smem_tiled_copy_Otmp,
    const SmemThrCopyOtmp& smem_thr_copy_Otmp,
    const GmemTiledCopyOtmp& gmem_tiled_copy_Otmp,
    const GmemThrCopyOtmp& gmem_thr_copy_Otmp,
    TensorAccO&     acc_o,
    TensorSmemOtmp& sOtmp,
    TensorGmemOtmp& gOtmp,
    const TOcOtmp& tOcOtmp,
    const TOpOtmp& tOpOtmp,
    const int count
) {
    //------------------------------------------------------------------//
    // 1. Registers -> Shared memory (stage 到 sOtmp)                    //
    //------------------------------------------------------------------//
    Tensor taccOrOtmp = smem_thr_copy_Otmp.retile_S(acc_o);   // src: regs
    Tensor taccOsOtmp = smem_thr_copy_Otmp.partition_D(sOtmp);// dst: smem

    cute::copy(smem_tiled_copy_Otmp, taccOrOtmp, taccOsOtmp);

    __syncthreads();   // 确保 sOtmp 写完

    //------------------------------------------------------------------//
    // 2. Shared memory -> Global memory                                //
    //------------------------------------------------------------------//
    Tensor tOsOtmp = gmem_thr_copy_Otmp.partition_S(sOtmp);   // src: smem
    Tensor tOgOtmp = gmem_thr_copy_Otmp.partition_D(gOtmp);   // dst: gmem

    // Clear_OOB_* 设为 false：不向 out-of-bounds 位置写零
    FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K,
                          /*Clear_OOB_MN=*/false,
                          /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Otmp,
        tOsOtmp, tOgOtmp,
        tOcOtmp, tOpOtmp,
        count);
}

template<bool Is_even_MN, bool Is_even_K,
         typename SmemTiledCopyOtmp,
         typename SmemThrCopyOtmp,
         typename GmemTiledCopyOtmp,
         typename GmemThrCopyOtmp,
         typename TensorAccO,
         typename TensorSmemOtmp,
         typename TensorGmemOtmp,
         typename TOcOtmp,
         typename TOpOtmp>
__forceinline__ __device__ void
o_tmp_load(
    const SmemTiledCopyOtmp& smem_tiled_copy_Otmp,
    const SmemThrCopyOtmp& smem_thr_copy_Otmp,
    const GmemTiledCopyOtmp& gmem_tiled_copy_Otmp,
    const GmemThrCopyOtmp& gmem_thr_copy_Otmp,
    TensorAccO&     acc_o,
    TensorSmemOtmp& sOtmp,
    TensorGmemOtmp& gOtmp,
    const TOcOtmp& tOcOtmp,
    const TOpOtmp& tOpOtmp,
    const int count
) {
    Tensor tOsOtmp = gmem_thr_copy_Otmp.partition_S(sOtmp);  // dst: smem
    Tensor tOgOtmp = gmem_thr_copy_Otmp.partition_D(gOtmp);  // src: gmem

    FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K,
                          /*Clear_OOB_MN=*/false,
                          /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_Otmp,
        tOgOtmp,  /*src*/
        tOsOtmp,  /*dst*/
        tOcOtmp, tOpOtmp,
        count);

    __syncthreads();                    // 保证 smem 中数据已准备好

    Tensor taccOrOtmp = smem_thr_copy_Otmp.retile_S(acc_o);   // dst: reg
    Tensor taccOsOtmp = smem_thr_copy_Otmp.partition_D(sOtmp);// src: smem

    cute::copy(smem_tiled_copy_Otmp, taccOsOtmp, taccOrOtmp);
    __syncthreads(); 
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_moba_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_lg_block) {

    // const int m_block = m_lg_block;

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kMxLgBlockM = Kernel_traits::kMxLgBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    const int kLgBlockM = params.m_lg_block_dim;
    const int kLgBlockN = params.n_lg_block_dim;

    const int m_sub_loop = kLgBlockM / kBlockM;
    const int n_sub_loop = kLgBlockN / kBlockN;


    // constexpr int lg_m_block_dim = params.lg_m_block_dim;
    // constexpr int lg_n_block_dim = params.lg_n_block_dim;


    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    FLASH_MOBA_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_lg_block * kLgBlockM >= binfo.actual_seqlen_q) return;


    const int m_block_base = m_lg_block * m_sub_loop;

    const int n_lg_block_min = 0;
    int n_lg_block_max = cute::ceil_div(binfo.actual_seqlen_k, kLgBlockN);
    int global_n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    const int global_m_block_max = std::min(m_block_base + m_sub_loop, cute::ceil_div(binfo.actual_seqlen_q, kBlockM));
    if (Is_causal) {
        n_lg_block_max = std::min(n_lg_block_max,
                               cute::ceil_div((m_lg_block + 1) * kLgBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kLgBlockN));
        global_n_block_max = std::min(global_n_block_max,
                               cute::ceil_div((global_m_block_max + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q, kBlockN));
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if constexpr (Is_causal) {
        if (n_lg_block_max <= n_lg_block_min) {
            // const int m_block_base = m_lg_block * m_sub_loop;
            for (int m_block = m_block_base; m_block < m_block_base + m_sub_loop; ++m_block){
                if (m_block * kBlockM >= binfo.actual_seqlen_q) break;

                Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                                    + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                        make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
                Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                    make_coord(m_block, 0));  // (kBlockM, kHeadDim)

                Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

                typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
                auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
                Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
                Tensor tOrO = make_tensor<Element>(shape(tOgO));
                clear(tOrO);
                // Construct identity layout for sO
                Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
                // Repeat the partitioning with identity layouts
                Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
                Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
                if (!Is_even_K) {
                    #pragma unroll
                    for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
                }
                // Clear_OOB_K must be false since we don't want to write zeros to gmem
                FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                    gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
                );
                #pragma unroll
                for (int m = 0; m < size<1>(tOgO); ++m) {
                    const int row = get<0>(tOcO(0, m, 0));
                    if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
                }
            }
            return;
        }
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // const int m_block_base = m_lg_block * m_sub_loop;

    Element* q_head_ptr = reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + bidh * params.q_head_stride;
    
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

    const index_t row_offset_o_tmp = binfo.q_offset(params.o_tmp_batch_stride, params.o_tmp_row_stride, bidb) 
                           + (params.cu_seqlens_q == nullptr ? 0 : index_t(params.max_kblockM) * bidb) * params.h * params.d;
    Tensor mOtmp = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.o_tmp_ptr)
                                        + row_offset_o_tmp),
                        make_shape(binfo.actual_seqlen_q + params.max_kblockM, params.h, kHeadDim),
                        make_stride(params.o_tmp_row_stride, params.o_tmp_head_stride, _1{}));
    Tensor gOtmp = local_tile(mOtmp(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block_base, 0));  // (kBlockM, kHeadDim)

    ElementAccum* o_tmp_head_ptr = reinterpret_cast<ElementAccum*>(params.o_tmp_ptr) + row_offset_o_tmp + bidh * params.o_tmp_head_stride;

    // const int m_block = m_block_base;
    // const int n_block_max = 0;
    // const int n_block_min = 0;
    // add by JXGuo: ######################################################### point to start tomorrow #########################################################
    // const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
    //     + m_block_base * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;
    const index_t row_offset_p = (bidb * params.h + bidh) * params.seqlen_q_rounded * params.seqlen_k_rounded;

    Tensor mP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
        make_shape(params.seqlen_q_rounded, params.seqlen_k_rounded),
        make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    
    Tensor sOtmp = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(smem_)),
                            typename Kernel_traits::SmemLayoutO{});

    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    // assert Share_Q_K_smem == false
    Tensor sK = make_tensor(
        make_smem_ptr(reinterpret_cast<Element *>(reinterpret_cast<char *>(sQ.data().get()) + sizeof(ElementAccum) * size(sQ))), 
        typename Kernel_traits::SmemLayoutKV{});

    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
    // Shared-memory tensor to cache row indices. Start right after the space occupied by sV.
    Tensor sMobaIdx = make_tensor(
        make_smem_ptr(
            reinterpret_cast<int *>(
                reinterpret_cast<char *>(sV.data().get()) + sizeof(Element) * size(sV)
            )
        ),
        typename Kernel_traits::SmemLayoutMobaIdx{}
    );

    Tensor sMobaRowSum = make_tensor(
        make_smem_ptr(
            reinterpret_cast<ElementAccum *>(
                reinterpret_cast<char *>(sMobaIdx.data().get()) + sizeof(int) * size(sMobaIdx)
            )
        ),
        typename Kernel_traits::SmemLayoutMobaRowStats{}
    );

    Tensor sMobaRowMax = make_tensor(
        sMobaRowSum.data() + size(sMobaRowSum), 
        typename Kernel_traits::SmemLayoutMobaRowStats{}
    );

    // init sMobaRowMax and sMobaRowSum - use all threads for parallel initialization
    // const int total_elements = size(sMobaRowMax);  // 4 * kBlockM elements
    // #pragma unroll
    // for (int i = tidx; i < size(sMobaRowMax); i += blockDim.x) {
    //     sMobaRowMax.data()[i] = -INFINITY;
    //     sMobaRowSum.data()[i] = 0;
    // }

    // __syncthreads();

    // ... existing code ...
    #pragma unroll
    for (int i = tidx; i < size(sMobaRowMax); i += blockDim.x) {
        sMobaRowMax.data()[i] = -INFINITY;
        sMobaRowSum.data()[i] = 0;
    }

    __syncthreads();

    int moba_idx_len = 0;

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyOtmp gmem_tiled_copy_Otmp;
    auto gmem_thr_copy_Otmp = gmem_tiled_copy_Otmp.get_thread_slice(tidx);


    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor tOsOtmp = gmem_thr_copy_Otmp.partition_S(sOtmp);   // src: smem
    Tensor tOgOtmp = gmem_thr_copy_Otmp.partition_D(gOtmp);   // dst: gmem

    typename Kernel_traits::TiledMma tiled_mma;

    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    // Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);


    auto  smem_tiled_copy_Otmp = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomOaccum{}, tiled_mma);
    auto  smem_thr_copy_Otmp   = smem_tiled_copy_Otmp.get_thread_slice(tidx);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor cOtmp = make_identity_tensor(make_shape(size<0>(sOtmp), size<1>(sOtmp)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)


    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tOcOtmp = gmem_thr_copy_Otmp.partition_D(cOtmp);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tOpOtmp = make_tensor<bool>(make_shape(size<2>(tOgOtmp)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tOpOtmp); ++k) { tOpOtmp(k) = get<1>(tOcOtmp(0, 0, k)) < params.d; }
    }

    // Prologue

    // int n_lg_block = n_lg_block_max - 1;
    Tensor taccOcO_row = make_acc_row<kBlockM, kHeadDim>(thr_mma);
    FLASH_MOBA_NAMESPACE::MobaSoftmax<2 * size<1>(acc_o), decltype(taccOcO_row)> softmax(m_block_base * kBlockM, taccOcO_row);
    FLASH_MOBA_NAMESPACE::IndexGatherScatter<Kernel_traits> idx_gs(params, bidb, bidh, kBlockM, kBlockN);

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    FLASH_MOBA_NAMESPACE::Mask<Is_causal, /*Is_local=*/false, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    const int n_lg_masking_steps = (!Is_causal) ? 1 : cute::ceil_div(kLgBlockM, kLgBlockN) + 1;

    int n_lg_block = n_lg_block_max - 1;

    #pragma unroll
    for (int lg_masking_step = 0; lg_masking_step < n_lg_masking_steps; ++lg_masking_step, --n_lg_block) {
        auto block_indices_result = idx_gs.get_lg_block_indices(m_lg_block, n_lg_block);
        moba_idx_len = block_indices_result.count;
        load_row_indices_to_smem<kMxLgBlockM>(block_indices_result.indices_ptr, moba_idx_len, sMobaIdx);
        const int n_block_min = n_lg_block * n_sub_loop;
        int n_block_max = std::min(global_n_block_max, n_block_min + n_sub_loop);
        for (int remain_idx_len = moba_idx_len; remain_idx_len > 0; remain_idx_len -= kBlockM) {
            int cur_idx_len = std::min(remain_idx_len, kBlockM);
            const int cur_idx_offset = moba_idx_len - remain_idx_len;
            int n_block = n_block_max - 1;
            FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
                                    binfo.actual_seqlen_k - n_block * kBlockN);
            Tensor gOtmp = make_gather_tensor_Otmp<ElementAccum, kBlockM, kHeadDim>(params, o_tmp_head_ptr, cur_idx_len, &sMobaIdx(cur_idx_offset));
            o_tmp_load</*Is_even_MN=*/false, Is_even_K>(smem_tiled_copy_Otmp, smem_thr_copy_Otmp, gmem_tiled_copy_Otmp, gmem_thr_copy_Otmp, acc_o, sOtmp, gOtmp, tOcOtmp, tOpOtmp, cur_idx_len);
            // __syncthreads();
            // add by JXGuo: load Q for this lg block
            Tensor gQ = make_gather_tensor_Q<Element, kBlockM, kHeadDim>(params, q_head_ptr, cur_idx_len, &sMobaIdx(cur_idx_offset));
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            FLASH_MOBA_NAMESPACE::copy<false /*Is_even_MN*/, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, cur_idx_len);
            cute::cp_async_fence();
            for (; n_block >= n_block_min; --n_block) {
                Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
                clear(acc_s);
                FLASH_MOBA_NAMESPACE::cp_async_wait<0>();
                __syncthreads();

                if (n_block < n_block_max - 1) {
                    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
                } else {
                    // Clear the smem tiles to account for predicated off loads
                    FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
                    );
                }
                cute::cp_async_fence();
                FLASH_MOBA_NAMESPACE::gemm</*A_in_regs=*/false>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
                if constexpr (Is_softcap){
                    FLASH_MOBA_NAMESPACE::apply_softcap(acc_s, params.softcap);
                }
            
                mask.template apply_moba_mask<Is_causal, Is_even_MN>(
                    acc_s, n_block * kBlockN, (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16, sMobaIdx, cur_idx_len, cur_idx_offset
                );

                FLASH_MOBA_NAMESPACE::cp_async_wait<0>(); // add by JXGuo: to make sure the V is ready
                __syncthreads();

                if (n_block > n_block_min) {
                    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
                    cute::cp_async_fence();
                }

                softmax.template softmax_rescale_o</*Is_first=*/false, /*Is_sparse=*/true, /*Check_inf=*/Is_causal/* || Is_local*/>(acc_s, acc_o, sMobaRowMax, sMobaRowSum, sMobaIdx, cur_idx_len, cur_idx_offset, params.scale_softmax_log2);

                // Convert acc_s from fp32 to fp16/bf16
                // add by JXGuo: we need to modify this
                Tensor rP = FLASH_MOBA_NAMESPACE::convert_type<Element>(acc_s);
                if constexpr (Return_softmax || Is_dropout) {
                    int block_row_idx = m_block_base * (kBlockM / 16) + tidx / 32;
                    int block_col_idx = n_block * (kBlockN / 32);
                    if (Return_softmax) {
                        Tensor rP_drop = make_fragment_like(rP);
                        cute::copy(rP, rP_drop);
                        dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                            rP_drop, block_row_idx, block_col_idx, kNWarps
                        );
                        Tensor gP = local_tile(mP, Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_coord(m_block_base, n_block));
                        Tensor tSgS  = thr_mma.partition_C(gP);
                        cute::copy(rP_drop, tSgS);
                    }
                    if (Is_dropout) {
                        dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
                    }
                }
                Tensor tOrP = make_tensor(rP.data(), FLASH_MOBA_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
                FLASH_MOBA_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            }
            o_tmp_store</*Is_even_MN=*/false, Is_even_K>(smem_tiled_copy_Otmp, smem_thr_copy_Otmp, gmem_tiled_copy_Otmp, gmem_thr_copy_Otmp, acc_o, sOtmp, gOtmp, tOcOtmp, tOpOtmp, cur_idx_len);
        }
        if (n_lg_masking_steps > 1 && n_lg_block <= n_lg_block_min) {
            --n_lg_block;
            break;
        }
    }


    for (; n_lg_block >= n_lg_block_min; --n_lg_block) {
        auto block_indices_result = idx_gs.get_lg_block_indices(m_lg_block, n_lg_block);
        moba_idx_len = block_indices_result.count;
        load_row_indices_to_smem<kMxLgBlockM>(block_indices_result.indices_ptr, moba_idx_len, sMobaIdx);
        const int n_block_min = n_lg_block * n_sub_loop;
        const int n_block_max = n_block_min + n_sub_loop;
        for (int remain_idx_len = moba_idx_len; remain_idx_len > 0; remain_idx_len -= kBlockM) {
            int cur_idx_len = std::min(remain_idx_len, kBlockM);
            const int cur_idx_offset = moba_idx_len - remain_idx_len;
            int n_block = n_block_max - 1;
            FLASH_MOBA_NAMESPACE::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV);
            Tensor gOtmp = make_gather_tensor_Otmp<ElementAccum, kBlockM, kHeadDim>(params, o_tmp_head_ptr, cur_idx_len, &sMobaIdx(cur_idx_offset));
            o_tmp_load</*Is_even_MN=*/false, Is_even_K>(smem_tiled_copy_Otmp, smem_thr_copy_Otmp, gmem_tiled_copy_Otmp, gmem_thr_copy_Otmp, acc_o, sOtmp, gOtmp, tOcOtmp, tOpOtmp, cur_idx_len);
            // __syncthreads();
            // add by JXGuo: load Q for this lg block
            Tensor gQ = make_gather_tensor_Q<Element, kBlockM, kHeadDim>(params, q_head_ptr, cur_idx_len, &sMobaIdx(cur_idx_offset));
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            FLASH_MOBA_NAMESPACE::copy<false /*Is_even_MN*/, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, cur_idx_len);
            cute::cp_async_fence();
            for (; n_block >= n_block_min; --n_block) {
                Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
                clear(acc_s);
                FLASH_MOBA_NAMESPACE::cp_async_wait<0>();
                __syncthreads();

                FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

                cute::cp_async_fence();
                FLASH_MOBA_NAMESPACE::gemm</*A_in_regs=*/false>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
                if constexpr (Is_softcap){
                    FLASH_MOBA_NAMESPACE::apply_softcap(acc_s, params.softcap);
                }
            
                mask.template apply_moba_mask</*Causal_mask=*/false>(
                    acc_s, n_block * kBlockN, (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16, sMobaIdx, cur_idx_len, cur_idx_offset
                );

                FLASH_MOBA_NAMESPACE::cp_async_wait<0>(); // add by JXGuo: to make sure the V is ready
                __syncthreads();

                if (n_block > n_block_min) {
                    FLASH_MOBA_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
                    cute::cp_async_fence();
                }

                softmax.template softmax_rescale_o</*Is_first=*/false, /*Is_sparse=*/true, /*Check_inf=*/false/* || Is_local*/>(acc_s, acc_o, sMobaRowMax, sMobaRowSum, sMobaIdx, cur_idx_len, cur_idx_offset, params.scale_softmax_log2); // add by JXGuo: check the true here

                // Convert acc_s from fp32 to fp16/bf16
                // add by JXGuo: we need to modify this
                Tensor rP = FLASH_MOBA_NAMESPACE::convert_type<Element>(acc_s);
                if constexpr (Return_softmax || Is_dropout) {
                    int block_row_idx = m_block_base * (kBlockM / 16) + tidx / 32;
                    int block_col_idx = n_block * (kBlockN / 32);
                    if (Return_softmax) {
                        Tensor rP_drop = make_fragment_like(rP);
                        cute::copy(rP, rP_drop);
                        dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                            rP_drop, block_row_idx, block_col_idx, kNWarps
                        );
                        Tensor gP = local_tile(mP, Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_coord(m_block_base, n_block));
                        Tensor tSgS  = thr_mma.partition_C(gP);
                        cute::copy(rP_drop, tSgS);
                    }
                    if (Is_dropout) {
                        dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
                    }
                }

                Tensor tOrP = make_tensor(rP.data(), FLASH_MOBA_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
                FLASH_MOBA_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            }
            o_tmp_store</*Is_even_MN=*/false, Is_even_K>(smem_tiled_copy_Otmp, smem_thr_copy_Otmp, gmem_tiled_copy_Otmp, gmem_thr_copy_Otmp, acc_o, sOtmp, gOtmp, tOcOtmp, tOpOtmp, cur_idx_len);
        }

    }

    // add by JXGuo: epilogue
    for (int m_block = m_block_base; m_block < global_m_block_max; ++m_block){
        Tensor gOtmp = local_tile(mOtmp(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));
        o_tmp_load<Is_even_MN/*Is_even_MN=*/, Is_even_K>(smem_tiled_copy_Otmp, smem_thr_copy_Otmp, gmem_tiled_copy_Otmp, gmem_thr_copy_Otmp, acc_o, sOtmp, gOtmp, tOcOtmp, tOpOtmp, binfo.actual_seqlen_q - m_block * kBlockM);
        const int cur_idx_offset = (m_block - m_block_base) * kBlockM;
        Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, sMobaRowMax, sMobaRowSum, cur_idx_offset, params.scale_softmax, params.rp_dropout);
        Tensor rO = FLASH_MOBA_NAMESPACE::convert_type<Element>(acc_o);
        Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);  // add by JXGuo: SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

        if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }
        // add by JXGuo: copy from register to smem
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                                + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

        __syncthreads();

        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_MOBA_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
    }



}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, /*bool Is_local,*/ bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_moba_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    FLASH_MOBA_NAMESPACE::compute_moba_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
    
    // FLASH_MOBA_NAMESPACE::compute_attn_1colblock_epilogue<Kernel_traits, Is_dropout, Is_even_MN, Is_even_K>(params, bidb, bidh);

    // FLASH_MOBA_NAMESPACE::compute_attn_1colblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

} // namespace FLASH_MOBA_NAMESPACE
